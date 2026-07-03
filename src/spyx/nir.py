import jax
import jax.numpy as jnp
import nir
import numpy as np
from flax import nnx

from .nn import (
    IF,
    LIF,
    RIF,
    RLIF,
    CuBaLIF,
    Flatten,
    RCuBaLIF,
    Sequential,
    SumPool,
    run,
)


def reorder_layers(init_params, trained_params):
    """
    Some optimization libraries may permute the keys of the network's PyTree;
    this is an issue as exporting to NIR assumes the keys are in their original order
    after initializing the network. This simple function takes the original and trained parameters
    and returns the trained parameters in the proper order for exportation.
    """
    return {k: trained_params[k] for k in init_params.keys()}


def _create_rnn_subgraph(graph: nir.NIRGraph, lif_nk: str, w_nk: str) -> nir.NIRGraph:
    """Take a NIRGraph plus the node keys for a LIF and a W_rec, and return a new NIRGraph
    which has the RNN subgraph replaced with a subgraph (i.e., a single NIRGraph node).
    """
    sg_key = lif_nk.split(".")[0]

    sg_edges = [
        (lif_nk, w_nk),
        (w_nk, lif_nk),
        (lif_nk, f"{sg_key}.output"),
        (f"{sg_key}.input", w_nk),
    ]
    sg_nodes = {
        lif_nk: graph.nodes[lif_nk],
        w_nk: graph.nodes[w_nk],
        f"{sg_key}.input": nir.Input(graph.nodes[lif_nk].input_type),  # ty: ignore[unresolved-attribute]  # untyped NIR node
        f"{sg_key}.output": nir.Output(graph.nodes[lif_nk].output_type),  # ty: ignore[unresolved-attribute]  # untyped NIR node
    }
    sg = nir.NIRGraph(nodes=sg_nodes, edges=sg_edges)

    graph.edges = [e for e in graph.edges if e not in [(lif_nk, w_nk), (w_nk, lif_nk)]]
    graph.nodes = {k: v for k, v in graph.nodes.items() if k not in [lif_nk, w_nk]}

    graph.edges = [(e[0], sg_key) if e[1] == lif_nk else e for e in graph.edges]
    graph.edges = [(sg_key, e[1]) if e[0] == lif_nk else e for e in graph.edges]

    graph.nodes[sg_key] = sg
    return graph


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> nir.NIRGraph:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    if len(set(graph.edges)) != len(graph.edges):
        print("[WARNING] duplicate edges found, removing")
        graph.edges = list(set(graph.edges))

    for edge1 in graph.edges:
        for edge2 in graph.edges:
            if not edge1 == edge2:
                if edge1[0] == edge2[1] and edge1[1] == edge2[0]:
                    lif_nk = edge1[0]
                    lif_n = graph.nodes[lif_nk]
                    w_nk = edge1[1]
                    w_n = graph.nodes[w_nk]
                    is_lif = isinstance(lif_n, (nir.LIF, nir.CubaLIF))
                    is_dense = isinstance(w_n, (nir.Affine, nir.Linear))
                    w_out_nk = [e[1] for e in graph.edges if e[0] == w_nk]
                    w_in_nk = [e[0] for e in graph.edges if e[1] == w_nk]
                    is_rnn = len(w_out_nk) == 1 and len(w_in_nk) == 1
                    if is_rnn and is_lif and is_dense:
                        print("[INFO] found RNN subgraph, replacing with NIRGraph node")
                        graph = _create_rnn_subgraph(graph, edge1[0], edge1[1])
    return graph


def _parse_rnn_subgraph(graph: nir.NIRGraph) -> tuple:
    sub_nodes = graph.nodes.values()
    assert len(sub_nodes) == 4, "only 4-node RNN allowed in subgraph"
    try:
        input_node = [n for n in sub_nodes if isinstance(n, nir.Input)][0]
        output_node = [n for n in sub_nodes if isinstance(n, nir.Output)][0]
        lif_node = [n for n in sub_nodes if isinstance(n, (nir.LIF, nir.CubaLIF))][0]
        wrec_node = [n for n in sub_nodes if isinstance(n, (nir.Affine, nir.Linear))][0]
    except IndexError:
        raise ValueError(
            "invalid RNN subgraph - could not find all required nodes"
        ) from None
    lif_size = list(input_node.input_type.values())[0][0]
    assert lif_size == list(output_node.output_type.values())[0][0], (
        "output size mismatch"
    )
    assert lif_size == lif_node.v_threshold.size, "lif size mismatch (v_threshold)"
    assert lif_size == wrec_node.weight.shape[0], "w_rec shape mismatch"
    assert lif_size == wrec_node.weight.shape[1], "w_rec shape mismatch"

    return lif_node, wrec_node, lif_size


# --- shape conventions -------------------------------------------------------
# NIR tensors are channels-first (C, H, W); spyx / NNX run channels-last
# (B, H, W, C). A spyx neuron after a conv therefore holds channels-last
# (H, W, C) state/parameters. These helpers bridge the two so that
# convolutional models round-trip.


def _spyx_param_to_nir(param, nir_shape):
    """spyx neuron param -> NIR array of ``nir_shape`` (channels-first).

    A scalar fills the shape; a channels-last (H, W, C) param is transposed to
    channels-first (C, H, W); a 1-D (dense) param is returned unchanged.
    """
    p = np.asarray(param, dtype=np.float32)
    nir_shape = tuple(int(d) for d in nir_shape)
    if p.ndim == 0:
        return np.full(nir_shape, float(p), dtype=np.float32)
    if len(nir_shape) == 3:  # (H, W, C) -> (C, H, W)
        return p.transpose(2, 0, 1).astype(np.float32)
    return p.astype(np.float32)


def _nir_to_spyx_shape(shape):
    """NIR channels-first (C, H, W) -> spyx channels-last (H, W, C); 1-D unchanged."""
    shape = tuple(int(d) for d in shape)
    return (shape[1], shape[2], shape[0]) if len(shape) == 3 else shape


def _nir_to_spyx_param(arr):
    """Transpose a channels-first (C, H, W) NIR param to channels-last (H, W, C)."""
    a = np.asarray(arr)
    return a.transpose(1, 2, 0) if a.ndim == 3 else a


def _nir_node_to_spyx_module(node, rngs: nnx.Rngs):
    """Converts a single NIR node to a Spyx/NNX module."""

    if isinstance(node, (nir.Input, nir.Output)):
        return None

    elif isinstance(node, nir.Affine):
        return nnx.Linear(node.weight.shape[1], node.weight.shape[0], rngs=rngs)

    elif isinstance(node, nir.Linear):
        return nnx.Linear(
            node.weight.shape[1], node.weight.shape[0], use_bias=False, rngs=rngs
        )

    elif isinstance(node, nir.Conv2d):
        # nir stores 'same'/'valid' as a lowercase string, or an int pair.
        pad = node.padding
        if isinstance(pad, str):
            padding = pad.upper()  # nnx.Conv wants 'SAME' / 'VALID'
        else:
            p0, p1 = int(pad[0]), int(pad[1])
            padding = [(p0, p0), (p1, p1)]
        stride = np.atleast_1d(np.asarray(node.stride)).tolist()
        strides = tuple(int(s) for s in (stride * 2 if len(stride) == 1 else stride))
        return nnx.Conv(
            in_features=int(node.weight.shape[1]),  # OIHW
            out_features=int(node.weight.shape[0]),
            kernel_size=tuple(int(k) for k in node.weight.shape[-2:]),
            strides=strides,
            padding=padding,
            use_bias=node.bias is not None,
            rngs=rngs,
        )

    elif isinstance(node, nir.SumPool2d):
        return SumPool(node.kernel_size, node.stride.tolist(), "VALID", channel_axis=1)

    elif isinstance(node, nir.IF):
        return IF(
            _nir_to_spyx_shape(node.r.shape),
            threshold=_nir_to_spyx_param(node.v_threshold),
        )

    elif isinstance(node, nir.LIF):
        return LIF(
            _nir_to_spyx_shape(node.tau.shape),
            threshold=_nir_to_spyx_param(node.v_threshold),
            rngs=rngs,
        )

    elif isinstance(node, nir.CubaLIF):
        return CuBaLIF(
            _nir_to_spyx_shape(node.tau_syn.shape),
            threshold=_nir_to_spyx_param(node.v_threshold),
            rngs=rngs,
        )

    elif isinstance(node, nir.Flatten):
        return Flatten()

    elif isinstance(node, nir.NIRGraph):
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)
        if isinstance(lif_node, nir.IF):
            # RIF inherits its spike threshold from v_threshold, matching the
            # non-recurrent IF path above (nir.IF has no v_leak attribute).
            return RIF((lif_size,), threshold=lif_node.v_threshold, rngs=rngs)
        elif isinstance(lif_node, nir.LIF):
            return RLIF((lif_size,), threshold=lif_node.v_threshold, rngs=rngs)
        else:
            return RCuBaLIF((lif_size,), threshold=lif_node.v_threshold, rngs=rngs)

    return None


def _build_model(nir_graph: nir.NIRGraph, dt: float, rngs: nnx.Rngs) -> Sequential:
    """Reconstruct the Spyx/NNX Sequential from a NIR graph (no execution)."""

    nir_graph = _replace_rnn_subgraph_with_nirgraph(nir_graph)

    # Simple linear ordering for now as per original.
    # In a real graph, we'd need a more complex assembly.

    modules = []
    node_keys = []

    # We'll use a simple sequential model based on the node order in nir_graph.edges
    # This is a simplification. Original used ordered_edge_list.
    # I'll keep the ordered_edge_list logic.

    def _find_tuple_with_first_element(lst, value):
        return next((tup for tup in lst if tup[0] == value), None)

    def _order_edge_list(edge_list):
        curr_node, next_node = "input", None
        ordered_list = []
        while next_node != "output":
            tup = _find_tuple_with_first_element(edge_list, curr_node)
            if tup is None:
                break
            ordered_list.append(tup)
            next_node = tup[1]
            curr_node = next_node
        return ordered_list

    sorted_edges = _order_edge_list(nir_graph.edges)

    for _i, (src, _dst) in enumerate(sorted_edges):
        if src == "input":
            continue
        node = nir_graph.nodes[src]
        mod = _nir_node_to_spyx_module(node, rngs)
        if mod:
            modules.append(mod)
            node_keys.append(src)

            # Parameter loading
            if isinstance(node, nir.Affine):
                mod.kernel[...] = jnp.array(node.weight.T)
                mod.bias[...] = jnp.array(node.bias)
            elif isinstance(node, nir.Linear):
                mod.kernel[...] = jnp.array(node.weight.T)
            elif isinstance(node, nir.Conv2d):
                # NIR weight is OIHW; NNX Conv kernel is HWIO.
                weight = node.weight.transpose((2, 3, 1, 0))
                mod.kernel[...] = jnp.array(weight)
                if node.bias is not None:
                    mod.bias[...] = jnp.array(node.bias)
            elif isinstance(node, nir.LIF):
                mod.beta[...] = jnp.array(_nir_to_spyx_param(1 - (dt / node.tau)))
            elif isinstance(node, nir.CubaLIF):
                mod.alpha[...] = jnp.array(_nir_to_spyx_param(1 - (dt / node.tau_syn)))
                mod.beta[...] = jnp.array(_nir_to_spyx_param(1 - (dt / node.tau_mem)))
            elif isinstance(node, nir.NIRGraph):
                lif_node, wrec_node, _ = _parse_rnn_subgraph(node)
                # NIR weight is (out, in); spyx recurrent_w is (hidden, hidden).
                # For recurrent layers in_features == out_features, but transpose
                # anyway to honour the (in, out) convention shared with nnx.Linear.
                mod.recurrent_w[...] = jnp.array(wrec_node.weight.T)
                if isinstance(lif_node, nir.LIF):
                    mod.beta[...] = jnp.array(1 - (dt / lif_node.tau))
                elif isinstance(lif_node, nir.CubaLIF):
                    mod.alpha[...] = jnp.array(1 - (dt / lif_node.tau_syn))
                    mod.beta[...] = jnp.array(1 - (dt / lif_node.tau_mem))
                # nir.IF carries no tau parameter; nothing to load for RIF.

    return Sequential(*modules)


def from_nir(
    nir_graph: nir.NIRGraph,
    input_data,
    dt: float = 1,
    *,
    return_all_states: bool = False,
    rngs: nnx.Rngs | None = None,
):
    """Reconstruct a Spyx/NNX model from a NIR graph and run it on ``input_data``.

    :param nir_graph: the NIR graph to import.
    :param input_data: time-major input, shape ``(T, B, ...)``; scanned over the
        leading time axis.
    :param dt: simulation timestep used to convert NIR time constants back to
        Spyx decay factors (must match the ``dt`` used on export).
    :param return_all_states: when True, also return the per-layer neuron states
        at *every* timestep (e.g. membrane-potential traces), as a pytree of
        ``(T, B, ...)`` arrays mirroring ``model.initial_state``.
    :param rngs: optional ``nnx.Rngs`` for reconstructing the modules.
    :return: ``(model, outputs)`` where ``outputs`` is ``(T, B, ...)``; or
        ``(model, (outputs, states))`` when ``return_all_states`` is True.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    model = _build_model(nir_graph, dt, rngs)

    if not return_all_states:
        outputs, _ = run(model, input_data)
        return model, outputs

    # Capture the per-layer state at each timestep (membrane traces, etc.).
    init_state = model.initial_state(input_data.shape[1])

    def _step(state, x_t):
        out, new_state = model(x_t, state)
        return new_state, (out, new_state)

    _, (outputs, states) = jax.lax.scan(_step, init_state, input_data)
    return model, (outputs, states)


def _spyx_recurrent_to_nirgraph(layer, node_key, dt) -> nir.NIRGraph:
    """Build the inner NIRGraph subgraph for an RIF / RLIF / RCuBaLIF layer.

    The subgraph mirrors the (input -> wrec, lif <-> wrec, lif -> output)
    topology produced by ``_replace_rnn_subgraph_with_nirgraph`` so that a
    Spyx -> NIR -> Spyx roundtrip is symmetric.
    """
    hidden_shape = layer.hidden_shape
    threshold = np.full(hidden_shape, layer.threshold)
    v_leak = np.zeros(hidden_shape)

    if isinstance(layer, RIF):
        lif = nir.IF(r=np.ones(hidden_shape), v_threshold=threshold)
    elif isinstance(layer, RLIF):
        beta = np.array(layer.beta[...])
        if beta.ndim == 0:
            beta = np.full(hidden_shape, beta)
        lif = nir.LIF(
            tau=dt / (1 - beta),
            v_threshold=threshold,
            v_leak=v_leak,
            r=beta,
        )
    else:  # RCuBaLIF
        alpha = np.array(layer.alpha[...])
        beta = np.array(layer.beta[...])
        if alpha.ndim == 0:
            alpha = np.full(hidden_shape, alpha)
        if beta.ndim == 0:
            beta = np.full(hidden_shape, beta)
        lif = nir.CubaLIF(
            tau_mem=dt / (1 - beta),
            tau_syn=dt / (1 - alpha),
            v_threshold=threshold,
            v_leak=v_leak,
            r=beta,
        )

    # NIR Linear weight is (out, in); spyx recurrent_w is (in, out).
    wrec = nir.Linear(weight=np.array(layer.recurrent_w[...]).T)

    lif_nk = f"{node_key}.lif"
    w_nk = f"{node_key}.w_rec"
    sub_nodes = {
        lif_nk: lif,
        w_nk: wrec,
        f"{node_key}.input": nir.Input(input_type={"input": np.array(hidden_shape)}),
        f"{node_key}.output": nir.Output(
            output_type={"output": np.array(hidden_shape)}
        ),
    }
    sub_edges = [
        (f"{node_key}.input", w_nk),
        (w_nk, lif_nk),
        (lif_nk, w_nk),
        (lif_nk, f"{node_key}.output"),
    ]
    return nir.NIRGraph(nodes=sub_nodes, edges=sub_edges)


def to_nir(model, input_shape, output_shape, dt=1) -> nir.NIRGraph:
    """Converts a Spyx/NNX model to a NIR graph."""

    nodes = {"input": nir.Input(input_shape), "output": nir.Output(output_shape)}
    edges = []

    prev_node = "input"

    # We assume a sequential model for now
    if not isinstance(model, nnx.Sequential):
        layers = [model]
    else:
        layers = model.layers

    # Track the per-sample tensor shape (NIR shapes exclude the batch axis) so
    # nodes that need it — Conv2d (spatial dims) and Flatten (full shape) — can
    # be constructed correctly. NIR uses channels-first (C, N_x, N_y).
    _in = next(iter(input_shape.values()))
    cur_shape = tuple(int(d) for d in np.ravel(np.asarray(_in)))

    for i, layer in enumerate(layers):
        node_key = f"layer_{i}"

        if isinstance(layer, nnx.Linear):
            if layer.bias is not None:
                nodes[node_key] = nir.Affine(
                    np.array(layer.kernel[...].T), np.array(layer.bias[...])
                )
            else:
                nodes[node_key] = nir.Linear(np.array(layer.kernel[...].T))
            cur_shape = cur_shape[:-1] + (int(layer.kernel.shape[-1]),)

        elif isinstance(layer, nnx.Conv):
            # NNX Conv is HWIO, NIR is OIHW
            weight = np.array(layer.kernel[...].transpose((3, 2, 0, 1)))
            spatial = tuple(cur_shape[-2:])  # (N_x, N_y)
            nodes[node_key] = nir.Conv2d(
                input_shape=spatial,  # required to disambiguate the shape
                weight=weight,
                bias=np.array(layer.bias[...]) if layer.bias is not None else None,
                dilation=1,  # Default
                stride=layer.strides,
                padding="same",  # nir expects lowercase 'same' / 'valid'
                groups=1,
            )
            # 'same' padding preserves the spatial dims; channels -> out_features.
            cur_shape = (int(layer.kernel.shape[-1]), *spatial)

        elif isinstance(layer, IF):
            # nir.IF requires array-valued r / v_threshold shaped to the layer.
            # cur_shape carries spatial dims when the neuron follows a conv.
            nodes[node_key] = nir.IF(
                r=np.ones(cur_shape, dtype=np.float32),
                v_threshold=_spyx_param_to_nir(layer.threshold, cur_shape),
            )

        elif isinstance(layer, LIF):
            beta = _spyx_param_to_nir(layer.beta[...], cur_shape)
            nodes[node_key] = nir.LIF(
                tau=dt / (1 - beta),
                v_threshold=_spyx_param_to_nir(layer.threshold, cur_shape),
                v_leak=np.zeros(cur_shape, dtype=np.float32),
                r=beta,
            )

        elif isinstance(layer, CuBaLIF):
            alpha = _spyx_param_to_nir(layer.alpha[...], cur_shape)
            beta = _spyx_param_to_nir(layer.beta[...], cur_shape)
            nodes[node_key] = nir.CubaLIF(
                tau_mem=dt / (1 - beta),
                tau_syn=dt / (1 - alpha),
                v_threshold=_spyx_param_to_nir(layer.threshold, cur_shape),
                v_leak=np.zeros(cur_shape, dtype=np.float32),
                r=beta,
            )

        elif isinstance(layer, (RIF, RLIF, RCuBaLIF)):
            nodes[node_key] = _spyx_recurrent_to_nirgraph(layer, node_key, dt)

        elif isinstance(layer, Flatten):
            # spyx.nn.Flatten collapses every non-batch dim; NIR shapes have no
            # batch axis, so flatten the whole per-sample shape (start_dim=0).
            nodes[node_key] = nir.Flatten(input_type=cur_shape, start_dim=0, end_dim=-1)
            cur_shape = (int(np.prod(cur_shape)),)

        else:
            print(
                f"[Warning] Layer {type(layer)} not recognized/supported for NIR export."
            )
            continue

        edges.append((prev_node, node_key))
        prev_node = node_key

    edges.append((prev_node, "output"))

    return nir.NIRGraph(nodes, edges)
