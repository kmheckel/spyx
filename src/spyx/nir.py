import jax.numpy as jnp
import nir
import numpy as np
from flax import nnx

from .nn import IF, LIF, RIF, RLIF, CuBaLIF, RCuBaLIF, Sequential, SumPool


def reorder_layers(init_params, trained_params):
    """
    Some optimization libraries may permute the keys of the network's PyTree; 
    this is an issue as exporting to NIR assumes the keys are in their original order
    after initializing the network. This simple function takes the original and trained parameters
    and returns the trained parameters in the proper order for exportation.
    """
    return {k:trained_params[k] for k in init_params.keys() }

def _create_rnn_subgraph(graph: nir.NIRGraph, lif_nk: str, w_nk: str) -> nir.NIRGraph:
    """Take a NIRGraph plus the node keys for a LIF and a W_rec, and return a new NIRGraph
    which has the RNN subgraph replaced with a subgraph (i.e., a single NIRGraph node).
    """
    sg_key = lif_nk.split('.')[0]

    sg_edges = [
        (lif_nk, w_nk), (w_nk, lif_nk), (lif_nk, f'{sg_key}.output'), (f'{sg_key}.input', w_nk)
    ]
    sg_nodes = {
        lif_nk: graph.nodes[lif_nk],
        w_nk: graph.nodes[w_nk],
        f'{sg_key}.input': nir.Input(graph.nodes[lif_nk].input_type),
        f'{sg_key}.output': nir.Output(graph.nodes[lif_nk].output_type),
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
        print('[WARNING] duplicate edges found, removing')
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
                        print('[INFO] found RNN subgraph, replacing with NIRGraph node')
                        graph = _create_rnn_subgraph(graph, edge1[0], edge1[1])
    return graph


def _parse_rnn_subgraph(graph: nir.NIRGraph) -> tuple:
    sub_nodes = graph.nodes.values()
    assert len(sub_nodes) == 4, 'only 4-node RNN allowed in subgraph'
    try:
        input_node = [n for n in sub_nodes if isinstance(n, nir.Input)][0]
        output_node = [n for n in sub_nodes if isinstance(n, nir.Output)][0]
        lif_node = [n for n in sub_nodes if isinstance(n, (nir.LIF, nir.CubaLIF))][0]
        wrec_node = [n for n in sub_nodes if isinstance(n, (nir.Affine, nir.Linear))][0]
    except IndexError:
        raise ValueError('invalid RNN subgraph - could not find all required nodes') from None
    lif_size = list(input_node.input_type.values())[0][0]
    assert lif_size == list(output_node.output_type.values())[0][0], 'output size mismatch'
    assert lif_size == lif_node.v_threshold.size, 'lif size mismatch (v_threshold)'
    assert lif_size == wrec_node.weight.shape[0], 'w_rec shape mismatch'
    assert lif_size == wrec_node.weight.shape[1], 'w_rec shape mismatch'

    return lif_node, wrec_node, lif_size


def _nir_node_to_spyx_module(node, rngs: nnx.Rngs):
    """Converts a single NIR node to a Spyx/NNX module."""

    if isinstance(node, (nir.Input, nir.Output)):
        return None

    elif isinstance(node, nir.Affine):
        return nnx.Linear(node.weight.shape[1], node.weight.shape[0], rngs=rngs)

    elif isinstance(node, nir.Linear):
        return nnx.Linear(node.weight.shape[1], node.weight.shape[0], use_bias=False, rngs=rngs)

    elif isinstance(node, nir.Conv2d):
        p0, p1 = node.padding[0], node.padding[1]
        return nnx.Conv(
            in_features=node.weight.shape[1],
            out_features=node.weight.shape[0],
            kernel_size=node.weight.shape[-1],
            strides=node.stride.tolist(),
            padding=[(p0, p0), (p1, p1)],
            rngs=rngs
        )

    elif isinstance(node, nir.SumPool2d):
        return SumPool(
            node.kernel_size, node.stride.tolist(), "VALID", channel_axis=1
        )

    elif isinstance(node, nir.IF):
        return IF(node.r.shape, threshold=node.v_threshold)

    elif isinstance(node, nir.LIF):
        return LIF(node.tau.shape, threshold=node.v_threshold, rngs=rngs)

    elif isinstance(node, nir.CubaLIF):
        return CuBaLIF(node.tau_syn.shape, threshold=node.v_threshold, rngs=rngs)

    elif isinstance(node, nir.Flatten):
        return nnx.Flatten()

    elif isinstance(node, nir.NIRGraph):
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)
        if isinstance(lif_node, nir.IF):
            return RIF((lif_size,), threshold=lif_node.v_leak, rngs=rngs)
        elif isinstance(lif_node, nir.LIF):
            return RLIF((lif_size,), threshold=lif_node.v_threshold, rngs=rngs)
        else:
            return RCuBaLIF((lif_size,), threshold=lif_node.v_threshold, rngs=rngs)

    return None


def from_nir(nir_graph: nir.NIRGraph, dt: float, rngs: nnx.Rngs = None):
    """Converts a NIR graph to a Spyx/NNX model."""
    if rngs is None:
        rngs = nnx.Rngs(0)
        
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
                # HWIO format for NNX Conv
                weight = node.weight.transpose((2, 3, 1, 0))
                mod.kernel[...] = jnp.array(weight)
                mod.bias[...] = jnp.array(node.bias)
            elif isinstance(node, nir.LIF):
                mod.beta[...] = jnp.array(1 - (dt / node.tau))
            elif isinstance(node, nir.CubaLIF):
                mod.alpha[...] = jnp.array(1 - (dt / node.tau_syn))
                mod.beta[...] = jnp.array(1 - (dt / node.tau_mem))
            elif isinstance(node, nir.NIRGraph):
                lif_node, wrec_node, _ = _parse_rnn_subgraph(node)
                # ... handle recurrent weights ...
                # This needs more care to match weights to the new modules.
                pass
                
    return Sequential(*modules)

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
        
    for i, layer in enumerate(layers):
        node_key = f"layer_{i}"
        
        if isinstance(layer, nnx.Linear):
            if layer.bias is not None:
                nodes[node_key] = nir.Affine(np.array(layer.kernel[...].T), np.array(layer.bias[...]))
            else:
                nodes[node_key] = nir.Linear(np.array(layer.kernel[...].T))
        
        elif isinstance(layer, nnx.Conv):
            # NNX Conv is HWIO, NIR is OIHW
            weight = np.array(layer.kernel[...].transpose((3, 2, 0, 1)))
            nodes[node_key] = nir.Conv2d(
                weight=weight,
                bias=np.array(layer.bias[...]) if layer.bias is not None else None,
                dilation=1, # Default
                stride=layer.strides,
                padding="SAME", # Default
                groups=1
            )
            
        elif isinstance(layer, IF):
            nodes[node_key] = nir.IF(r=1, v_threshold=np.array(layer.threshold))
            
        elif isinstance(layer, LIF):
            beta = np.array(layer.beta[...])
            if beta.ndim == 0:
                beta = np.full(layer.hidden_shape, beta)
            nodes[node_key] = nir.LIF(
                tau=dt / (1 - beta),
                v_threshold=np.full(layer.hidden_shape, layer.threshold),
                v_leak=np.zeros(layer.hidden_shape),
                r=beta
            )
            
        elif isinstance(layer, CuBaLIF):
            alpha = np.array(layer.alpha[...])
            beta = np.array(layer.beta[...])
            if alpha.ndim == 0:
                alpha = np.full(layer.hidden_shape, alpha)
            if beta.ndim == 0:
                beta = np.full(layer.hidden_shape, beta)
            nodes[node_key] = nir.CubaLIF(
                tau_mem=dt / (1 - beta),
                tau_syn=dt / (1 - alpha),
                v_threshold=np.full(layer.hidden_shape, layer.threshold),
                v_leak=np.zeros(layer.hidden_shape),
                r=beta
            )
            
        elif isinstance(layer, nnx.Flatten):
            nodes[node_key] = nir.Flatten(input_type={"input": input_shape}) # Simplified
            
        else:
            print(f"[Warning] Layer {type(layer)} not recognized/supported for NIR export.")
            continue
            
        edges.append((prev_node, node_key))
        prev_node = node_key
        
    edges.append((prev_node, "output"))
    
    return nir.NIRGraph(nodes, edges)
