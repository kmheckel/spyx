import jax
import jax.numpy as jnp

import nir
import numpy as np
import haiku as hk

from .nn import IF
from .nn import LIF
from .nn import CuBaLIF

from .nn import RIF
from .nn import RLIF
from .nn import RCuBaLIF

from .nn import SumPool

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
    # NOTE: assuming that the LIF and W_rec have keys of form `xyz.abc`
    sg_key = lif_nk.split('.')[0]  # TODO: make this more general?

    # create subgraph for RNN
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

    # remove subgraph edges from graph
    graph.edges = [e for e in graph.edges if e not in [(lif_nk, w_nk), (w_nk, lif_nk)]]
    # remove subgraph nodes from graph
    graph.nodes = {k: v for k, v in graph.nodes.items() if k not in [lif_nk, w_nk]}

    # change edges of type (x, lif_nk) to (x, sg_key)
    graph.edges = [(e[0], sg_key) if e[1] == lif_nk else e for e in graph.edges]
    # change edges of type (lif_nk, x) to (sg_key, x)
    graph.edges = [(sg_key, e[1]) if e[0] == lif_nk else e for e in graph.edges]

    # insert subgraph into graph and return
    graph.nodes[sg_key] = sg
    return graph


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> nir.NIRGraph:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    if len(set(graph.edges)) != len(graph.edges):
        print('[WARNING] duplicate edges found, removing')
        graph.edges = list(set(graph.edges))

    # find cycle of LIF <> Dense nodes
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
                    # check if the dense only connects to the LIF
                    w_out_nk = [e[1] for e in graph.edges if e[0] == w_nk]
                    w_in_nk = [e[0] for e in graph.edges if e[1] == w_nk]
                    is_rnn = len(w_out_nk) == 1 and len(w_in_nk) == 1
                    # check if we found an RNN - if so, then parse it
                    if is_rnn and is_lif and is_dense:
                        print('[INFO] found RNN subgraph, replacing with NIRGraph node')
                        print(f'[INFO] subgraph edges: {edge1}, {edge2}')
                        graph = _create_rnn_subgraph(graph, edge1[0], edge1[1])
    return graph


def _parse_rnn_subgraph(graph: nir.NIRGraph) -> (nir.NIRNode, nir.NIRNode, int):
    """Try parsing the graph as a RNN subgraph.

    Assumes four nodes: Input, Output, LIF | CubaLIF, Affine | Linear
    Checks that all nodes have consistent shapes.
    Will throw an error if either not all nodes are found or consistent shapes are found.

    Returns:
        lif_node: LIF | CubaLIF node
        wrec_node: Affine | Linear node
        lif_size: int, number of neurons in the RNN
    """
    sub_nodes = graph.nodes.values()
    assert len(sub_nodes) == 4, 'only 4-node RNN allowed in subgraph'
    try:
        input_node = [n for n in sub_nodes if isinstance(n, nir.Input)][0]
        output_node = [n for n in sub_nodes if isinstance(n, nir.Output)][0]
        lif_node = [n for n in sub_nodes if isinstance(n, (nir.LIF, nir.CubaLIF))][0]
        wrec_node = [n for n in sub_nodes if isinstance(n, (nir.Affine, nir.Linear))][0]
    except IndexError:
        raise ValueError('invalid RNN subgraph - could not find all required nodes')
    lif_size = list(input_node.input_type.values())[0][0]
    assert lif_size == list(output_node.output_type.values())[0][0], 'output size mismatch'
    assert lif_size == lif_node.v_threshold.size, 'lif size mismatch (v_threshold)'
    assert lif_size == wrec_node.weight.shape[0], 'w_rec shape mismatch'
    assert lif_size == wrec_node.weight.shape[1], 'w_rec shape mismatch'

    return lif_node, wrec_node, lif_size


def _nir_node_to_spyx_node(node_pair: nir.NIRNode):
    """Converts a NIR node to a Spyx node."""
    # NOTE: all nodes have node.input_type and node.output_type
    # which specify the input and output shape of the node.
    node, next_node = node_pair

    if isinstance(node, (nir.Input, nir.Output)):
        node.input_type
        return None

    elif isinstance(node, nir.Affine):
        # NOTE: node.weight, node.bias are npy arrays
        return hk.Linear(node.weight.shape[-1], with_bias=True)

    elif isinstance(node, nir.Linear):
        # NOTE: node.weight
        return hk.Linear(node.weight.shape[-1], with_bias=False)

    elif isinstance(node, nir.Conv1d):  # not needed atm
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        pass

    elif isinstance(node, nir.Conv2d):
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        p0, p1 = node.padding[0], node.padding[1]
        return hk.Conv2D(
            output_channels=node.weight.shape[0],
            kernel_shape=node.weight.shape[-1],
            rate=node.dilation.tolist(),
            padding=[(p0, p0), (p1, p1)],
            stride=node.stride.tolist(),
            data_format="NCHW",
            feature_group_count=node.groups,
        )

    elif isinstance(node, nir.SumPool2d):
        return SumPool(
            node.kernel_size, node.stride.tolist(), "VALID", channel_axis=1
        )  # hacky...

    elif isinstance(node, nir.IF):  # getting shape is an issue...?
        # NOTE: node.r, node.v_threshold
        return IF(node.r.shape, threshold=node.v_threshold)

    elif isinstance(node, nir.LIF):
        # NOTE: node.r, node.v_threshold, node.tau, node.v_leak
        return LIF(node.tau.shape, threshold=node.v_threshold)

    elif isinstance(node, nir.CubaLIF):
        # NOTE: node.r, node.v_threshold, node.v_leak
        # node.tau_mem, node.tau_syn
        # node.w_in
        return CuBaLIF(node.tau_syn.shape, threshold=node.v_threshold)

    elif isinstance(node, nir.Flatten):
        # NOTE: node.start_dim, node.end_dim
        return hk.Flatten()

    elif isinstance(node, nir.I):  # not needed atm
        pass

    elif isinstance(node, nir.Sequence):  # not needed atm
        pass

    elif isinstance(node, nir.Scale):  # not needed atm
        pass

    elif isinstance(node, nir.Delay):  # not needed atm
        pass

    elif isinstance(node, nir.Threshold):  # not needed atm
        pass

    elif isinstance(node, nir.NIRGraph):
        print('found subgraph, trying to parse as RNN')
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)
        # TODO: implement RIF, RLIF generation
        
        if isinstance(lif_node, nir.IF):
            return RIF(lif_size, threshold=lif_node.v_leak)
        elif isinstance(lif_node, nir.LIF):
            return RLIF(lif_node.tau.shape, threshold=lif_node.v_threshold)
        else:
            return RCuBaLIF(lif_node.tau_syn.shape, threshold=lif_node.v_threshold)

    else:
        print("[Warning] Layer not recognized by NIR.")
        print("Unsupported layer was not added to NIRGraph:", node.__class__)


def _nir_node_to_spyx_params(node_pair: nir.NIRNode, dt: float):
    """Converts a NIR node to a Spyx node."""
    # NOTE: all nodes have node.input_type and node.output_type
    # which specify the input and output shape of the node.

    node, next_node = node_pair

    if isinstance(node, (nir.Input, nir.Output)):
        node.input_type
        return None

    elif isinstance(node, nir.Affine):
        # NOTE: node.weight, node.bias are npy arrays
        tau = 1
        if isinstance(next_node, nir.LIF):
            tau = next_node.tau
        elif isinstance(next_node, nir.CubaLIF):
            tau = next_node.tau_syn
            w_in = next_node.w_in
        elif isinstance(next_node, nir.NIRGraph):
            next_lif, _, _ = _parse_rnn_subgraph(next_node)
            if isinstance(next_lif, nir.LIF):
                tau = next_lif.tau
            elif isinstance(next_lif, nir.CubaLIF):
                tau = next_lif.tau_syn
                w_in = next_lif.w_in
            else:
                pass
        else:
            tau = 1
        
        w_scale = dt / tau
        if w_in is not None: # need some treatment for arbitrary R in the future...
            w_scale *= w_in
        return {
            "w": jnp.array(node.weight) * w_scale,
            "b": jnp.array(node.bias) * w_scale,
        }

    elif isinstance(node, nir.Linear):
        # NOTE: node.weight
        if isinstance(next_node, nir.LIF):
            tau = next_node.tau
        elif isinstance(next_node, nir.LI):
            tau = next_node.tau
        elif isinstance(next_node, nir.CubaLIF):
            tau = next_node.tau_syn
            w_in = next_node.w_in
        elif isinstance(next_node, nir.NIRGraph):
            next_lif, _, _ = _parse_rnn_subgraph(next_node)
            if isinstance(next_lif, nir.LIF):
                tau = next_lif.tau
            elif isinstance(next_lif, nir.CubaLIF):
                tau = next_lif.tau_syn
                w_in = next_lif.w_in
            else:
                pass
        else:
            tau = 1
        w_scale = dt / tau
        if w_in is not None: # need some treatment for arbitrary R in the future...
            w_scale *= w_in
        return {"w": jnp.array(node.weight) * w_scale}

    elif isinstance(node, nir.Conv1d):  # not needed atm
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        pass

    elif isinstance(node, nir.Conv2d):
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        if isinstance(next_node, nir.LIF):
            tau = next_node.tau
        elif isinstance(next_node, nir.CubaLIF):
            tau = next_node.tau_syn
            w_in = next_node.w_in
        else:
            tau = 1

        w_scale = dt / tau # NOTE: cannot support direct pooling of conv layers.
        if w_in is not None: # need some treatment for arbitrary R in the future...
            w_scale *= w_in
        # hk.conv2d expects weights in the format HWIO, NIR is OIHW
        weight = node.weight.transpose((2, 3, 1, 0)) * w_scale
        bias = node.bias.reshape(-1, 1, 1) * w_scale

        return {"w": jnp.array(weight), "b": jnp.array(bias)}

    elif isinstance(node, nir.IF):  # getting shape is an issue...?
        # NOTE: node.r, node.v_threshold
        return {}  # might need to return none/pass here, not sure yet.

    elif isinstance(node, nir.LIF):
        # NOTE: node.r, node.v_threshold, node.tau, node.v_leak
        return {"beta": 1 - (dt / node.tau)}

    elif isinstance(node, nir.CubaLIF):
        # NOTE: node.r, node.v_threshold, node.v_leak
        # node.tau_mem, node.tau_syn
        # node.w_in
        return {"alpha": 1 - (dt / node.tau_syn), "beta": 1 - (dt / node.tau_mem)}

    elif isinstance(node, nir.Flatten):
        # NOTE: node.start_dim, node.end_dim
        return {}

    elif isinstance(node, nir.I):  # not needed atm
        pass

    elif isinstance(node, nir.Sequence):  # not needed atm
        pass

    elif isinstance(node, nir.Scale):  # not needed atm
        pass

    elif isinstance(node, nir.Delay):  # not needed atm
        pass

    elif isinstance(node, nir.Threshold):  # not needed atm
        pass

    elif isinstance(node, nir.NIRGraph):
        print('found subgraph, trying to parse as RNN')
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)
        # TODO: implement RNN subgraph parsing

        if isinstance(wrec_node, nir.Linear):
            bias = jnp.zeros(wrec_node.weight.shape[0])
        else:
            bias = wrec_node.bias

        if isinstance(lif_node, nir.IF):
            w_scale = lif_node.r * dt
            return {
                "w": jnp.array(wrec_node.weight.T)*w_scale,
                "b": jnp.array(bias)*w_scale
            }
        elif isinstance(lif_node, nir.LIF):
            w_scale = lif_node.r * dt / lif_node.tau
            return {
                "w": jnp.array(wrec_node.weight.T)*w_scale,
                "b": jnp.array(bias)*w_scale,
                "beta":  1 - (dt / lif_node.tau)
            }
        else: # RCuBaLIF # need option to enable/disable weight scaling...
            w_scale = lif_node.w_in * dt / lif_node.tau_syn
            return {
                "w": jnp.array(wrec_node.weight.T)*w_scale,
                "b": jnp.array(bias)*w_scale,
                "alpha": 1 - (dt / lif_node.tau_syn),
                "beta":  1 - (dt / lif_node.tau_mem)
            }

        pass

    else:
        print('[Warning] node not recognized:', node.__class__)


def to_nir(spyx_pytree, input_shape, output_shape, dt) -> nir.NIRGraph:
    """Converts a Spyx network to a NIR graph. Under Construction. Currently only supports exporting networks without explicit recurrence/feedback."""
    # construct the edge list for the NIRGraph
    keys = list(spyx_pytree.keys())
    edges = [
        (keys[i], keys[i + 1]) for i in range(len(keys) - 1)
    ]  # assume linear connectivity
    edges.insert(0, ("input", edges[0][0]))
    edges.append((edges[-1][1], "output"))

    # begin constructing the node list:
    nodes = {"input": nir.Input(input_shape), "output": nir.Output(output_shape)}

    for layer, params in spyx_pytree.items():
        layer_type = layer.split("_")[0]
        if layer_type == "linear":
            if "b" in params:
                nodes[layer] = nir.Affine(np.array(params["w"]), np.array(params["b"]))
            else:
                nodes[layer] = nir.Linear(np.array(params["w"]))
        elif layer_type == "conv2":
            # this is hard coded... allow dicts for each layer for more flexible config?
            #p0, p1 = node.padding[0], node.padding[1]  # TODO: figure out how to let the user specify this stuff...
            nodes[layer] = nir.Conv2d(
                weights=np.array(weight=params["w"]),
                bias=np.array(params["b"]),
                dilation=1,
                stride=1,
                padding="SAME",#[(p0, p0), (p1, p1)],
                groups=1,
            )
        elif layer_type == "IF":
            nodes[layer] = nir.IF(r=1, v_threshold=1)
        elif layer_type == "LI":
            nodes[layer] = nir.LI(
                tau=dt / (1 - np.array(params["beta"])),
                v_leak=np.zeros_like(params["beta"]),
                r=np.array(params["beta"]),)
        elif layer_type == "LIF":
            nodes[layer] = nir.LIF(
                tau=dt / (1 - np.array(params["beta"])),
                v_threshold=np.ones_like(params["beta"]),
                v_leak=np.zeros_like(params["beta"]),
                r=np.array(params["beta"]),
            )
        elif layer_type == "CuBaLIF":
            nodes[layer] = nir.CubaLIF(
                tau_mem=dt / (1 - np.array(params["beta"])),
                tau_syn=dt / (1 - np.array(params["alpha"])),
                v_threshold=np.ones_like(params["beta"]),
                v_leak=np.zeros_like(params["beta"]),
                r=np.array(params["beta"]),
            )
        else: # TODO: implement explicit recurrent export via subgraphs...
            print("[Warning] Layer not recognized by NIR or export not yet supported (explicit recurrent layers).")
            print("Unsupported layer was not added to NIRGraph:", layer)

    return nir.NIRGraph(nodes, edges)


# spyx has built in RIF/RLIF/RCuBaLIF, so we need to fuse these nodes to work.
def _remove_recurrent_links(nir_graph):
    pass


def _find_tuple_with_first_element(lst, value):
    return next((tup for tup in lst if tup[0] == value), None)


def _order_edge_list(edge_list):  # needs reviewed...
    curr_node, next_node = "input", None
    ordered_list = []
    while next_node != "output":
        tup = _find_tuple_with_first_element(edge_list, curr_node)
        ordered_list.append(tup)
        next_node = tup[1]
        curr_node = next_node
    return ordered_list


# right now NIR is storing the affine weights in a transposed format, so we need to fix them first.
def _transpose_affine_weights(nodes):
    for k, n in nodes.items():
        if isinstance(n, nir.Linear):
            nodes[k].weight = nodes[k].weight.T
        elif isinstance(n, nir.Affine):
            nodes[k].weight = nodes[k].weight.T
            nodes[k].bias = nodes[k].bias.T
        else:
            continue
    return nodes


def from_nir(
    nir_graph: nir.NIRGraph,
    sample_batch: jnp.array,
    dt: float,
    time_major: bool = False,
    return_all_states: bool = False,
):
    """Converts a NIR graph to a Spyx network."""
    # find valid RNN subgraphs, and replace them with a single NIRGraph node
    nir_graph = _replace_rnn_subgraph_with_nirgraph(nir_graph)

    # NOTE: iterate over nir_graph, convert each node to a Spyx module
    # NOTE: Need to iterate over nirgraph edes and nodes,
    # replacing seperate recurrent layers with merged versions that can then be
    # loaded into spyx as either RIF, RLIF, or RCuBaLIF neurons.
    # Also sort the list so that it flows from input to output, since sorting is not assured.
    sorted_edges = _order_edge_list(nir_graph.edges)
    nir_graph.nodes = _transpose_affine_weights(nir_graph.nodes)

    def snn(x):
        core = hk.DeepRNN(
            [
                _nir_node_to_spyx_node((nir_graph.nodes[n[0]], nir_graph.nodes[n[1]]))
                for n in sorted_edges[1:]
            ]
        )

        # This takes our SNN core and computes it across the input data.
        spikes, V = hk.dynamic_unroll(
            core,
            x,
            core.initial_state(x.shape[0]),
            time_major=time_major,
            return_all_states=return_all_states,
        )

        return spikes, V

    SNN = hk.without_apply_rng(hk.transform(snn))

    param_names = SNN.init(jax.random.PRNGKey(0), sample_batch).keys()

    parametrized_layers = []
    for edge in sorted_edges[1:]:
        if isinstance(nir_graph.nodes[edge[0]], (nir.Conv2d, nir.Affine, nir.Linear, nir.LIF, nir.CubaLIF, nir.NIRGraph)):
            parametrized_layers.append(
                (nir_graph.nodes[edge[0]], nir_graph.nodes[edge[1]])
            )

    params = {
        k: _nir_node_to_spyx_params(n, dt)
        for k, n in zip(param_names, parametrized_layers)
    }

    return SNN, params
