import jax
import jax.numpy as jnp

import nir
import numpy as np
import haiku as hk

from .nn import LIF
from .nn import IF
from .nn import CuBaLIF

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
            padding=[(p0,p0),(p1,p1)],
            stride=node.stride.tolist(),
            data_format="NCHW",
            feature_group_count=node.groups
        )

    elif isinstance(node, nir.SumPool2d):
        return hk.AvgPool(node.kernel_size, node.stride.tolist(), "VALID", channel_axis=1) # hacky...

    elif isinstance(node, nir.IF): # getting shape is an issue...?
        # NOTE: node.r, node.v_threshold
        return IF(node.r.shape, threshold=node.v_threshold)

    elif isinstance(node, nir.LIF):
        # NOTE: node.r, node.v_threshold, node.tau, node.v_leak
        return LIF(node.tau.shape, threshold=node.v_threshold)

    elif isinstance(node, nir.CubaLIF):
        # NOTE: node.r, node.v_threshold, node.v_leak
        # node.tau_mem, node.tau_syn
        # node.w_in
        return CuBaLIF(node.tau_mem.shape, threshold=node.v_threshold)

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
        else:
            tau = 1

        w_scale = dt / tau
        return {"w":jnp.array(node.weight)*w_scale, "b":jnp.array(node.bias)*w_scale}

    elif isinstance(node, nir.Linear):
        # NOTE: node.weight
        tau = 1
        if isinstance(next_node, nir.LIF):
            tau = next_node.tau
        elif isinstance(next_node, nir.CubaLIF):
            tau = next_node.tau_syn
        else:
            tau = 1

        w_scale = dt / tau
        return {"w":jnp.array(node.weight)*w_scale}

    elif isinstance(node, nir.Conv1d):  # not needed atm
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        pass

    elif isinstance(node, nir.Conv2d):
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        tau = 1
        if isinstance(next_node, nir.LIF):
            tau = next_node.tau
        elif isinstance(next_node, nir.CubaLIF):
            tau = next_node.tau_syn
        else:
            tau = 1

        w_scale = 1 # dt / tau # NOTE: cannot support direct pooling of conv layers.
        return {"w":jnp.array(node.weight)*w_scale, "b":jnp.array(node.bias)*w_scale}

    elif isinstance(node, nir.IF): # getting shape is an issue...?
        # NOTE: node.r, node.v_threshold
        return {} # might need to return none/pass here, not sure yet.

    elif isinstance(node, nir.LIF):
        # NOTE: node.r, node.v_threshold, node.tau, node.v_leak
        return {"beta":1-(dt/node.tau)}

    elif isinstance(node, nir.CubaLIF):
        # NOTE: node.r, node.v_threshold, node.v_leak
        # node.tau_mem, node.tau_syn
        # node.w_in
        return {"alpha":1-(dt/node.tau_syn), "beta":1-(dt/node.tau_mem)}

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


def to_nir(spyx_pytree, input_shape, output_shape, dt) -> nir.NIRGraph:
    """Converts a Spyx network to a NIR graph."""
    # construct the edge list for the NIRGraph
    keys = list(spyx_pytree.keys())
    edges = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1)] # assume linear connectivity
    edges.insert(0, ("input", edges[0][0]))
    edges.append((edges[-1][1], "output"))

    # begin constructing the node list:
    nodes = {
        "input" : nir.Input(input_shape),
        "output" : nir.Output(output_shape)
    }

    for layer, params in spyx_pytree.items():
        layer_type = layer.split("_")[0]
        if layer_type == "linear":
            if "b" in params:
                nodes[layer] = nir.Affine(np.array(params["w"]), np.array(params["b"]))
            else:
                nodes[layer] = nir.Linear(np.array(params["w"]))
        elif layer_type == "conv2": # this is hard coded... might need to allow dicts for each layer for more flexible config...
            p0, p1 = node.padding[0], node.padding[1]
            nodes[layer] = nir.Conv2d(
                weights=np.array(weight=params["w"]),
                bias=np.array(params["b"]),
                dilation=1,
                stride=1, 
                padding=[(p0,p0),(p1,p1)],
                groups=1
                )
        elif layer_type == "IF":
            nodes[layer] = nir.IF(r=1, v_threshold=1)
        elif layer_type == "LIF":
            nodes[layer] = nir.LIF(
                tau=dt/(1-np.array(params["beta"])),
                v_threshold=np.ones_like(params["beta"]),
                v_leak=np.zeros_like(params["beta"]),
                r=np.array(params["beta"])
            )
        elif layer_type == "CuBaLIF":
            nodes[layer] = nir.CubaLIF(
                tau_mem=dt/(1-np.array(params["beta"])),
                tau_syn=dt/(1-np.array(params["alpha"])),
                v_threshold=np.ones_like(params["beta"]),
                v_leak=np.zeros_like(params["beta"]),
                r=np.array(params["beta"])
            )
        else:
            print("Attempted exportation of a model which contains a layer not supported/layer name not recognized by NIR.")
            print("Unsupported layer was not added to NIRGraph:", layer)

    return nir.NIRGraph(nodes, edges)
    

# spyx has built in RIF/RLIF/RCuBaLIF, so we need to fuse these nodes to work.
def _remove_recurrent_links(nir_graph):
    pass

def _find_tuple_with_first_element(lst, value):
    return next((tup for tup in lst if tup[0] == value), None)

def _order_edge_list(edge_list): # needs reviewed...
    curr_node, next_node = "input", None
    ordered_list = []
    while next_node != "output":
        tup = _find_tuple_with_first_element(edge_list, curr_node)
        ordered_list.append(tup)
        next_node = tup[1]
        curr_node = next_node
    return ordered_list

# right now NIR is storing the affine weights in a transposed format for no good reason, so we need to fix them first.
def _transpose_affine_weights(nodes):
    for k,n in nodes.items():
        if isinstance(n, nir.Linear):
            nodes[k].weight = nodes[k].weight.T
        elif isinstance(n, nir.Affine):
            nodes[k].weight = nodes[k].weight.T
            nodes[k].bias = nodes[k].bias.T
        else:
            continue
    return nodes

def from_nir(nir_graph: nir.NIRGraph, sample_batch: jnp.array, dt: float, time_major: bool = False, return_all_states:bool = False):
    """Converts a NIR graph to a Spyx network."""
    # NOTE: iterate over nir_graph, convert each node to a Spyx module
    # NOTE: Need to iterate over nirgraph edes and nodes,
    # replacing seperate recurrent layers with merged versions that can then be
    # loaded into spyx as either RIF, RLIF, or RCuBaLIF neurons.
    # Also sort the list so that it flows from input to output, since sorting is not assured.
    sorted_edges = _order_edge_list(nir_graph.edges)
    nir_graph.nodes = _transpose_affine_weights(nir_graph.nodes)

    def snn(x):
        
        core = hk.DeepRNN([ _nir_node_to_spyx_node((nir_graph.nodes[n[0]], nir_graph.nodes[n[1]])) for n in sorted_edges[1:] ])
    
        # This takes our SNN core and computes it across the input data.
        spikes, V = hk.dynamic_unroll(core, x, core.initial_state(x.shape[0]), time_major=time_major, return_all_states=return_all_states)
    
        return spikes, V

    SNN = hk.without_apply_rng(hk.transform(snn))

    param_names = SNN.init(jax.random.PRNGKey(0), sample_batch).keys()
    
    parametrized_layers = []
    for edge in sorted_edges[1:]:
        if isinstance(nir_graph.nodes[edge[0]], nir.Conv2d) or isinstance(nir_graph.nodes[edge[0]], nir.Affine) or isinstance(nir_graph.nodes[edge[0]], nir.Linear):
            parametrized_layers.append((nir_graph.nodes[edge[0]], nir_graph.nodes[edge[1]]))

    params = { k:_nir_node_to_spyx_params(n, dt) for k,n in zip(param_names, parametrized_layers) }
    
    return SNN, params

