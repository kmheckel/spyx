import jax
import jax.numpy as jnp

import nir
import numpy as np
import haiku as hk

from .nn import LIF
from .nn import IF
from .nn import CuBaLIF

def _nir_node_to_spyx_node(node: nir.NIRNode):
    """Converts a NIR node to a Spyx node."""
    # NOTE: all nodes have node.input_type and node.output_type
    # which specify the input and output shape of the node.
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
        return hk.Conv2D(
            output_channels=node.weight.shape[0],
            kernel_shape=node.weight.shape[-1],
            rate=node.dilation,
            padding=node.padding,
            stride=node.stride,
            feature_group_count=node.groups
        )

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
        w_scale = next_node.r * dt / next_node.tau
        return {"w":jnp.array(node.weight)*w_scale, "b":jnp.array(node.bias)*w_scale}

    elif isinstance(node, nir.Linear):
        # NOTE: node.weight
        w_scale = next_node.r * dt / next_node.tau
        return {"w":jnp.array(node.weight)*w_scale}

    elif isinstance(node, nir.Conv1d):  # not needed atm
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        pass

    elif isinstance(node, nir.Conv2d):
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        w_scale = next_node.r * dt / next_node.tau # NOTE: cannot support direct pooling of conv layers.
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
            nodes[layer] = nir.Conv2d(
                weights=np.array(weight=params["w"]),
                bias=np.array(params["b"]),
                dilation=1,
                stride=1, 
                padding="same",
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
    


def from_nir(nir_graph: nir.NIRGraph, sample_batch: jnp.array, dt: float, time_major: bool = False, return_all_states:bool = False):
    """Converts a NIR graph to a Spyx network."""
    # NOTE: iterate over nir_graph, convert each node to a Spyx module
    # (using _nir_node_to_spyx_node)
    # could do this cleanly by using a list comprehension on the NIRGraph and then passing the list to the hk.RNNCore constructor.
    # actually, this might be more complicated. Might want to create entire haiku function in here and transform it, returning
    # just the pure function object and the associated parameter pytree. Could make things a lot cleaner.

    def snn(x):
        
        core = hk.DeepRNN([ _nir_node_to_spyx_node(nir_graph.nodes[n[0]]) for n in nir_graph.edges[1:] ])
    
        # This takes our SNN core and computes it across the input data.
        spikes, V = hk.dynamic_unroll(core, x, core.initial_state(x.shape[0]), time_major=time_major, return_all_states=return_all_states)
    
        return spikes, V

    SNN = hk.without_apply_rng(hk.transform(snn))

    param_names = SNN.init(jax.random.PRNGKey(0), sample_batch).keys()
    
    params = { k:_nir_node_to_spyx_params((nir_graph.nodes[n[0]], nir_graph.nodes[n[1]]), dt) for k,n in zip(param_names, nir_graph.edges[1:]) }
    
    return SNN, params

