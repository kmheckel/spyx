import nir
import numpy as np
import haiku as hk
from .nn import LIF

def _nir_node_to_spyx_node(node: nir.NIRNode):
    """Converts a NIR node to a Spyx node."""
    # NOTE: all nodes have node.input_type and node.output_type
    # which specify the input and output shape of the node.
    if isinstance(node, (nir.Input, nir.Output)):
        node.input_type
        return None

    elif isinstance(node, nir.Affine):
        # NOTE: node.weight, node.bias are npy arrays
        pass
        # return hk.Linear()

    elif isinstance(node, nir.Linear):
        # NOTE: node.weight
        pass
        # return hk.Linear()

    elif isinstance(node, nir.Conv1d):  # not needed atm
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        pass

    elif isinstance(node, nir.Conv2d):
        # NOTE: node.bias, node.weight
        # node.dilation, node.groups, node.padding, node.stride
        pass

    elif isinstance(node, nir.IF):
        # NOTE: node.r, node.v_threshold
        pass

    elif isinstance(node, nir.LIF):
        # NOTE: node.r, node.v_threshold, node.tau, node.v_leak
        pass

    elif isinstance(node, nir.CubaLIF):
        # NOTE: node.r, node.v_threshold, node.v_leak
        # node.tau_mem, node.tau_syn
        # node.w_in
        pass

    elif isinstance(node, nir.Flatten):
        # NOTE: node.start_dim, node.end_dim
        pass

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


def to_nir(spyx_pytree, input_shape, output_shape) -> nir.NIRGraph:
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
        elif layer_type == "conv2d":
            pass # nodes[layer] = nir.Conv2d()
        elif layer_type == "IF":
            nodes[layer] = nir.IF(r=1, v_threshold=1)
        elif layer_type == "LIF":
            nodes[layer] = nir.LIF(
                tau=1/(1-np.array(params["beta"])),
                v_threshold=np.ones_like(params["beta"]),
                v_leak=np.zeros_like(params["beta"]),
                r=np.array(params["beta"])
            )
        else:
            print("Attempted exportation of a model which contains a layer not support by NIR.")
            print("Unsupported layer name was not added to NIRGraph:", layer)

    return nir.NIRGraph(nodes, edges)
    


def from_nir(nir_graph: nir.NIRGraph):
    """Converts a NIR graph to a Spyx network."""
    # NOTE: iterate over nir_graph, convert each node to a Spyx module
    # (using _nir_node_to_spyx_node)
    # could do this cleanly by using a list comprehension on the NIRGraph and then passing the list to the hk.RNNCore constructor.
    # actually, this might be more complicated. Might want to create entire haiku function in here and transform it, returning
    # just the pure function object and the associated parameter pytree. Could make things a lot cleaner.
    pass
    # for every non-input/output node, convert to a spyx node and return in list form.
    # return [ _nir_node_to_spyx_node(nir_graph.nodes[n[0]]) for n in nir_graph.edges[1:] ]
