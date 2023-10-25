import nir


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

    elif isinstance(node, nir.Linear):
        # NOTE: node.weight
        pass

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


def to_nir(spyx_graph) -> nir.NIRGraph:
    """Converts a Spyx network to a NIR graph."""
    pass


def from_nir(nir_graph: nir.NIRGraph):
    """Converts a NIR graph to a Spyx network."""
    # NOTE: iterate over nir_graph, convert each node to a Spyx module
    # (using _nir_node_to_spyx_node)
    pass
