import jax
import jax.numpy as jnp
import nir
import pytest
from flax import nnx

from spyx import nir as spyx_nir
from spyx import nn, phasor


def _roundtrip(original, input_shape, output_shape, x):
    """Export ``original`` to NIR, re-import + run it on ``x`` (time-major), and
    assert the imported model reproduces the original's output.

    Returns the imported model so callers can add parameter-level checks.
    """
    graph = spyx_nir.to_nir(original, {"input": input_shape}, {"output": output_shape})
    # from_nir builds *and runs* the model (run-and-return API).
    imported, out = spyx_nir.from_nir(graph, x, dt=1, rngs=nnx.Rngs(1))
    ref_out, _ = nn.run(original, x)
    assert jnp.allclose(ref_out, out, atol=1e-5)
    return imported, graph


def test_nir_export_import_lif():
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs), nn.LIF((20,), beta=0.8, rngs=rngs)
    )
    x = jax.random.normal(jax.random.PRNGKey(42), (7, 5, 10))  # (T, B, in)
    imported, _ = _roundtrip(original, (10,), (20,), x)

    assert jnp.allclose(original.layers[0].kernel[...], imported.layers[0].kernel[...])
    assert jnp.allclose(original.layers[1].beta[...], imported.layers[1].beta[...])


def test_nir_export_import_cubalif():
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(10, 15, use_bias=False, rngs=rngs),
        nn.CuBaLIF((15,), alpha=0.9, beta=0.7, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (7, 5, 10))
    imported, _ = _roundtrip(original, (10,), (15,), x)

    assert jnp.allclose(original.layers[1].alpha[...], imported.layers[1].alpha[...])
    assert jnp.allclose(original.layers[1].beta[...], imported.layers[1].beta[...])


def test_nir_export_import_rlif():
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(8, 12, use_bias=False, rngs=rngs),
        nn.RLIF((12,), beta=0.85, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (6, 4, 8))
    imported, _ = _roundtrip(original, (8,), (12,), x)

    assert jnp.allclose(
        original.layers[1].recurrent_w[...], imported.layers[1].recurrent_w[...]
    )
    assert jnp.allclose(original.layers[1].beta[...], imported.layers[1].beta[...])


def test_nir_export_import_rcubalif():
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(6, 6, use_bias=False, rngs=rngs),
        nn.RCuBaLIF((6,), alpha=0.95, beta=0.9, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(3), (6, 4, 6))
    imported, _ = _roundtrip(original, (6,), (6,), x)

    assert jnp.allclose(
        original.layers[1].recurrent_w[...], imported.layers[1].recurrent_w[...]
    )
    assert jnp.allclose(original.layers[1].alpha[...], imported.layers[1].alpha[...])
    assert jnp.allclose(original.layers[1].beta[...], imported.layers[1].beta[...])


def test_nir_export_import_if():
    """IF export previously passed r=1 (int) to nir.IF, which requires arrays."""
    rngs = nnx.Rngs(0)
    original = nn.Sequential(nnx.Linear(10, 8, rngs=rngs), nn.IF((8,)))
    x = jax.random.normal(jax.random.PRNGKey(42), (7, 5, 10))
    _roundtrip(original, (10,), (8,), x)


def test_nir_export_import_flatten():
    """Flatten export previously built a malformed nir.Flatten (nested dict)."""
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nn.Flatten(),
        nnx.Linear(2 * 3 * 3, 8, rngs=rngs),
        nn.LIF((8,), beta=0.8, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(42), (7, 5, 2, 3, 3))
    _, graph = _roundtrip(original, (2, 3, 3), (8,), x)
    # NIR flatten node carries the flattened shape.
    assert int(graph.nodes["layer_0"].output_type["output"].prod()) == 2 * 3 * 3


def test_nir_export_import_conv_flatten_dense():
    """Conv -> Flatten -> Linear -> LIF (conv feature extractor + dense head)."""
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Conv(2, 4, (3, 3), rngs=rngs),
        nn.Flatten(),
        nnx.Linear(4 * 8 * 8, 10, rngs=rngs),
        nn.LIF((10,), beta=0.8, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (5, 3, 8, 8, 2))  # (T, B, H, W, C)
    _roundtrip(original, (2, 8, 8), (10,), x)


def test_nir_export_import_spiking_conv():
    """Conv -> spatial LIF -> Flatten -> Linear -> LIF (SCNN with spiking conv).

    Exercises the channels-first (NIR) <-> channels-last (spyx) neuron-shape
    bridge for a neuron that follows a convolution.
    """
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Conv(2, 4, (3, 3), rngs=rngs),
        nn.LIF((8, 8, 4), beta=0.8, rngs=rngs),  # spatial (H, W, C) state
        nn.Flatten(),
        nnx.Linear(4 * 8 * 8, 10, rngs=rngs),
        nn.LIF((10,), beta=0.9, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(5), (5, 3, 8, 8, 2))
    _roundtrip(original, (2, 8, 8), (10,), x)


def test_nir_export_import_conv_sumpool_scnn():
    """Full SCNN: Conv -> spatial IF -> SumPool -> Conv -> Flatten -> Linear -> LIF.

    Covers channels-first (NIR) <-> channels-last (spyx) for SumPool as well as
    spiking convs.
    """
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Conv(2, 4, (3, 3), rngs=rngs),
        nn.IF((8, 8, 4)),
        nn.SumPool((2, 2), (2, 2), "VALID"),  # (8,8,4) -> (4,4,4)
        nnx.Conv(4, 6, (3, 3), rngs=rngs),
        nn.Flatten(),
        nnx.Linear(6 * 4 * 4, 10, rngs=rngs),
        nn.LIF((10,), beta=0.8, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(7), (4, 3, 8, 8, 2))  # (T,B,H,W,C)
    _roundtrip(original, (2, 8, 8), (10,), x)


def test_nir_export_import_psu_lif():
    """PSU_LIF is reset-free, so it exports to nir.LI (the reset-free leaky
    membrane) + nir.Threshold (the spike rule) -- a faithful, gap-free mapping
    that re-imports to a single PSU_LIF and reproduces the original output.
    """
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs),
        nn.PSU_LIF((20,), beta=0.8, threshold=1.0, rngs=rngs),
    )
    x = jax.random.normal(jax.random.PRNGKey(42), (7, 5, 10))  # (T, B, in)
    imported, graph = _roundtrip(original, (10,), (20,), x)

    # Structural check: PSU_LIF becomes an LI membrane + a Threshold, NOT an
    # nir.LIF (which would inject a reset PSU_LIF does not have).
    assert isinstance(graph.nodes["layer_1"], nir.LI)
    assert isinstance(graph.nodes["layer_1_threshold"], nir.Threshold)
    assert not any(isinstance(n, nir.LIF) for n in graph.nodes.values())

    # The re-imported layer is a single PSU_LIF with matching decay/threshold.
    assert isinstance(imported.layers[1], nn.PSU_LIF)
    assert jnp.allclose(original.layers[1].beta[...], imported.layers[1].beta[...])
    assert jnp.allclose(
        original.layers[1].threshold, imported.layers[1].threshold, atol=1e-6
    )


def test_nir_psu_lif_li_tau_matches_beta():
    """The exported nir.LI carries tau = dt / (1 - beta) for beta = 0.8, dt = 1."""
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(4, 6, rngs=rngs),
        nn.PSU_LIF((6,), beta=0.8, rngs=rngs),
    )
    graph = spyx_nir.to_nir(original, {"input": (4,)}, {"output": (6,)}, dt=1)
    li = graph.nodes["layer_1"]
    assert jnp.allclose(jnp.asarray(li.tau), 1.0 / (1.0 - 0.8), atol=1e-5)
    assert jnp.allclose(jnp.asarray(li.v_leak), 0.0)


def test_nir_export_resonatefire_not_implemented():
    """ResonateFire has no faithful NIR primitive (no complex / oscillatory
    node), so exporting it raises NotImplementedError rather than faking a
    real-valued mapping that would discard the oscillatory dynamics.
    """
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nnx.Linear(8, 6, use_bias=False, rngs=rngs),
        phasor.ResonateFire((6,), lambda_init=0.1, omega_init=1.0, rngs=rngs),
    )
    with pytest.raises(NotImplementedError, match="ResonateFire"):
        spyx_nir.to_nir(model, {"input": (8,)}, {"output": (6,)})


def test_from_nir_return_all_states():
    """return_all_states yields (outputs, per-layer states)."""
    rngs = nnx.Rngs(0)
    original = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs), nn.LIF((20,), beta=0.8, rngs=rngs)
    )
    graph = spyx_nir.to_nir(original, {"input": (10,)}, {"output": (20,)})
    x = jax.random.normal(jax.random.PRNGKey(6), (7, 5, 10))

    _, outputs = spyx_nir.from_nir(graph, x, dt=1)
    assert outputs.shape == (7, 5, 20)

    _, (outputs2, states) = spyx_nir.from_nir(graph, x, dt=1, return_all_states=True)
    assert jnp.allclose(outputs, outputs2)
    # states mirror initial_state (a per-layer list); the LIF membrane trace is
    # captured at every timestep -> leading time axis of length T.
    lif_state = states[1]
    assert lif_state.shape == (7, 5, 20)


if __name__ == "__main__":
    test_nir_export_import_lif()
    test_nir_export_import_cubalif()
    test_nir_export_import_rlif()
    test_nir_export_import_rcubalif()
    test_nir_export_import_if()
    test_nir_export_import_flatten()
    test_nir_export_import_conv_flatten_dense()
    test_nir_export_import_spiking_conv()
    test_nir_export_import_psu_lif()
    test_nir_psu_lif_li_tau_matches_beta()
    test_from_nir_return_all_states()
    print("NIR tests passed!")
