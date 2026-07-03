import jax
import jax.numpy as jnp
from flax import nnx

from spyx import nir as spyx_nir
from spyx import nn


def test_nir_export_import_lif():
    rngs = nnx.Rngs(0)
    # Define original model
    original_model = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs), nn.LIF((20,), beta=0.8, rngs=rngs)
    )

    # Export to NIR
    input_shape = {"input": (10,)}
    output_shape = {"output": (20,)}
    nir_graph = spyx_nir.to_nir(original_model, input_shape, output_shape)

    # Import from NIR
    imported_model = spyx_nir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))

    # Compare parameters
    # original_model.layers[0] is Linear
    # original_model.layers[1] is LIF

    # Check linear kernel
    assert jnp.allclose(
        original_model.layers[0].kernel[...], imported_model.layers[0].kernel[...]
    )
    # Check LIF beta
    assert jnp.allclose(
        original_model.layers[1].beta[...], imported_model.layers[1].beta[...]
    )

    # Compare outputs for same input
    x = jax.random.normal(jax.random.PRNGKey(42), (5, 10))  # batch of 5

    # Since they are stateful, we need to initialize state
    state_orig = original_model.initial_state(5)
    state_imp = imported_model.initial_state(5)

    out_orig, _ = original_model(x, state_orig)
    out_imp, _ = imported_model(x, state_imp)

    assert jnp.allclose(out_orig, out_imp)


def test_nir_export_import_cubalif():
    rngs = nnx.Rngs(0)
    original_model = nn.Sequential(
        nnx.Linear(10, 15, use_bias=False, rngs=rngs),
        nn.CuBaLIF((15,), alpha=0.9, beta=0.7, rngs=rngs),
    )

    input_shape = {"input": (10,)}
    output_shape = {"output": (15,)}
    nir_graph = spyx_nir.to_nir(original_model, input_shape, output_shape)

    imported_model = spyx_nir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))

    assert jnp.allclose(
        original_model.layers[0].kernel[...], imported_model.layers[0].kernel[...]
    )
    assert jnp.allclose(
        original_model.layers[1].alpha[...], imported_model.layers[1].alpha[...]
    )
    assert jnp.allclose(
        original_model.layers[1].beta[...], imported_model.layers[1].beta[...]
    )


def test_nir_export_import_rlif():
    rngs = nnx.Rngs(0)
    original_model = nn.Sequential(
        nnx.Linear(8, 12, use_bias=False, rngs=rngs),
        nn.RLIF((12,), beta=0.85, rngs=rngs),
    )

    input_shape = {"input": (8,)}
    output_shape = {"output": (12,)}
    nir_graph = spyx_nir.to_nir(original_model, input_shape, output_shape)
    imported_model = spyx_nir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))

    # Linear preceeding the recurrent block.
    assert jnp.allclose(
        original_model.layers[0].kernel[...], imported_model.layers[0].kernel[...]
    )
    # Recurrent weights should round-trip exactly.
    assert jnp.allclose(
        original_model.layers[1].recurrent_w[...],
        imported_model.layers[1].recurrent_w[...],
    )
    assert jnp.allclose(
        original_model.layers[1].beta[...], imported_model.layers[1].beta[...]
    )


def test_nir_export_import_rcubalif():
    rngs = nnx.Rngs(0)
    original_model = nn.Sequential(
        nnx.Linear(6, 6, use_bias=False, rngs=rngs),
        nn.RCuBaLIF((6,), alpha=0.95, beta=0.9, rngs=rngs),
    )

    input_shape = {"input": (6,)}
    output_shape = {"output": (6,)}
    nir_graph = spyx_nir.to_nir(original_model, input_shape, output_shape)
    imported_model = spyx_nir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))

    assert jnp.allclose(
        original_model.layers[1].recurrent_w[...],
        imported_model.layers[1].recurrent_w[...],
    )
    assert jnp.allclose(
        original_model.layers[1].alpha[...], imported_model.layers[1].alpha[...]
    )
    assert jnp.allclose(
        original_model.layers[1].beta[...], imported_model.layers[1].beta[...]
    )


def test_nir_export_import_if():
    """IF export previously passed r=1 (int) to nir.IF, which requires arrays."""
    rngs = nnx.Rngs(0)
    model = nn.Sequential(nnx.Linear(10, 8, rngs=rngs), nn.IF((8,)))

    graph = spyx_nir.to_nir(model, {"input": (10,)}, {"output": (8,)})
    imported = spyx_nir.from_nir(graph, dt=1, rngs=nnx.Rngs(1))

    x = jax.random.normal(jax.random.PRNGKey(42), (5, 10))
    out_orig, _ = model(x, model.initial_state(5))
    out_imp, _ = imported(x, imported.initial_state(5))
    assert jnp.allclose(out_orig, out_imp)


def test_nir_export_import_flatten():
    """Flatten export previously built a malformed nir.Flatten (nested dict)."""
    rngs = nnx.Rngs(0)
    model = nn.Sequential(
        nn.Flatten(),
        nnx.Linear(2 * 3 * 3, 8, rngs=rngs),
        nn.LIF((8,), beta=0.8, rngs=rngs),
    )

    graph = spyx_nir.to_nir(model, {"input": (2, 3, 3)}, {"output": (8,)})
    # NIR flatten node carries the flattened shape.
    assert int(graph.nodes["layer_0"].output_type["output"].prod()) == 2 * 3 * 3

    imported = spyx_nir.from_nir(graph, dt=1, rngs=nnx.Rngs(1))
    x = jax.random.normal(jax.random.PRNGKey(42), (5, 2, 3, 3))
    out_orig, _ = model(x, model.initial_state(5))
    out_imp, _ = imported(x, imported.initial_state(5))
    assert out_orig.shape == (5, 8)
    assert jnp.allclose(out_orig, out_imp)


if __name__ == "__main__":
    test_nir_export_import_lif()
    test_nir_export_import_cubalif()
    test_nir_export_import_rlif()
    test_nir_export_import_rcubalif()
    test_nir_export_import_if()
    test_nir_export_import_flatten()
    print("NIR tests passed!")
