import jax
import jax.numpy as jnp
from flax import nnx

from spyx import nir as spyx_nir
from spyx import nn


def test_nir_export_import_lif():
    rngs = nnx.Rngs(0)
    # Define original model
    original_model = nn.Sequential(
        nnx.Linear(10, 20, rngs=rngs),
        nn.LIF((20,), beta=0.8, rngs=rngs)
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
    assert jnp.allclose(original_model.layers[0].kernel[...], imported_model.layers[0].kernel[...])
    # Check LIF beta
    assert jnp.allclose(original_model.layers[1].beta[...], imported_model.layers[1].beta[...])
    
    # Compare outputs for same input
    x = jax.random.normal(jax.random.PRNGKey(42), (5, 10)) # batch of 5
    
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
        nn.CuBaLIF((15,), alpha=0.9, beta=0.7, rngs=rngs)
    )
    
    input_shape = {"input": (10,)}
    output_shape = {"output": (15,)}
    nir_graph = spyx_nir.to_nir(original_model, input_shape, output_shape)
    
    imported_model = spyx_nir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))
    
    assert jnp.allclose(original_model.layers[0].kernel[...], imported_model.layers[0].kernel[...])
    assert jnp.allclose(original_model.layers[1].alpha[...], imported_model.layers[1].alpha[...])
    assert jnp.allclose(original_model.layers[1].beta[...], imported_model.layers[1].beta[...])

if __name__ == "__main__":
    test_nir_export_import_lif()
    test_nir_export_import_cubalif()
    print("NIR tests passed!")
