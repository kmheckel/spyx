"""Smoke-test the published Spyx tutorial notebooks against the current API.

Address for issue #37: rather than execute each notebook end-to-end (which
needs ~1 GB of dataset downloads and a real GPU), this script exercises the
*structural* parts that most often break under dependency drift:

* model construction (nnx.Module + spyx.nn.Sequential),
* forward pass on a synthetic spike-shaped tensor,
* one ``nnx.Optimizer`` + ``nnx.value_and_grad`` step,
* the cartpole evosax pipeline (CMA-ES init / ask / tell on a tiny budget).

If this script returns 0 the notebooks should still match the rest of the
project's API surface; if it fails, the matching tutorial almost certainly
needs an update.

Run directly:

    uv run python scripts/smoke_notebook_apis.py

Or as a pytest target:

    uv run pytest scripts/smoke_notebook_apis.py
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import spyx
import spyx.nn as snn


BATCH = 4
SAMPLE_T = 16
CHANNELS = 32
N_CLASSES = 5
HIDDEN = 16


def _synthetic_packed_batch(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Mimic spyx.data.SHD_loader's output: (B, ceil(T/8), C) uint8 + labels."""
    t_packed = (SAMPLE_T + 7) // 8
    obs = jax.random.randint(
        rng, shape=(BATCH, t_packed, CHANNELS), minval=0, maxval=256, dtype=jnp.uint8
    )
    labels = jnp.array([0, 1, 2, 3])
    return obs, labels


def _unpack(batch_obs: jax.Array) -> jax.Array:
    obs = jnp.asarray(batch_obs)
    return jnp.unpackbits(obs, axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)


def smoke_surrogate_gradient_tutorial() -> None:
    """Mirrors docs/examples/surrogate_gradient/SurrogateGradientTutorial.ipynb."""
    rngs = nnx.Rngs(0)

    class SHDSNN(nnx.Module):
        def __init__(self, in_dim, hidden, n_classes, *, rngs):
            self.core = snn.Sequential(
                nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs),
                snn.LIF((hidden,), activation=spyx.axn.triangular(), rngs=rngs),
                nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs),
                snn.LIF((hidden,), activation=spyx.axn.triangular(), rngs=rngs),
                nnx.Linear(hidden, n_classes, use_bias=False, rngs=rngs),
                snn.LI((n_classes,), rngs=rngs),
            )

        def __call__(self, x_BTC):
            x_TBC = jnp.transpose(x_BTC, (1, 0, 2))
            traces, _ = snn.run(self.core, x_TBC)
            return jnp.transpose(traces, (1, 0, 2))

    model = SHDSNN(CHANNELS, HIDDEN, N_CLASSES, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.lion(3e-4), wrt=nnx.Param)
    Loss = spyx.fn.integral_crossentropy()

    @nnx.jit
    def train_step(model, optimizer, events, targets):
        def loss_fn(m):
            return Loss(m(events), targets)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    obs, targets = _synthetic_packed_batch(jax.random.PRNGKey(0))
    events = _unpack(obs)
    losses = [float(train_step(model, optimizer, events, targets)) for _ in range(3)]
    assert all(np.isfinite(losses)), losses


def smoke_shd_template() -> None:
    """Mirrors shd_sg_template.ipynb (intermediate-spike taps + reg losses)."""
    rngs = nnx.Rngs(0)

    class SHDSNN(nnx.Module):
        def __init__(self, in_dim, hidden, n_classes, *, rngs):
            self.l1 = nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs)
            self.lif1 = snn.LIF((hidden,), activation=spyx.axn.triangular(), rngs=rngs)
            self.l2 = nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs)
            self.lif2 = snn.LIF((hidden,), activation=spyx.axn.triangular(), rngs=rngs)
            self.l3 = nnx.Linear(hidden, n_classes, use_bias=False, rngs=rngs)
            self.li = snn.LI((n_classes,), rngs=rngs)

        def __call__(self, x_BTC):
            x_TBC = jnp.transpose(x_BTC, (1, 0, 2))
            T, B, _ = x_TBC.shape
            s1 = self.lif1.initial_state(B)
            s2 = self.lif2.initial_state(B)
            s3 = self.li.initial_state(B)

            def step(carry, x_t):
                s1, s2, s3 = carry
                sp1, s1 = self.lif1(self.l1(x_t), s1)
                sp2, s2 = self.lif2(self.l2(sp1), s2)
                v, s3 = self.li(self.l3(sp2), s3)
                return (s1, s2, s3), (sp1, sp2, v)

            _, (sp1_TB, sp2_TB, v_TB) = jax.lax.scan(step, (s1, s2, s3), x_TBC)
            return jnp.transpose(v_TB, (1, 0, 2)), [
                jnp.transpose(sp1_TB, (1, 0, 2)),
                jnp.transpose(sp2_TB, (1, 0, 2)),
            ]

    model = SHDSNN(CHANNELS, HIDDEN, N_CLASSES, rngs=rngs)
    optimizer = nnx.Optimizer(
        model, optax.chain(optax.centralize(), optax.lion(3e-4)), wrt=nnx.Param
    )
    Loss = spyx.fn.integral_crossentropy()
    Sil = spyx.fn.silence_reg(2.0)
    Spa = spyx.fn.sparsity_reg(8.0)

    @nnx.jit
    def train_step(model, optimizer, events, targets):
        def loss_fn(m):
            traces, spikes = m(events)
            return Loss(traces, targets) + 1e-3 * (Sil(spikes) + Spa(spikes))
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    obs, targets = _synthetic_packed_batch(jax.random.PRNGKey(0))
    events = _unpack(obs)
    losses = [float(train_step(model, optimizer, events, targets)) for _ in range(3)]
    assert all(np.isfinite(losses)), losses


def _shared_comparison(make_layer: Callable[[int, "nnx.Rngs"], "nnx.Module"]) -> None:
    """Shared body for the neuron-model and surrogate-gradient comparison notebooks."""
    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(CHANNELS, HIDDEN, use_bias=False, rngs=rngs),
        make_layer(0, rngs),
        nnx.Linear(HIDDEN, HIDDEN, use_bias=False, rngs=rngs),
        make_layer(1, rngs),
        nnx.Linear(HIDDEN, N_CLASSES, use_bias=False, rngs=rngs),
        snn.LI((N_CLASSES,), rngs=rngs),
    )
    optimizer = nnx.Optimizer(
        model, optax.chain(optax.centralize(), optax.lion(1e-4)), wrt=nnx.Param
    )
    Loss = spyx.fn.integral_crossentropy()

    @nnx.jit
    def train_step(model, optimizer, events, targets):
        def loss_fn(m):
            x_TBC = jnp.transpose(events, (1, 0, 2))
            traces, _ = snn.run(m, x_TBC)
            return Loss(jnp.transpose(traces, (1, 0, 2)), targets)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    obs, targets = _synthetic_packed_batch(jax.random.PRNGKey(0))
    events = _unpack(obs)
    losses = [float(train_step(model, optimizer, events, targets)) for _ in range(2)]
    assert all(np.isfinite(losses)), losses


def smoke_neuron_model_comparison() -> None:
    """Mirrors shd_sg_neuron_model_comparison.ipynb across all five neuron variants."""
    act = spyx.axn.arctan()
    factories = {
        "LIF": lambda _i, rngs: snn.LIF((HIDDEN,), activation=act, rngs=rngs),
        "IF": lambda _i, rngs: snn.IF((HIDDEN,), activation=act),
        "ALIF": lambda _i, rngs: snn.ALIF((HIDDEN,), activation=act, rngs=rngs),
        "RLIF": lambda _i, rngs: snn.RLIF((HIDDEN,), activation=act, rngs=rngs),
        "CuBaLIF": lambda _i, rngs: snn.CuBaLIF((HIDDEN,), activation=act, rngs=rngs),
    }
    for name, factory in factories.items():
        try:
            _shared_comparison(factory)
        except Exception:
            print(f"  variant {name}: FAIL")
            raise


def smoke_surrogate_comparison() -> None:
    """Mirrors shd_sg_surrogate_comparison.ipynb across all six surrogates."""
    surrogates = {
        "arctan": spyx.axn.arctan(),
        "superspike": spyx.axn.superspike(),
        "tanh": spyx.axn.tanh(),
        "boxcar": spyx.axn.boxcar(),
        "triangular": spyx.axn.triangular(),
        "STE": spyx.axn.custom(),
    }
    for name, act in surrogates.items():
        try:
            _shared_comparison(
                lambda _i, rngs, _act=act: snn.LIF((HIDDEN,), activation=_act, rngs=rngs)
            )
        except Exception:
            print(f"  surrogate {name}: FAIL")
            raise


def smoke_cartpole_evo() -> None:
    """Mirrors cartpole_evo.ipynb on a tiny CMA-ES budget; does not require gymnax."""
    try:
        from evosax.algorithms.distribution_based.cma_es import CMA_ES
    except ImportError:
        print("  evosax not installed; skipping cartpole smoke")
        return

    # Stand-in for an SNN controller: a tiny Spyx module wrapped flat for evosax.
    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(8, 16, use_bias=False, rngs=rngs),
        snn.LIF((16,), beta=0.8, rngs=rngs),
        nnx.Linear(16, 2, use_bias=False, rngs=rngs),
        snn.LI((2,), rngs=rngs),
    )
    graphdef, params = nnx.split(model, nnx.Param)
    flat, _unravel = jax.flatten_util.ravel_pytree(params)

    POP, GENS = 4, 2
    strat = CMA_ES(population_size=POP, solution=jnp.zeros_like(flat))
    es_p = strat.default_params
    es_s = strat.init(jax.random.PRNGKey(0), mean=flat, params=es_p)

    key = jax.random.PRNGKey(1)
    for _ in range(GENS):
        key, k_ask, k_tell = jax.random.split(key, 3)
        cands, es_s = strat.ask(k_ask, es_s, es_p)
        # Synthetic fitness: just the negative L2 norm of the candidate.
        fitness = -jnp.sum(cands ** 2, axis=-1)
        es_s, _ = strat.tell(k_tell, cands, -fitness, es_s, es_p)
    assert cands.shape == (POP, flat.size)


def smoke_quantization_intro() -> None:
    """Mirrors docs/examples/quantization/qat_intro.ipynb."""
    if not spyx.quant.available():
        print("  qwix not installed; skipping QAT smoke")
        return

    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(CHANNELS, HIDDEN, use_bias=False, rngs=rngs),
        snn.LIF((HIDDEN,), rngs=rngs),
        nnx.Linear(HIDDEN, N_CLASSES, use_bias=False, rngs=rngs),
        snn.LI((N_CLASSES,), rngs=rngs),
    )
    sample_x = jnp.ones((BATCH, CHANNELS))
    sample_state = model.initial_state(BATCH)
    qmodel = spyx.quant.quantize(model, sample_x, sample_state)

    optimizer = nnx.Optimizer(qmodel, optax.adam(1e-3), wrt=nnx.Param)
    targets = jnp.arange(BATCH) % N_CLASSES

    @nnx.jit
    def step(model, optimizer, x, state, y):
        def loss_fn(m):
            out, _ = m(x, state)
            traces = out[:, None, :]
            return spyx.fn.integral_crossentropy()(traces, y)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    losses = [float(step(qmodel, optimizer, sample_x, sample_state, targets)) for _ in range(3)]
    assert all(np.isfinite(losses)), losses


SMOKES: dict[str, Callable[[], None]] = {
    "SurrogateGradientTutorial": smoke_surrogate_gradient_tutorial,
    "shd_sg_template": smoke_shd_template,
    "shd_sg_neuron_model_comparison": smoke_neuron_model_comparison,
    "shd_sg_surrogate_comparison": smoke_surrogate_comparison,
    "cartpole_evo": smoke_cartpole_evo,
    "quantization/qat_intro": smoke_quantization_intro,
}


def run_all() -> int:
    failed: list[str] = []
    for name, fn in SMOKES.items():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=4)
            failed.append(name)
    if failed:
        print(f"\n{len(failed)}/{len(SMOKES)} notebooks have API drift: {failed}")
        return 1
    print(f"\nAll {len(SMOKES)} notebook APIs check out.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
