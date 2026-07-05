"""Language recipe: an SSM (S5) next-token model on a synthetic sequence.

**Experimental.** Part of :mod:`spyx.experimental.zoo`; the API may change
without a deprecation cycle.

A tiny character/token language model built on :class:`spyx.ssm.S5Diag`
predicts the next token in a synthetic repeating sequence (a cyclic
``0, 1, ..., V-1`` pattern with a random per-sample phase). Tokens are
embedded, run through the diagonal SSM over time, and read out to vocabulary
logits; the model is trained by standard gradient descent (softmax
cross-entropy on the shifted sequence). Because the sequence is fully periodic,
the loss drops quickly.

Synthetic data only, no downloads; a few steps run in well under a second on
CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ... import optimize, ssm

NAME = "language-s5"
APPLICATION = "language"
METHOD = "gradient"
ARCHITECTURE = "S5"
DESCRIBE = (
    "Diagonal-SSM (S5Diag) next-token language model on a synthetic repeating "
    "token sequence, trained by standard gradient descent."
)

VOCAB = 6
D_MODEL = 16
D_STATE = 16
SEQ_LEN = 32


class S5LanguageModel(nnx.Module):
    """Embed → :class:`spyx.ssm.S5Diag` → linear readout next-token model.

    ``__call__`` maps token ids ``(B, T)`` to logits ``(B, T, VOCAB)``. The SSM
    core operates time-major ``(T, B, D_MODEL)`` internally.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(VOCAB, D_MODEL, rngs=rngs)
        self.ssm = ssm.S5Diag(D_MODEL, D_STATE, rngs=rngs)
        self.readout = nnx.Linear(D_MODEL, VOCAB, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        x = self.embed(tokens)  # (B, T, D_MODEL)
        x = jnp.transpose(x, (1, 0, 2))  # (T, B, D_MODEL)
        y = self.ssm(x)  # (T, B, D_MODEL)
        logits = self.readout(y)  # (T, B, VOCAB)
        return jnp.transpose(logits, (1, 0, 2))  # (B, T, VOCAB)


def build(rngs: nnx.Rngs) -> nnx.Module:
    """Build the S5-based language model."""
    return S5LanguageModel(rngs=rngs)


def synthetic_batch(key: jax.Array | None = None, batch_size: int = 16) -> tuple:
    """Generate a batch of cyclic token sequences with random phase.

    Each sequence is ``(phase + t) mod VOCAB`` for ``t = 0 .. SEQ_LEN-1``.

    :return: a 1-tuple ``(tokens,)`` with ``tokens`` of shape ``[B, SEQ_LEN]``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    phase = jax.random.randint(key, (batch_size, 1), 0, VOCAB)
    tokens = (jnp.arange(SEQ_LEN)[None, :] + phase) % VOCAB
    return (tokens,)


def loss(model: nnx.Module, tokens: jax.Array) -> jax.Array:
    """Next-token softmax cross-entropy over the shifted sequence."""
    import optax  # lazy import; optax is a core dependency

    logits = model(tokens)  # (B, T, VOCAB)
    inputs = logits[:, :-1]
    targets = tokens[:, 1:]
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(inputs, targets))


def demo(steps: int = 40, *, seed: int = 0, learning_rate: float = 5e-3) -> list[float]:
    """Train a few gradient steps and return the next-token loss history.

    :steps: number of gradient steps.
    :seed: PRNG seed for model init and data.
    :learning_rate: Adam learning rate.
    :return: list of per-step training losses (should decrease).
    """
    import optax  # lazy import; optax is a core dependency

    key = jax.random.PRNGKey(seed)
    k_model, k_data = jax.random.split(key)
    model = build(nnx.Rngs(k_model))
    (tokens,) = synthetic_batch(k_data)

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    step = optimize.make_train_step(loss)

    history: list[float] = []
    for _ in range(steps):
        history.append(float(step(model, optimizer, tokens)))
    return history
