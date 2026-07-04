"""Sparse-memory routing vs a compressed-state SSM on synthetic associative recall.

Scientific claim under test
---------------------------
Raven (Afzal, Bick, Xing, Cevher, Gu, 2026; "High-recall sequence modeling with
sparse memory routing") argues that compressed-state recurrent models (a single
SSM state with uniform decay) fail at *exact recall* because every new token
perturbs the whole state, so stored key/value bindings interfere. Their fix --
**Routing Slot Memory (RSM)** -- partitions the state into ``M`` independent
slots and uses a learned **sparse router** to write only selected slots, leaving
the rest shielded from interference.

This study puts that claim to a controlled test on the synthetic multi-query
associative-recall (MQAR-style) task shipped in
:func:`spyx.raven.make_recall_batch`. We sweep a difficulty knob -- the number of
key/value bindings ``n_pairs`` (which also grows the sequence length
``T = 2*n_pairs + 1``) -- and compare three sequence models matched on width and
training budget:

* **S5Diag** -- a diagonal complex SSM baseline (``spyx.ssm.S5Diag``): the
  compressed-state model Raven says should degrade as bindings accumulate.
* **RavenRSM** -- the routing-slot memory (``spyx.raven.RavenRSM``): a bank of
  ``M`` slots written by a sparse (straight-through top-``k``) router.
* **SpikingSlotMemory** -- the spiking sibling (``spyx.raven.SpikingSlotMemory``):
  the same router over reset-free spiking (PSU_LIF-style) slot membranes.

**Hypothesis:** as ``n_pairs`` grows, the diagonal SSM's recall accuracy falls
off (interference in the shared state) while the slot-routed models sustain higher
recall (each binding can live in its own shielded slot).

Model wiring
------------
All three share the *same* outer scaffold so the comparison is about the recurrent
core, not the head: a real ``encoder`` Linear (one-hot token -> hidden), the core,
and a real ``readout`` Linear that maps the **final timestep** (the query step) to
``n_values`` class logits. The loss is a plain softmax cross-entropy on the query
step (a single next-token prediction), and accuracy is exact-match on the recalled
value id. Parameter counts are reported per model (matched as closely as practical
by choosing ``d_state`` / ``n_slots`` / ``d_slot``; not forced equal).

Reductions worth remembering (tested in ``tests/test_raven.py``, not re-derived
here): a *dense* router recovers a gated diagonal SSM; a one-hot cyclic router
recovers sliding-window attention. Here the slot models use a sparse top-``k``
router so the shielding is real (unselected slots pass through byte-for-byte).

Data & scale
------------
The recall task is fully synthetic, so the *entire* comparison is CPU-runnable at
small scale -- ``SMOKE=1`` runs every model across a small difficulty sweep in a
few seconds and prints the comparison table. The maintainer re-runs the larger
config on GPU. Only ``spyx.raven`` primitives and ``spyx.ssm.S5Diag`` are used.

Run::

    SMOKE=1 uv run python research/new/raven_sparse_memory_recall/run.py
    uv run python research/new/raven_sparse_memory_recall/run.py   # full config
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from spyx.experimental.raven import RavenRSM, SpikingSlotMemory, make_recall_batch
from spyx.ssm import S5Diag

SMOKE = bool(os.environ.get("SMOKE"))

if SMOKE:
    VOCAB = 8  # n_keys == n_values == VOCAB
    DIFFICULTIES = (2, 4)  # n_pairs sweep (harder = more bindings + longer T)
    HIDDEN = 16
    N_SLOTS = 4
    BATCH = 16
    N_TRAIN_BATCHES = 8
    N_TEST_BATCHES = 4
    EPOCHS = int(os.environ.get("EPOCHS", "15"))
else:
    VOCAB = 16
    DIFFICULTIES = (2, 4, 8, 16)
    HIDDEN = 64
    N_SLOTS = 8
    BATCH = 64
    N_TRAIN_BATCHES = 40
    N_TEST_BATCHES = 8
    EPOCHS = int(os.environ.get("EPOCHS", "60"))

# Difficulties may be overridden for quick experiments, e.g. DIFFICULTIES=2,3,5.
if os.environ.get("DIFFICULTIES"):
    DIFFICULTIES = tuple(int(x) for x in os.environ["DIFFICULTIES"].split(","))

LR = float(os.environ.get("LR", "3e-3"))
# Sparse router: keep only the k most-active slots per step (real shielding).
HARD_TOP_K = int(os.environ.get("HARD_TOP_K", str(max(1, N_SLOTS // 2))))
SEED = int(os.environ.get("SEED", "0"))


# --------------------------------------------------------------------------- data
def make_dataset(key, n_batches, *, n_pairs):
    """Stack ``n_batches`` MQAR batches into ``(u, target)`` arrays.

    ``u``: ``(n_batches, T, BATCH, d_model)`` one-hots; ``target``:
    ``(n_batches, BATCH)`` int32 value ids. ``d_model = 2 * VOCAB`` is constant
    across the difficulty sweep (only ``T`` and the binding count change).
    """
    us, ts = [], []
    for _ in range(n_batches):
        key, sub = jax.random.split(key)
        u, target = make_recall_batch(
            sub, batch=BATCH, n_pairs=n_pairs, n_keys=VOCAB, n_values=VOCAB
        )
        us.append(u)
        ts.append(target)
    return jnp.stack(us), jnp.stack(ts), key


# ------------------------------------------------------------------------- models
class SSMRecall(nnx.Module):
    """Baseline: encoder Linear -> S5Diag diagonal SSM -> readout on query step.

    The compressed-state model: a single diagonal state accumulates every token,
    so bindings interfere as ``n_pairs`` grows (the failure mode Raven targets).
    """

    def __init__(self, d_model, hidden, n_values, *, rngs):
        self.encoder = nnx.Linear(d_model, hidden, rngs=rngs)
        self.ssm = S5Diag(d_model=hidden, d_state=hidden, use_skip=True, rngs=rngs)
        self.readout = nnx.Linear(hidden, n_values, rngs=rngs)

    def __call__(self, u):  # u: (T, B, d_model) -> (B, n_values)
        y = self.ssm(self.encoder(u))  # (T, B, hidden)
        return self.readout(y[-1])  # readout the final (query) step


class RavenRecall(nnx.Module):
    """Routing-Slot Memory: encoder -> RavenRSM (sparse-routed slots) -> readout.

    RavenRSM already projects its slot read back to ``hidden``, so the head is
    identical to the SSM baseline's.
    """

    def __init__(self, d_model, hidden, n_values, n_slots, hard_top_k, *, rngs):
        self.encoder = nnx.Linear(d_model, hidden, rngs=rngs)
        self.core = RavenRSM(hidden, n_slots=n_slots, hard_top_k=hard_top_k, rngs=rngs)
        self.readout = nnx.Linear(hidden, n_values, rngs=rngs)

    def __call__(self, u):  # (T, B, d_model) -> (B, n_values)
        y = self.core(self.encoder(u))  # (T, B, hidden)
        return self.readout(y[-1])


class SpikingRecall(nnx.Module):
    """Spiking Routing-Slot Memory: encoder -> SpikingSlotMemory -> readout.

    The core emits a slot spike train ``(T, B, M, d_slot)``; the readout mean-pools
    the final-step spikes over slots to a ``hidden``-wide feature (dual sparsity:
    sparse in time and in slots). Pooling over slots keeps the readout width equal
    to the S5Diag / RavenRSM baselines so the comparison is capacity-fair (RavenRSM
    likewise pools its slots to ``hidden`` via its internal query-gated read).
    """

    def __init__(self, d_model, hidden, n_values, n_slots, hard_top_k, *, rngs):
        self.encoder = nnx.Linear(d_model, hidden, rngs=rngs)
        self.core = SpikingSlotMemory(
            hidden, n_slots=n_slots, hard_top_k=hard_top_k, rngs=rngs
        )
        self.readout = nnx.Linear(hidden, n_values, rngs=rngs)

    def __call__(self, u):  # (T, B, d_model) -> (B, n_values)
        s = self.core(self.encoder(u))  # (T, B, M, d_slot=hidden)
        last = s[-1].mean(axis=1)  # (B, d_slot) — mean-pool over slots
        return self.readout(last)


def build_models(d_model, n_values, rngs_seed):
    """Instantiate the three models with a shared seed (matched inits where shared)."""
    return {
        "S5Diag": SSMRecall(d_model, HIDDEN, n_values, rngs=nnx.Rngs(rngs_seed)),
        "RavenRSM": RavenRecall(
            d_model, HIDDEN, n_values, N_SLOTS, HARD_TOP_K, rngs=nnx.Rngs(rngs_seed)
        ),
        "SpikingSlotMemory": SpikingRecall(
            d_model, HIDDEN, n_values, N_SLOTS, HARD_TOP_K, rngs=nnx.Rngs(rngs_seed)
        ),
    }


def count_params(model):
    """Total trainable scalar count (nnx.Param leaves)."""
    params = nnx.state(model, nnx.Param)
    return int(sum(x.size for x in jax.tree.leaves(params)))


# --------------------------------------------------------------------- train / eval
def accuracy(model, us, ts):
    """Exact-match recall accuracy over a stacked dataset (no param updates)."""
    correct = jnp.stack(
        [
            jnp.mean((jnp.argmax(model(us[i]), -1) == ts[i]).astype(jnp.float32))
            for i in range(us.shape[0])
        ]
    )
    return float(jnp.mean(correct))


def train_model(model, train, test, *, epochs, lr):
    """Adam on softmax-CE at the query step. Returns (test_acc, train_seconds)."""
    tr_u, tr_t = train
    te_u, te_t = test
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, u, t):
        def lf(mm):
            return optax.softmax_cross_entropy_with_integer_labels(mm(u), t).mean()

        loss, g = nnx.value_and_grad(lf)(m)
        o.update(m, g)
        return loss

    # Warm compile excluded from the timed loop.
    step(model, opt, tr_u[0], tr_t[0])
    jax.block_until_ready(accuracy(model, te_u, te_t))

    t0 = time.perf_counter()
    for _ in range(epochs):
        for i in range(tr_u.shape[0]):
            step(model, opt, tr_u[i], tr_t[i])
    jax.block_until_ready(jax.tree.leaves(nnx.state(model, nnx.Param)))
    train_s = time.perf_counter() - t0
    return accuracy(model, te_u, te_t), train_s


# ---------------------------------------------------------------------------- main
def main():
    print(
        f"backend={jax.default_backend()}  device={jax.devices()[0]}  SMOKE={SMOKE}",
        flush=True,
    )
    d_model = 2 * VOCAB
    chance = 1.0 / VOCAB
    print(
        f"vocab={VOCAB}  d_model={d_model}  hidden={HIDDEN}  n_slots={N_SLOTS}  "
        f"top_k={HARD_TOP_K}  epochs={EPOCHS}  chance={chance * 100:.1f}%\n"
        f"difficulty sweep n_pairs={list(DIFFICULTIES)} "
        f"(T = 2*n_pairs+1)\n",
        flush=True,
    )

    key = jax.random.PRNGKey(SEED)
    model_names = ["S5Diag", "RavenRSM", "SpikingSlotMemory"]
    results = {name: {} for name in model_names}
    param_counts = {}

    for n_pairs in DIFFICULTIES:
        T = 2 * n_pairs + 1
        key, kt, kv = jax.random.split(key, 3)
        tr_u, tr_t, _ = make_dataset(kt, N_TRAIN_BATCHES, n_pairs=n_pairs)
        te_u, te_t, _ = make_dataset(kv, N_TEST_BATCHES, n_pairs=n_pairs)
        print(
            f"== difficulty n_pairs={n_pairs}  T={T}  "
            f"train={tuple(tr_u.shape)} test={tuple(te_u.shape)} ==",
            flush=True,
        )

        models = build_models(d_model, VOCAB, SEED)
        for name in model_names:
            model = models[name]
            param_counts[name] = count_params(model)
            acc, train_s = train_model(
                model, (tr_u, tr_t), (te_u, te_t), epochs=EPOCHS, lr=LR
            )
            results[name][n_pairs] = {"acc": acc, "train_s": train_s}
            print(
                f"  {name:18s}  acc={acc * 100:6.2f}%  "
                f"train={train_s:6.2f}s  params={param_counts[name]}",
                flush=True,
            )
        print(flush=True)

    # ------------------------------------------------------------------- table
    print("== recall accuracy (%) vs difficulty (n_pairs) ==", flush=True)
    header = "  model              " + "".join(f"  np={p:<4d}" for p in DIFFICULTIES)
    header += "   params"
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for name in model_names:
        row = f"  {name:18s}"
        for p in DIFFICULTIES:
            row += f"  {results[name][p]['acc'] * 100:6.2f}"
        row += f"   {param_counts[name]:>8d}"
        print(row, flush=True)
    print(f"\n  (chance = {chance * 100:.1f}%)", flush=True)

    print("\n== wall-clock train time (s) vs difficulty ==", flush=True)
    print(header.replace("   params", ""), flush=True)
    for name in model_names:
        row = f"  {name:18s}"
        for p in DIFFICULTIES:
            row += f"  {results[name][p]['train_s']:6.2f}"
        print(row, flush=True)

    out = {
        "config": {
            "smoke": SMOKE,
            "vocab": VOCAB,
            "d_model": d_model,
            "hidden": HIDDEN,
            "n_slots": N_SLOTS,
            "hard_top_k": HARD_TOP_K,
            "difficulties": list(DIFFICULTIES),
            "epochs": EPOCHS,
            "batch": BATCH,
            "n_train_batches": N_TRAIN_BATCHES,
            "n_test_batches": N_TEST_BATCHES,
            "lr": LR,
            "seed": SEED,
            "chance": chance,
        },
        "param_counts": param_counts,
        "results": {
            name: {str(p): results[name][p] for p in DIFFICULTIES}
            for name in model_names
        },
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "study_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote study_results.json", flush=True)


if __name__ == "__main__":
    main()
