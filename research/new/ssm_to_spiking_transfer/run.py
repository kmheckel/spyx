"""SSM -> spiking transfer: pretrain an S5Diag backbone, transplant its diagonal
eigenvalues into ResonateFire, finetune the spiking neuron.

Scientific claim under test
---------------------------
``spyx.ssm.S5Diag`` is a stable, fully-parallel diagonal complex SSM whose
discrete diagonal eigenvalue has *the same functional form* as the pole of the
spiking ``spyx.phasor.ResonateFire`` neuron. So a well-behaved S5Diag backbone,
which is cheap to pretrain with an ``associative_scan`` in O(log T) depth, can
serve as a *warm start* for the harder-to-train spiking resonate-and-fire neuron:
extract S5's per-state continuous diagonal, map it onto ResonateFire's
``(lambda, omega)``, and finetune. We measure whether this beats training
ResonateFire from scratch on the same budget (accuracy and wall-clock).

The eigenvalue mapping (the scientifically load-bearing part)
------------------------------------------------------------
``S5Diag`` stores a *continuous-time* diagonal ``A = A_re + i*A_im`` (init
HiPPO-LegS ``-1/2 + i*pi*n``) plus a per-state learnable log-step ``log_dt``. Its
realised **discrete** pole is (see ``S5Diag._complex_matrices``)::

    dt_s5 = exp(log_dt)  # per state
    lam = exp(A * dt_s5)  # complex64, |lam| < 1 while Re(A) < 0

``ResonateFire`` stores a non-negative decay ``lambda = softplus(raw_lambda)`` and
an angular frequency ``omega`` (both real params), and realises the pole
(see ``ResonateFire.a``)::

    a = exp(dt_rf * (-lambda + i*omega))

Setting ResonateFire's ``dt_rf = 1`` and *folding S5's per-state ``dt_s5`` into
the transferred continuous diagonal* makes the two poles algebraically identical:

    want   a == lam
         = exp(A * dt_s5) = exp( Re(A)*dt_s5 + i*Im(A)*dt_s5 )
    with dt_rf = 1:
         a = exp( -lambda + i*omega )
    =>   lambda = -Re(A) * dt_s5
         omega  =  Im(A) * dt_s5
         raw_lambda = inverse_softplus(lambda)

This is an *exact* transfer (we assert ``max|a - lam| < tol`` post-transfer),
subject to one honest caveat: ResonateFire enforces ``lambda >= 0`` (``|a| <= 1``)
by construction, so any S5 state that drifted to ``Re(A) >= 0`` during pretraining
(an unstable, magnitude-growing pole ``|lam| >= 1``) *cannot* be represented and
is clamped to the stability boundary. We count and report those.

Directly-compatible linear weights are transferred too: both classifiers wrap the
neuron/SSM in the *same* real ``encoder`` (in -> hidden) and ``readout``
(hidden -> classes) ``nnx.Linear`` layers, so those kernels/biases copy across
verbatim. S5Diag's *internal* complex ``B``/``C``/``D`` are SSM mechanics with no
ResonateFire counterpart and are deliberately not transferred (documented
limitation).

Data
----
Real mode trains on SHD (bit-packed .npz via ``SHD_CACHE``, same mechanism as
``research/new/parallel_spiking_neurons/run_study.py``). ``SMOKE=1`` runs a tiny
synthetic classification task end-to-end on CPU, exercising the full transfer
path (pretrain -> extract -> assign -> assert pole match -> finetune -> baseline).

Run::

    SMOKE=1 uv run python research/new/ssm_to_spiking_transfer/run.py
    SHD_CACHE=/path/shd_cache.npz EPOCHS=40 uv run python .../run.py
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import spyx
import spyx.axn
import spyx.fn
from spyx.phasor import ResonateFire
from spyx.ssm import S5Diag

SMOKE = bool(os.environ.get("SMOKE"))

if SMOKE:
    SAMPLE_T, CHANNELS, N_CLASSES, HIDDEN = 16, 16, 4, 16
    PRETRAIN_EPOCHS = int(os.environ.get("EPOCHS", "3"))
    FINETUNE_EPOCHS = PRETRAIN_EPOCHS
else:
    SAMPLE_T, CHANNELS, N_CLASSES, HIDDEN = 128, 128, 20, 64
    PRETRAIN_EPOCHS = int(os.environ.get("EPOCHS", "40"))
    FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", str(PRETRAIN_EPOCHS)))

CACHE = os.environ.get("SHD_CACHE", "./shd_cache.npz")
POLE_TOL = 1e-3
DECAY_FLOOR = 1e-4  # min representable ResonateFire decay (|a| just under 1)


# --------------------------------------------------------------------------- data
def _synthetic_data():
    """Tiny separable synthetic spike dataset (SMOKE mode).

    Each class has a dedicated channel band that spikes at an elevated rate, so
    an SSM/spiking classifier can actually reduce the loss — enough signal to
    exercise (not benchmark) the transfer path end-to-end on CPU.
    """
    rng = np.random.default_rng(0)
    band = CHANNELS // N_CLASSES

    def make(n_batches, batch):
        obs, lab = [], []
        for _ in range(n_batches):
            labels = rng.integers(0, N_CLASSES, size=batch)
            x = (rng.random((batch, SAMPLE_T, CHANNELS)) < 0.05).astype(np.float32)
            for b in range(batch):
                lo = labels[b] * band
                x[b, :, lo : lo + band] += (rng.random((SAMPLE_T, band)) < 0.3).astype(
                    np.float32
                )
            obs.append(np.clip(x, 0.0, 1.0))
            lab.append(labels.astype(np.int32))
        return jnp.asarray(np.stack(obs)), jnp.asarray(np.stack(lab))

    tro, trl = make(3, 8)
    teo, tel = make(2, 8)
    return tro, trl, teo, tel


def _build_cache(path):
    import spyx.data

    dl = spyx.data.SHD_loader(
        batch_size=256, sample_T=SAMPLE_T, channels=CHANNELS, worker_count=16
    )

    def stage(epoch):
        obs, lab = [], []
        for b in epoch:
            obs.append(np.asarray(b.obs))
            lab.append(np.asarray(b.labels))
        return np.stack(obs), np.stack(lab)

    tro, trl = stage(dl.train_epoch())
    teo, tel = stage(dl.test_epoch())
    np.savez(path, train_obs=tro, train_labels=trl, test_obs=teo, test_labels=tel)


def _load_shd():
    if not os.path.exists(CACHE):
        print(f"prestaging SHD -> {CACHE} (one-time, slow) ...", flush=True)
        _build_cache(CACHE)
    d = np.load(CACHE)

    def unpack(p):  # (N,B,T_packed,C) uint8 -> (N,B,T,C) f32
        return np.unpackbits(p, axis=2)[:, :, :SAMPLE_T, :].astype(np.float32)

    return (
        jnp.asarray(unpack(d["train_obs"])),
        jnp.asarray(d["train_labels"]),
        jnp.asarray(unpack(d["test_obs"])),
        jnp.asarray(d["test_labels"]),
    )


def load_data():
    return _synthetic_data() if SMOKE else _load_shd()


# ------------------------------------------------------------------------- models
class S5Classifier(nnx.Module):
    """encoder (real Linear) -> S5Diag SSM -> readout (real Linear), time-major.

    Output is a ``(B, T, n_classes)`` logit trace so the same
    ``spyx.fn.integral_*`` loss/metric (sum over the time axis) scores it exactly
    like the spiking classifier.
    """

    def __init__(self, in_dim, hidden, n_classes, *, rngs):
        self.encoder = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.ssm = S5Diag(d_model=hidden, d_state=hidden, use_skip=True, rngs=rngs)
        self.readout = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, x_BTC):
        x = jnp.transpose(x_BTC, (1, 0, 2))  # (T, B, C)
        h = self.encoder(x)
        y = self.ssm(h)  # (T, B, H)
        logits = self.readout(y)  # (T, B, n_classes)
        return jnp.transpose(logits, (1, 0, 2))  # (B, T, n_classes)


class RFClassifier(nnx.Module):
    """encoder (real Linear) -> ResonateFire (parallel scan) -> readout (real Linear).

    Same outer ``encoder``/``readout`` shapes as :class:`S5Classifier`, so those
    weights transfer verbatim; the SSM core is swapped for the spiking neuron.
    """

    def __init__(self, in_dim, hidden, n_classes, *, rngs):
        self.encoder = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.neuron = ResonateFire(
            (hidden,), activation=spyx.axn.triangular(), rngs=rngs
        )
        self.readout = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, x_BTC):
        x = jnp.transpose(x_BTC, (1, 0, 2))  # (T, B, C)
        h = self.encoder(x)  # (T, B, H)
        spikes = self.neuron.parallel(h)  # (T, B, H)
        logits = self.readout(spikes)  # (T, B, n_classes)
        return jnp.transpose(logits, (1, 0, 2))  # (B, T, n_classes)


# ------------------------------------------------------------------- eigen transfer
def _inverse_softplus(y):
    """x such that softplus(x) == y (requires y > 0). Matches phasor._inverse_softplus."""
    return jnp.log(jnp.expm1(y))


def transfer_eigenvalues(s5_model: S5Classifier, rf_model: RFClassifier) -> dict:
    """Transplant S5Diag's discrete diagonal poles into ResonateFire.

    See the module docstring for the derivation. Returns diagnostics including
    the post-transfer pole-match error (asserted < POLE_TOL) and the number of
    unstable S5 states that had to be clamped to the stability boundary.
    """
    ssm = s5_model.ssm
    A = (ssm.A_re[...] + 1j * ssm.A_im[...]).astype(jnp.complex64)
    dt_s5 = jnp.exp(ssm.log_dt[...]).astype(jnp.float32)  # per-state step, shape (H,)

    # Target: S5's realised discrete pole (identical expression to S5._complex_matrices).
    lam = jnp.exp(A * dt_s5.astype(A.dtype))

    # ResonateFire uses dt_rf = 1 (module default); fold dt_s5 into the diagonal.
    decay_raw = -jnp.real(A) * dt_s5  # = lambda before the stability clamp
    omega = jnp.imag(A) * dt_s5
    unstable = decay_raw < DECAY_FLOOR  # Re(A) >= 0 -> |lam| >= 1, not representable
    decay = jnp.clip(decay_raw, DECAY_FLOOR, None)

    assert rf_model.neuron.dt == 1.0, "transfer assumes ResonateFire dt == 1.0"
    rf_model.neuron.raw_lambda.value = _inverse_softplus(decay).astype(jnp.float32)
    rf_model.neuron.omega.value = omega.astype(jnp.float32)

    # Directly-compatible outer real linear weights copy across verbatim.
    rf_model.encoder.kernel.value = s5_model.encoder.kernel[...]
    rf_model.encoder.bias.value = s5_model.encoder.bias[...]
    rf_model.readout.kernel.value = s5_model.readout.kernel[...]
    rf_model.readout.bias.value = s5_model.readout.bias[...]

    # Assert the realised ResonateFire pole matches S5's pole where representable.
    a_rf = rf_model.neuron.a
    stable = ~unstable
    if bool(jnp.any(stable)):
        pole_err = float(jnp.max(jnp.abs(a_rf - lam)[stable]))
    else:
        pole_err = 0.0
    assert pole_err < POLE_TOL, (
        f"eigenvalue transfer failed: max|a - lam| = {pole_err:.3e} >= {POLE_TOL}"
    )

    return {
        "pole_max_err": pole_err,
        "n_states": int(A.shape[0]),
        "n_clamped_unstable": int(jnp.sum(unstable)),
        "lam_mag_min": float(jnp.abs(lam).min()),
        "lam_mag_max": float(jnp.abs(lam).max()),
    }


# --------------------------------------------------------------------- train / eval
def evaluate(model, data):
    """Test accuracy only — never touches the params (safe for @init readout)."""
    _, _, teo, tel = data
    acc_fn = spyx.fn.integral_accuracy()
    return float(
        jnp.mean(
            jnp.stack([acc_fn(model(teo[i]), tel[i])[0] for i in range(teo.shape[0])])
        )
    )


def train(model, data, epochs, label, lr=3e-3):
    tro, trl, teo, tel = data
    loss_fn = spyx.fn.integral_crossentropy()
    acc_fn = spyx.fn.integral_accuracy()
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def step(m, o, ob, lb):
        def lf(mm):
            return loss_fn(mm(ob), lb)

        loss, g = nnx.value_and_grad(lf)(m)
        o.update(m, g)
        return loss

    def test_acc(m):
        return float(
            jnp.mean(
                jnp.stack([acc_fn(m(teo[i]), tel[i])[0] for i in range(teo.shape[0])])
            )
        )

    # Warm compile excluded from the timed loop.
    step(model, opt, tro[0], trl[0])
    jax.block_until_ready(test_acc(model))

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for i in range(tro.shape[0]):
            step(model, opt, tro[i], trl[i])
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(
                f"  [{label}] epoch {ep:3d} test_acc={test_acc(model) * 100:.2f}%",
                flush=True,
            )
    train_s = time.perf_counter() - t0
    acc = test_acc(model)
    print(f"  [{label}] FINAL acc={acc * 100:.2f}%  train={train_s:.2f}s", flush=True)
    return acc, train_s


# ---------------------------------------------------------------------------- main
def main():
    print(
        f"backend={jax.default_backend()}  device={jax.devices()[0]}  SMOKE={SMOKE}",
        flush=True,
    )
    data = load_data()
    print(
        f"data: train {tuple(data[0].shape)}  test {tuple(data[2].shape)}  "
        f"hidden={HIDDEN}  pretrain_epochs={PRETRAIN_EPOCHS}\n",
        flush=True,
    )

    # 1) Pretrain the fast, parallel, well-behaved S5Diag backbone.
    print("== 1. pretrain S5Diag backbone ==", flush=True)
    s5 = S5Classifier(CHANNELS, HIDDEN, N_CLASSES, rngs=nnx.Rngs(0))
    s5_acc, s5_train_s = train(s5, data, PRETRAIN_EPOCHS, "S5Diag")

    # 2) Build ResonateFire classifier and transfer eigenvalues + linear weights.
    print("\n== 2. transfer eigenvalues S5Diag -> ResonateFire ==", flush=True)
    rf_transfer = RFClassifier(CHANNELS, HIDDEN, N_CLASSES, rngs=nnx.Rngs(1))
    diag = transfer_eigenvalues(s5, rf_transfer)
    print(
        f"  pole match: max|a - lam|={diag['pole_max_err']:.2e} (< {POLE_TOL})  "
        f"clamped {diag['n_clamped_unstable']}/{diag['n_states']} unstable states\n"
        f"  S5 discrete |lam| in [{diag['lam_mag_min']:.3f}, {diag['lam_mag_max']:.3f}]",
        flush=True,
    )
    acc0 = evaluate(rf_transfer, data)  # pre-finetune accuracy (no param update)
    print(f"  RF-transfer @init acc={acc0 * 100:.2f}% (before any finetuning)")

    # 3) Finetune the transferred ResonateFire spiking neuron.
    print("\n== 3. finetune transferred ResonateFire ==", flush=True)
    rf_ft_acc, rf_ft_train_s = train(rf_transfer, data, FINETUNE_EPOCHS, "RF-transfer")

    # 4) Baseline: ResonateFire trained from scratch, same budget.
    print("\n== 4. baseline: ResonateFire from scratch ==", flush=True)
    rf_scratch = RFClassifier(CHANNELS, HIDDEN, N_CLASSES, rngs=nnx.Rngs(1))
    rf_sc_acc, rf_sc_train_s = train(rf_scratch, data, FINETUNE_EPOCHS, "RF-scratch")

    print("\n== summary ==", flush=True)
    print(f"  S5Diag backbone        acc={s5_acc * 100:.2f}%  ({s5_train_s:.2f}s)")
    print(f"  RF-transfer @init      acc={acc0 * 100:.2f}%  (no finetune)")
    print(
        f"  RF-transfer finetuned  acc={rf_ft_acc * 100:.2f}%  "
        f"({rf_ft_train_s:.2f}s finetune, {s5_train_s + rf_ft_train_s:.2f}s total)"
    )
    print(
        f"  RF-scratch baseline    acc={rf_sc_acc * 100:.2f}%  ({rf_sc_train_s:.2f}s)"
    )

    out = {
        "config": {
            "smoke": SMOKE,
            "sample_T": SAMPLE_T,
            "channels": CHANNELS,
            "n_classes": N_CLASSES,
            "hidden": HIDDEN,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
        },
        "transfer_diagnostics": diag,
        "results": {
            "s5_backbone": {"acc": s5_acc, "train_s": s5_train_s},
            "rf_transfer_at_init": {"acc": acc0},
            "rf_transfer_finetuned": {
                "acc": rf_ft_acc,
                "finetune_s": rf_ft_train_s,
                "total_s": s5_train_s + rf_ft_train_s,
            },
            "rf_scratch": {"acc": rf_sc_acc, "train_s": rf_sc_train_s},
        },
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "study_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote study_results.json")


if __name__ == "__main__":
    main()
