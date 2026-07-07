"""Honest energy accounting for spiking networks: three numbers, side by side.

SNN "energy efficiency" claims almost always quote a single hardware-agnostic
proxy -- ``E_AC x (spikes x fanout x T)`` -- which counts *arithmetic only* and
ignores the memory traffic that dominates real silicon. This study reports
**three** analytic energy numbers per model so the claim can be made honestly:

1. **SOP proxy** (Horowitz ISSCC 2014, 45 nm): ``E_AC x SOPs``. The standard,
   flawed, compute-only number the SNN literature headlines. Included purely for
   comparability -- it is an under-count.
2. **Hardware-aware, memory-inclusive SNN energy** (Yan, Bai & Wong,
   arXiv:2409.08290): the SOP arithmetic *plus* weight-memory fetches (one per
   accumulate) *plus* membrane-state read/write every timestep. This is what the
   SNN actually costs.
3. **Bit-matched quantized-ANN baseline** (same paper): a rate-coded T-step SNN
   is functionally a ``ceil(log2(T+1))``-bit ANN of the *same architecture*. We
   cost that ANN's dense MACs + weight/activation memory, and report the honest
   **SNN / QANN energy ratio** -- never a bare SNN number.

The estimator is **analytic**: given a spike rate and T it evaluates all three in
closed form, so the crossover spike rate (where SNN energy == QANN energy) is
exact. Yan et al. report that for ``T in [5,10]`` an SNN only beats the matched
QANN below an average spike rate of ~6.4%. We reproduce that crossover, then
ground it by *training* a small spiking classifier
(``Linear -> LIF -> Linear -> LI``) at several drive levels / activity-regulariser
strengths and dropping the **measured** spike rates onto the curve.

HONESTY: every pJ constant here is a **literature value** (Horowitz 2014 for
45 nm compute; Yan et al. 2024 for 22 nm compute + memory). This is an *analytic
model*, not a silicon measurement. The crossover depends on the memory model and
on the SNN-vs-QANN weight-precision assumption (documented below); we state the
assumption and its sensitivity rather than hide it.

Run::

    SPYX_SMOKE=1 uv run python research/new/honest_energy_accounting/honest_energy.py
    uv run python research/new/honest_energy_accounting/honest_energy.py
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import spyx.axn as axn
import spyx.fn as fn
import spyx.nn as snn

SMOKE = bool(os.environ.get("SPYX_SMOKE") or os.environ.get("SMOKE"))
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Energy constants -- ALL literature values (see citations in the results JSON).
# --------------------------------------------------------------------------- #
# Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC
# 2014 -- 45 nm, 32-bit floating point. The arm-1 SOP proxy uses these.
E_ADD32_PJ = 0.9   # 32-bit FP add  -> the E_AC of the standard SOP proxy
E_MULT32_PJ = 3.7  # 32-bit FP multiply
E_MAC32_PJ = E_ADD32_PJ + E_MULT32_PJ  # 4.6 pJ, 32-bit FP multiply-accumulate

# Yan, Bai & Wong, "Reconsidering the energy efficiency of spiking neural
# networks", arXiv:2409.08290 -- 22 nm process. Per-operation compute costs:
E_ACC_PJ = 0.05448   # accumulation  (their E_ACC)
E_CMP_PJ = 0.05448   # comparison / threshold (their E_CMP)
E_SUB_PJ = 0.05448   # subtraction / reset    (their E_SUB)
# Data movement. Off-chip weight memory: their DRAM figure is ~1300 pJ per
# 64-bit word == 20.3 pJ/bit; we use 20 pJ/bit. On-chip (membrane state,
# activations) uses their *dense* data-movement figure, 0.25 pJ/bit/hop.
E_MEM_WEIGHT_PJ_PER_BIT = 20.0   # off-chip weight fetch (DRAM, 1300 pJ / 64 b)
E_MEM_ONCHIP_PJ_PER_BIT = 0.25   # on-chip state / activation movement
E_MEM_SPARSE_PJ_PER_BIT = 3.0    # Loihi-style sparse event routing (sensitivity)

# Precision assumptions.
#   * SNN weights & membrane state are kept at 8 bits (a typical post-QAT SNN).
#   * The bit-matched QANN uses ceil(log2(T+1)) bits (arXiv:2409.08290): a T-step
#     binary spike accumulator spans T+1 integer levels.
# This asymmetry (8-bit SNN weights vs 3-4-bit QANN weights) is what sets the
# ~6% crossover; see the sensitivity note in the JSON.
SNN_WEIGHT_BITS = 8
SNN_STATE_BITS = 8


def qann_bits(T: int) -> int:
    """Bit-matched ANN precision for a T-step rate-coded SNN (arXiv:2409.08290)."""
    return math.ceil(math.log2(T + 1))


def e_ac(bits: int) -> float:
    """Accumulate energy scaled from the 32-bit FP add (~linear in width)."""
    return E_ADD32_PJ * (bits / 32.0)


def e_mac(bits: int) -> float:
    """MAC energy: multiply ~quadratic in width, add ~linear (Horowitz)."""
    return E_MULT32_PJ * (bits / 32.0) ** 2 + E_ADD32_PJ * (bits / 32.0)


# --------------------------------------------------------------------------- #
# Reference architecture. The energy model and the trained classifier share it,
# so the measured spike rates and the analytic crossover are self-consistent.
# Large fan-in (C) is deliberate: it is the realistic regime where weight traffic
# dominates and the membrane-state term is a small correction (as in the paper).
# --------------------------------------------------------------------------- #
C_IN = 256    # input channels / fan-in
H_HID = 128   # hidden LIF neurons
N_OUT = 10    # readout (LI) classes

# Synapse (weight) counts and state-holding neuron count.
W1 = C_IN * H_HID   # input -> hidden
W2 = H_HID * N_OUT  # hidden -> output
W_TOTAL = W1 + W2
M_STATE = H_HID + N_OUT  # LIF hidden + LI output both carry a membrane potential


# --------------------------------------------------------------------------- #
# The three energy numbers, all analytic, all in pJ, per single input sample.
# `s` is the average spike rate applied to both the input encoding and the hidden
# layer (the paper's single average spike rate s_r).
# --------------------------------------------------------------------------- #
def energy_sop_proxy(s: float, T: int, s_hid: float | None = None) -> float:
    """Arm 1: hardware-agnostic SOP proxy, E_AC x SOPs (compute only).

    `s` drives the input (layer-1) synapses; `s_hid` the hidden (layer-2) ones,
    defaulting to `s` (the paper's single average spike rate).
    """
    if s_hid is None:
        s_hid = s
    sops = T * (s * W1 + s_hid * W2)
    return e_ac(SNN_WEIGHT_BITS) * sops


def energy_snn_memory_inclusive(s: float, T: int, s_hid: float | None = None) -> float:
    """Arm 2: memory-inclusive SNN energy (arXiv:2409.08290).

    SOP arithmetic + one weight fetch per accumulate + membrane read/write every
    timestep for every state-holding neuron. `s` is the input spike rate,
    `s_hid` the hidden spike rate (defaults to `s`).
    """
    if s_hid is None:
        s_hid = s
    sops = T * (s * W1 + s_hid * W2)
    # Compute: accumulate per SOP, threshold-compare every LIF neuron every step,
    # reset-subtract once per hidden spike.
    e_compute = sops * E_ACC_PJ + T * H_HID * E_CMP_PJ + (T * s_hid * H_HID) * E_SUB_PJ
    # Weight movement: every accumulate fetches one SNN_WEIGHT_BITS weight (DRAM).
    e_weight = sops * SNN_WEIGHT_BITS * E_MEM_WEIGHT_PJ_PER_BIT
    # Membrane state: read + write every neuron every timestep (on-chip SRAM).
    e_state = T * M_STATE * SNN_STATE_BITS * 2 * E_MEM_ONCHIP_PJ_PER_BIT
    return e_compute + e_weight + e_state


def energy_qann(T: int) -> float:
    """Arm 3: bit-matched quantized-ANN baseline (arXiv:2409.08290).

    One dense forward pass at b = ceil(log2(T+1)) bits: dense MACs + one weight
    fetch per MAC + activation movement.
    """
    b = qann_bits(T)
    e_compute = W_TOTAL * e_mac(b) + M_STATE * 2 * E_CMP_PJ
    e_weight = W_TOTAL * b * E_MEM_WEIGHT_PJ_PER_BIT
    e_act = (C_IN + H_HID + N_OUT) * b * E_MEM_ONCHIP_PJ_PER_BIT
    return e_compute + e_weight + e_act


def crossover_spike_rate(T: int) -> float:
    """Exact spike rate where SNN (arm 2) == QANN (arm 3).

    Arm 2 is affine in s: E_snn(s) = A*s + B. Solve A*s + B = E_qann.
    """
    b_const = energy_snn_memory_inclusive(0.0, T)          # B
    slope = energy_snn_memory_inclusive(1.0, T) - b_const  # A
    e_q = energy_qann(T)
    return (e_q - b_const) / slope


# --------------------------------------------------------------------------- #
# A real spyx spiking classifier, used to ground the analytic curve in *measured*
# spike rates driven by input rate and by spyx.fn activity regularisation.
# --------------------------------------------------------------------------- #
class SpikingClassifier(nnx.Module):
    """Linear -> LIF -> Linear -> LI, split so hidden spikes are observable."""

    def __init__(self, *, rngs):
        l1 = nnx.Linear(C_IN, H_HID, rngs=rngs)
        lif = snn.LIF((H_HID,), activation=axn.superspike(), rngs=rngs)
        l2 = nnx.Linear(H_HID, N_OUT, rngs=rngs)
        li = snn.LI((N_OUT,), rngs=rngs)
        self.enc = snn.Sequential(l1, lif)   # input -> hidden spikes
        self.dec = snn.Sequential(l2, li)    # hidden spikes -> readout

    def forward_hidden(self, x_TBC):
        """Return (output traces [B,T,N], hidden spikes [T,B,H])."""
        h_TBH, _ = snn.run(self.enc, x_TBC)
        o_TBN, _ = snn.run(self.dec, h_TBH)
        return jnp.transpose(o_TBN, (1, 0, 2)), h_TBH

    def __call__(self, x_TBC):
        return self.forward_hidden(x_TBC)[0]


ce = fn.integral_crossentropy(smoothing=0.2)
acc_fn = fn.integral_accuracy()


def synthetic_data(seed, T, input_p, n, batch):
    """Bernoulli spike trains with a per-class active band; `input_p` sets drive."""
    rng = np.random.default_rng(seed)
    band = max(1, C_IN // N_OUT)
    labels = rng.integers(0, N_OUT, size=n)
    x = (rng.random((n, T, C_IN)) < input_p * 0.5).astype(np.float32)
    for i in range(n):
        lo = labels[i] * band
        x[i, :, lo : lo + band] += (rng.random((T, band)) < input_p).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    obs, lab = [], []
    for start in range(0, n - batch + 1, batch):
        obs.append(jnp.transpose(jnp.asarray(x[start : start + batch]), (1, 0, 2)))
        lab.append(jnp.asarray(labels[start : start + batch].astype(np.int32)))
    return obs, lab


def train_and_measure(seed, T, input_p, reg_strength, reg_target, epochs, n, batch):
    """Train briefly, return (accuracy, mean input rate, mean hidden spike rate)."""
    model = SpikingClassifier(rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(5e-3), wrt=nnx.Param)
    sparsity = fn.sparsity_reg(reg_target) if reg_strength > 0 else None

    @nnx.jit
    def step(m, o, xb, yb):
        def loss(mm):
            out, hidden = mm.forward_hidden(xb)
            base = ce(out, yb)
            if sparsity is not None:
                base = base + reg_strength * sparsity(hidden)
            return base

        loss_val, grads = nnx.value_and_grad(loss)(m)
        o.update(m, grads)
        return loss_val

    xs, ys = synthetic_data(seed, T, input_p, n, batch)
    for _ in range(epochs):
        for xb, yb in zip(xs, ys, strict=True):
            step(model, opt, xb, yb)

    accs, in_rates, hid_rates = [], [], []
    for xb, yb in zip(xs, ys, strict=True):
        out, hidden = model.forward_hidden(xb)
        accs.append(float(acc_fn(out, yb)[0]))
        in_rates.append(float(jnp.mean((xb != 0).astype(jnp.float32))))
        hid_rates.append(float(jnp.mean((hidden != 0).astype(jnp.float32))))
    return float(np.mean(accs)), float(np.mean(in_rates)), float(np.mean(hid_rates))


# --------------------------------------------------------------------------- #
def _energies_at(s, T, s_hid=None):
    e1 = energy_sop_proxy(s, T, s_hid)
    e2 = energy_snn_memory_inclusive(s, T, s_hid)
    e3 = energy_qann(T)
    return {
        "spike_rate": s,
        "hidden_spike_rate": s if s_hid is None else s_hid,
        "T": T,
        "e_sop_proxy_pJ": e1,
        "e_snn_memory_inclusive_pJ": e2,
        "e_qann_pJ": e3,
        "snn_over_qann_ratio": e2 / e3,
        "proxy_underreports_x": e2 / e1,
        "snn_wins": bool(e2 < e3),
    }


def main():
    t_start = time.perf_counter()
    print(
        f"backend={jax.default_backend()} SMOKE={SMOKE}  "
        f"arch: C={C_IN} H={H_HID} N={N_OUT} (W={W_TOTAL} synapses, "
        f"M={M_STATE} state neurons)  SNN weights={SNN_WEIGHT_BITS}b",
        flush=True,
    )

    # --- 1. Analytic crossover per T (the headline verification). ----------- #
    t_verify = list(range(5, 11))  # paper's T in [5,10]
    crossovers = {}
    print("\nAnalytic crossover (SNN == QANN):")
    print(f"  {'T':>3}  {'QANN bits':>9}  {'crossover s*':>12}")
    for T in t_verify:
        s_star = crossover_spike_rate(T)
        crossovers[T] = s_star
        print(f"  {T:>3}  {qann_bits(T):>9}  {s_star * 100:>11.2f}%", flush=True)
    mean_cross = float(np.mean(list(crossovers.values())))
    lit_threshold = 0.064
    matches = abs(mean_cross - lit_threshold) <= 0.02  # within 2 percentage points
    print(
        f"  mean over T in [5,10] = {mean_cross * 100:.2f}%  "
        f"(paper ~6.4%; match={matches})",
        flush=True,
    )

    # --- 2. Three-number energy sweep over spike rate x T. ------------------ #
    rate_grid = [0.01, 0.02, 0.05, 0.064, 0.10, 0.20]
    sweep = []
    print("\nThree-number energy sweep (pJ per sample):")
    print(
        f"  {'T':>3} {'s':>6}  {'SOP proxy':>11}  {'SNN(mem)':>11}  "
        f"{'QANN':>11}  {'SNN/QANN':>8}  {'proxy_under':>11}  win"
    )
    for T in (5, 7, 10):
        for s in rate_grid:
            row = _energies_at(s, T)
            sweep.append(row)
            print(
                f"  {T:>3} {s:>6.3f}  {row['e_sop_proxy_pJ']:>11.3e}  "
                f"{row['e_snn_memory_inclusive_pJ']:>11.3e}  {row['e_qann_pJ']:>11.3e}  "
                f"{row['snn_over_qann_ratio']:>8.2f}  {row['proxy_underreports_x']:>10.0f}x"
                f"  {'Y' if row['snn_wins'] else 'n'}",
                flush=True,
            )

    # --- 3. Ground the curve in a trained classifier's measured spike rates. - #
    if SMOKE:
        epochs, n, batch, seeds = 3, 64, 32, [0]
    else:
        epochs, n, batch, seeds = 12, 256, 32, [0, 1]

    # (label, input_p, reg_strength, reg_target) -- span low -> high activity.
    # "regularised" takes the high-drive input but adds a strong spyx.fn
    # sparsity_reg (target 2% of the hidden layer) to pull the rate back down,
    # demonstrating the activity regulariser as the spike-rate knob.
    drive_settings = [
        ("sparse_input", 0.05, 0.0, 0.0),   # sparse enough to sit below crossover
        ("regularised", 0.70, 20.0, 0.02),
        ("low_drive", 0.12, 0.0, 0.0),
        ("mid_drive", 0.30, 0.0, 0.0),
        ("high_drive", 0.70, 0.0, 0.0),
    ]
    T_operating = 8  # inside [5,10]; qann_bits(8) = 4
    operating_points = []
    print(f"\nTrained-classifier operating points (T={T_operating}):")
    print(
        f"  {'setting':>12}  {'acc':>5}  {'in_rate':>7}  {'hid_rate':>8}  "
        f"{'mean_sr':>7}  {'SNN/QANN':>8}  win"
    )
    for label, input_p, reg_strength, reg_target in drive_settings:
        accs, in_rs, hid_rs = [], [], []
        for seed in seeds:
            a, ir, hr = train_and_measure(
                seed, T_operating, input_p, reg_strength, reg_target,
                epochs, n, batch,
            )
            accs.append(a)
            in_rs.append(ir)
            hid_rs.append(hr)
        acc = float(np.mean(accs))
        in_rate = float(np.mean(in_rs))
        hid_rate = float(np.mean(hid_rs))
        mean_sr = 0.5 * (in_rate + hid_rate)  # network average spike rate
        # Faithful per-layer energy: layer-1 charged at the input rate, layer-2
        # at the measured hidden rate (they differ a lot in a trained net).
        row = _energies_at(in_rate, T_operating, s_hid=hid_rate)
        row.update(
            {
                "setting": label,
                "accuracy": acc,
                "input_rate": in_rate,
                "hidden_rate": hid_rate,
                "mean_spike_rate": mean_sr,
            }
        )
        operating_points.append(row)
        print(
            f"  {label:>12}  {acc * 100:>4.0f}%  {in_rate * 100:>6.1f}%  "
            f"{hid_rate * 100:>7.1f}%  {mean_sr * 100:>6.1f}%  "
            f"{row['snn_over_qann_ratio']:>8.2f}  {'Y' if row['snn_wins'] else 'n'}",
            flush=True,
        )

    # --- 4. Optional plot. --------------------------------------------------- #
    plot_path = None
    if not SMOKE:
        try:
            plot_path = _make_plot(crossovers)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"  (plot skipped: {exc})", flush=True)

    # --- 5. Results JSON. ---------------------------------------------------- #
    results = {
        "config": {
            "smoke": SMOKE,
            "architecture": {"C_in": C_IN, "H_hidden": H_HID, "N_out": N_OUT},
            "synapses": W_TOTAL,
            "state_neurons": M_STATE,
            "snn_weight_bits": SNN_WEIGHT_BITS,
            "snn_state_bits": SNN_STATE_BITS,
            "T_operating": T_operating,
            "seeds": seeds,
            "epochs": epochs,
        },
        "constants_pJ": {
            "compute_horowitz_2014_45nm_fp32": {
                "E_ADD32": E_ADD32_PJ,
                "E_MULT32": E_MULT32_PJ,
                "E_MAC32": E_MAC32_PJ,
                "note": "Horowitz, Computing's Energy Problem, ISSCC 2014, 45nm; "
                "E_AC=E_ADD32 scaled ~linearly, E_MAC mult ~quadratic in bits.",
            },
            "snn_yan_2024_22nm": {
                "E_ACC": E_ACC_PJ,
                "E_CMP": E_CMP_PJ,
                "E_SUB": E_SUB_PJ,
                "E_MEM_weight_DRAM_per_bit": E_MEM_WEIGHT_PJ_PER_BIT,
                "E_MEM_onchip_dense_per_bit": E_MEM_ONCHIP_PJ_PER_BIT,
                "E_MEM_sparse_route_per_bit": E_MEM_SPARSE_PJ_PER_BIT,
                "note": "Yan, Bai & Wong, arXiv:2409.08290, 22nm. Weight DRAM = "
                "1300 pJ / 64-bit word ~= 20 pJ/bit; dense movement 0.25 pJ/bit.",
            },
        },
        "citations": [
            "Horowitz, M. 'Computing's Energy Problem (and what we can do about "
            "it).' ISSCC 2014.",
            "Yan, Z., Bai, Z., Wong, W.-F. 'Reconsidering the energy efficiency "
            "of spiking neural networks.' arXiv:2409.08290, 2024.",
        ],
        "honesty": (
            "All pJ constants are literature values (Horowitz 2014, 45nm; Yan et "
            "al. 2024, 22nm). This is an ANALYTIC energy model, not a silicon "
            "measurement. The crossover is exact given the model; its value "
            "depends on the memory assumptions and on the SNN(8b)-vs-QANN"
            "(ceil(log2(T+1))b) weight-precision asymmetry."
        ),
        "crossover": {
            "per_T": {str(T): crossovers[T] for T in t_verify},
            "mean_spike_rate_T5_to_T10": mean_cross,
            "literature_threshold": lit_threshold,
            "matches_literature": bool(matches),
            "sensitivity_note": (
                "Crossover ~ qann_bits / (SNN_WEIGHT_BITS * T). If SNN weights "
                "were bit-matched to the QANN (both b bits) the crossover rises "
                "to ~1/T (10-20%); if the SNN paid Loihi sparse-routing cost "
                "(3.0 pJ/bit) for weight fetch it falls to ~1-2%. The 8-bit-SNN-"
                "weight assumption is what reproduces the paper's ~6.4%."
            ),
        },
        "energy_sweep": sweep,
        "trained_operating_points": operating_points,
        "wall_seconds": time.perf_counter() - t_start,
    }
    if plot_path is not None:
        results["plot"] = plot_path.name

    out_path = HERE / "honest_energy_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}  ({results['wall_seconds']:.1f}s)", flush=True)
    return results


def _make_plot(crossovers):  # pragma: no cover - best-effort figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_grid = np.linspace(0.001, 0.25, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {5: "#1f77b4", 7: "#ff7f0e", 10: "#2ca02c"}
    for T in (5, 7, 10):
        snn = [energy_snn_memory_inclusive(s, T) for s in s_grid]
        qann = energy_qann(T)
        ax.plot(s_grid * 100, snn, color=colors[T], label=f"SNN (mem), T={T}")
        ax.axhline(qann, color=colors[T], ls="--", lw=1, alpha=0.7)
        s_star = crossovers.get(T, crossover_spike_rate(T))
        ax.plot([s_star * 100], [qann], "o", color=colors[T], ms=7)
    ax.axvline(6.4, color="k", ls=":", lw=1, label="paper ~6.4%")
    ax.set_xlabel("average spike rate (%)")
    ax.set_ylabel("energy per sample (pJ)")
    ax.set_yscale("log")
    ax.set_title(
        "Memory-inclusive SNN vs bit-matched QANN (dashed)\n"
        "markers = crossover; SNN wins only to their left"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = HERE / "honest_energy_crossover.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
