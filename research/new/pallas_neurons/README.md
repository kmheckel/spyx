# Pallas neurons — profiling first (issue #24)

## Question

[Issue #24](https://github.com/kmheckel/spyx/issues/24) asks whether a **Pallas
kernel** should implement the neuron dynamics for speed. A fused kernel is only
worth writing if the **neuron update is the bottleneck** — but the neuron step is
elementwise **O(H)** while the `Linear` feeding it is **O(H²)**, so which one
dominates wall-clock is regime-dependent (hidden width `H`, timesteps `T`,
device). This study measures that *before* anyone writes a kernel.

## Method

`profile_neurons.py` runs a component ablation through [`spyx.bench`](../../../src/spyx/bench.py)
(median latency, XLA-cost-model FLOPs, peak mem) at matched `(H, T, B)`:

| Component | What it isolates | Driver |
| --- | --- | --- |
| `linear` | `Sequential(Linear(H,H))` — matmul per timestep | `spyx.nn.run` (scan) |
| `lif_scan` | `LIF((H,))` — the sequential recurrence | `spyx.nn.run` (scan) |
| `psu_parallel` | `PSU_LIF((H,))` — the associative-scan neuron | its `.parallel` |
| `linear_lif` | `Sequential(Linear(H,H), LIF((H,)))` — a real layer | `spyx.nn.run` |

Read-outs per regime:
- **`lif_scan / linear`** — > 1 means the neuron scan dominates → a fused
  Pallas / parallel kernel can help; < 1 means the matmul dominates → a faster
  neuron barely moves wall-clock (optimise the `Linear` instead).
- **`lif_scan / psu_parallel`** — what the *portable* associative-scan already
  buys for linearizable neurons, i.e. the win you can get **without** Pallas.

## How to run

```bash
uv run python research/new/pallas_neurons/profile_neurons.py            # default sweep
HIDDENS=128,512 SEQLENS=256,1024 BATCH=128 uv run python research/new/pallas_neurons/profile_neurons.py
```

Writes `profile_results.json` (records `backend`/`device`). **Run it on the target
device** — the crossover shifts with hardware: a GPU's higher matmul throughput
pushes the neuron-bound regime to *larger* `H` than CPU does.

## Findings

Sweep on **CPU** (`backend=cpu`, B=64, 20 iters, 83 s; `profile_results.json`).
`neuron/matmul` = `lif_scan` latency ÷ `linear` latency — **< 1 means the matmul
dominates.**

| H | T | neuron/matmul (fwd) | neuron/matmul (fwd+bwd) | assoc-scan ×vs-scan |
|---|---|---|---|---|
| 64 | 64 | 0.35 | 0.67 | 0.62 |
| 64 | 256 | 0.29 | **1.06** | 2.12 |
| 64 | 1024 | 0.38 | 0.97 | 1.25 |
| 256 | 64 | 0.20 | 0.27 | 1.23 |
| 256 | 256 | 0.21 | 0.46 | 1.13 |
| 256 | 1024 | 0.24 | 0.53 | 0.48 |
| 1024 | 64 | 0.20 | 0.44 | 0.89 |
| 1024 | 256 | 0.32 | 0.32 | 0.52 |
| 1024 | 1024 | 0.22 | 0.35 | 0.40 |

**1. Matmul-bound in every forward regime on CPU.** The neuron scan is only
**20–38 %** of the `Linear` it follows, and **16–33 %** of a full `Linear+LIF`
layer. A fused Pallas neuron kernel cannot move forward wall-clock here — the
`Linear` (O(H²)) is the time, the neuron (O(H)) is glue.

**2. The one place the neuron approaches parity is *small-width training.*** In the
backward pass (BPTT — what actually costs during training) the ratio climbs, and
at **H=64 it reaches ~0.7–1.06×** the matmul: the surrogate-gradient scan backward
is comparatively expensive when the matmul is small. So the neuron kernel's payoff,
if any, lives in **narrow networks during training** — which, notably, is also the
regime where evolution is competitive (the thesis result). At H≥256 it stays
matmul-bound (0.27–0.53×) even in the backward.

**3. Associative-scan is *not* a free win — and not even always a win.** The
portable `PSU_LIF` parallel path beats the sequential scan only at **small H /
moderate T** (up to 2.1×) and **loses up to 2.5×** at large H·T (0.40× at
H=1024,T=1024), with **no memory saving** (both store the full trajectory:
identical peak MB). On CPU there is no parallel slack to amortise its extra work.
This matches the `PSU_LIF` note that its wins come "by device slack" — the parallel
route is itself device-gated, so it must be validated on the target GPU too.

**4. Everything above is CPU; the decision needs GPU numbers.** A GPU's far higher
matmul throughput *raises the neuron's relative share*, so the crossover into
neuron-bound territory can appear at H/T sizes people actually train — exactly
where CPU says "matmul-bound." Re-run `profile_neurons.py` on the 8060S / gfx1151
ROCm venv before deciding; that JSON, not this one, is the evidence for #24.

## Decision rule

- **Neuron-bound** on the target device at the sizes you train → worth a fused
  kernel. First try extending the **portable** `associative_scan` path
  (`PSU_LIF`) for linearizable neurons; reserve Pallas for the hard-reset / ALIF
  neurons that *can't* be linearized (see the discussion in issue #24).
- **Matmul-bound** everywhere you care about → skip the neuron kernel; the Linear
  (or its precision — see [`spyx.experimental.matfree`](../../../src/spyx/experimental/matfree.py))
  is where the time is.
