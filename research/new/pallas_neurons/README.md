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

Writes `profile_results_<backend>.json` (e.g. `_gpu` / `_cpu`, recording
`backend`/`device`). **Run it on the target device** — the crossover shifts with
hardware, and (per the findings) the associative-scan win only materialises where
there is GPU parallel slack.

## Findings

Two sweeps, B=64, 20 iters: **GPU** (`profile_results_gpu.json`, Radeon 8060S /
gfx1151, ROCm, 77 s) — the decision-relevant one — and **CPU**
(`profile_results_cpu.json`, 83 s) for contrast. `neuron/matmul` = `lif_scan` ÷
`linear` latency; **< 1 means the matmul dominates.** `assoc ×` = sequential-scan ÷
associative-scan (`PSU_LIF`) latency; **> 1 means the parallel scan is faster.**

### GPU — 8060S (the one that decides #24)

| H | T | neuron/matmul (fwd) | neuron/matmul (fwd+bwd) | assoc × (fwd) |
|---|---|---|---|---|
| 64 | 64 | 0.44 | 0.62 | **8.9×** |
| 64 | 256 | 0.44 | 0.62 | **20.9×** |
| 64 | 1024 | 0.45 | 0.64 | **21.1×** |
| 256 | 64 | 0.30 | 0.68 | 6.4× |
| 256 | 256 | 0.30 | 0.72 | 6.8× |
| 256 | 1024 | 0.31 | 0.72 | 4.8× |
| 1024 | 64 | 0.21 | 0.40 | 2.8× |
| 1024 | 256 | 0.21 | 0.38 | 1.7× |
| 1024 | 1024 | 0.21 | 0.37 | 1.7× |

**1. Matmul-bound on GPU too — in every forward regime.** The sequential neuron
scan is **21–45 %** of the `Linear` it follows (forward). A Pallas kernel that only
speeds up the neuron *forward* cannot move end-to-end wall-clock — the O(H²) matmul
is the time.

**2. In training the neuron grows to a real minority — but stays below the matmul.**
`fwd+bwd` (BPTT, what actually costs) climbs to **0.62–0.72× at H≤256** and
0.37–0.40× at H=1024. So the surrogate-scan backward is a meaningful ~40–70 % of
the matmul for narrow/mid nets, but never the majority. Still matmul-bound.

**3. The decisive result: the neuron speedup you'd reach for is *already here*,
portable, no Pallas.** On GPU the associative-scan `PSU_LIF` beats the sequential
scan by **1.7–21×** (biggest at small H, still 1.7× at H=1024,T=1024), forward and
backward, with **no memory penalty**. This is the "device slack" the CPU sweep
lacked — on CPU the same parallel path was a wash-to-loss (0.4–2.1×); the GPU's
parallelism is exactly what makes it pay. For *linearizable* neurons you get the
2–21× today, without writing a kernel.

### CPU — for contrast

Same shape on the forward (matmul-bound, neuron 20–38 % of the `Linear`), but
associative-scan is a **wash-to-loss** (0.4–2.1×, no parallel slack to amortise its
extra work) — confirming the parallel route is itself device-gated.

## Decision (issue #24)

**Do not prioritise a fused Pallas neuron kernel.** The data closes the question:

- SNN forward is **matmul-bound on both devices** — accelerating the neuron barely
  moves wall-clock; the `Linear` is the cost.
- The neuron speedup that *would* matter on GPU is **already delivered portably** by
  `associative_scan` (`PSU_LIF`): **2–21×**, forward and backward, no memory cost,
  runs on the deployment device.
- A Pallas kernel's only remaining niche is the neurons `associative_scan`
  **can't** linearize — hard-reset LIF, ALIF adaptive-threshold — *and* only if a
  specific model profiles as neuron-bound in training. Given even H=1024,T=1024
  sits at 0.37× in `fwd+bwd`, that bar is high.

**Higher-ROI directions than part (b):** (i) widen the portable win — give more
neuron types a `.parallel` associative-scan path; (ii) attack the actual
bottleneck, the `Linear` — precision ([`spyx.experimental.matfree`](../../../src/spyx/experimental/matfree.py))
or shape. Reserve a Pallas kernel for a *proven* neuron-bound hard-reset/ALIF model,
NVIDIA-first, behind a `lax.scan` fallback (Pallas ROCm support is experimental).

## Decision rule

- **Neuron-bound** on the target device at the sizes you train → worth a fused
  kernel. First try extending the **portable** `associative_scan` path
  (`PSU_LIF`) for linearizable neurons; reserve Pallas for the hard-reset / ALIF
  neurons that *can't* be linearized (see the discussion in issue #24).
- **Matmul-bound** everywhere you care about → skip the neuron kernel; the Linear
  (or its precision — see [`spyx.experimental.matfree`](../../../src/spyx/experimental/matfree.py))
  is where the time is.
