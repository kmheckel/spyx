# Reproductions

Faithful re-implementations of a published result, checked in Spyx.

## What belongs here

- A study that re-implements a paper's headline claim as closely as practical and
  reports whether it holds in Spyx.
- Cross-framework comparisons of the same task (Spyx vs Torch / Norse /
  SpikingJelly / mlGeNN) count as reproductions — they reproduce a reference
  number and measure Spyx against it.
- One study = one folder = one claim reproduced (or refuted).

## What does NOT belong here

- Changing the method (new dataset, new neuron, ablations) — that is an
  [extension](../extensions/README.md).
- Ideas with no published reference — that is [new research](../new/README.md).

## Existing work of this kind

- [`../paper/`](../paper/) — the Spyx paper's SHD & N-MNIST benchmarks
  reproduced across five frameworks.
- [`../SPSN/`](../SPSN/) — Stochastic Parallelizable Spiking Neurons, prior art
  for Spyx's parallel neurons.

## How to add one

Copy [`../_template/README.md`](../_template/README.md) into a new folder here,
fill in every section, and record seeds + hardware + commit so the run can be
reproduced.
