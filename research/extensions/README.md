# Extensions

A published method taken further than the original paper.

## What belongs here

- Scaling sweeps (vary channels, sequence length, width, depth).
- Ablations (surrogate function, optimizer, neuron model, regularizer).
- Applying a method to a new dataset or task.
- Swapping a component (e.g. a different neuron model into a known architecture)
  and measuring the effect.

The reference method is published; the twist is yours.

## What does NOT belong here

- A straight re-implementation with no change — that is a
  [reproduction](../reproductions/README.md).
- An idea with no published starting point — that is
  [new research](../new/README.md).

## Existing work of this kind

- [`../scaling_experiments/`](../scaling_experiments/) — SHD surrogate-gradient
  training swept over input channels (72–700) and sequence length T (128–1024).
- [`../misc/`](../misc/) — surrogate-function, optimizer, and neuron-model
  comparisons, plus evolutionary-training variants.

## How to add one

Copy [`../_template/README.md`](../_template/README.md) into a new folder here,
state clearly what you changed relative to the original method, and record seeds +
hardware + commit.
