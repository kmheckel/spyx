# New research

Novel ideas that do not yet have a paper.

## What belongs here

- New neuron models, training methods, or architectures explored in Spyx before
  (or instead of) publication.
- Studies whose "claim under test" is your own hypothesis rather than a
  published result.

Once an idea here is published, its follow-up work moves to
[`reproductions/`](../reproductions/README.md) (someone reproduces it) or
[`extensions/`](../extensions/README.md) (someone extends it).

## What does NOT belong here

- Work that starts from a published method — use
  [reproductions](../reproductions/README.md) or
  [extensions](../extensions/README.md).

## Index

Every study here — with its one-line claim, honest verdict, and promotion status —
is indexed in the **[findings ledger](../FINDINGS.md)** (grouped by theme:
quantization & low precision · parallel spiking neurons · training methods / SSMs /
memory). That ledger is the single source of truth; this folder is just where the
study code lives.

## Related prior art

- [`../SPSN/`](../SPSN/) — Stochastic Parallelizable Spiking Neurons. Read before
  starting parallel-spiking-neuron work.

## How to add one

Copy [`../_template/README.md`](../_template/README.md) into a new folder here.
Set the Paper field to "novel — no paper yet", write your hypothesis in "Claim
under test", and record seeds + hardware + commit.
