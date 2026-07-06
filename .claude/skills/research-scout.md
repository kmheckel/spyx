---
name: research-scout
description: Survey current literature and propose Spyx research backlog items, classified as replication / extension / novelty. Use when the user says "scout papers", "what should we research next", "find replication targets", or when the backlog is running low. Feeds research/BACKLOG.md candidates for human triage.
---

# Scout the literature → backlog candidates

Turn the current state of the field into **falsifiable, Spyx-shaped backlog items**.
You research against real papers and classify each opportunity onto the repo's
taxonomy (see `research/README.md`): **replication**, **extension**, or **novelty**.
You propose; the human triages. Read `research/BACKLOG.md` and `research/FINDINGS.md`
first so you don't re-propose done or in-flight work.

## Steps

1. **Pick a focus.** Default to the active track in `research/BACKLOG.md`
   (quantization & efficient architectures: low-precision training — NVFP4/MXFP4,
   ternary/BitNet, shift-add; SSMs; matmul-free; spiking efficiency). Or take the
   user's topic.
2. **Search real work** with WebSearch / WebFetch — recent arXiv, top venues, and the
   primary sources behind claims (do not rely on memory for anything post-cutoff or
   for specific numbers). Shortlist ~5–10 papers with links and dates.
3. **Classify each into one falsifiable opportunity:**
   - **Replication** — a headline claim you can reproduce in Spyx and check whether it
     holds (and how it compares to Spyx's existing methods). Best when the claim is a
     specific number/trend on a task Spyx can express.
   - **Extension** — a published method pushed somewhere new: a Spyx neuron/SSM/quant
     format it hasn't been tried with, a scaling sweep, an ablation, a harder task.
   - **Novelty** — a gap the paper exposes but does not fill, that Spyx's primitives
     are unusually suited to (e.g. binary-spike × low-precision weights; gradient-free
     ES on a non-differentiable objective the paper trains with STE).
4. **Verify the gap is real (do NOT skip).** A "gap" is not a gap until it survives an
   adversarial literature check. For each candidate, actively try to *refute* its
   novelty: search for a paper that already did it (recent arXiv, the exact method on
   the exact benchmark). If one exists, either drop the candidate or **narrow it to the
   residual gap** (e.g. "not reported" → "no *controlled* apples-to-apples comparison")
   and cite the prior work as baseline. Two of the program's first flagships were killed
   this way; only survivors earn a candidate slot. Note the nearest prior/boundary work.
5. **Assess fit and cost honestly:** does Spyx already have the pieces? Can it be a
   self-contained synthetic-smoke study, or does it need a dataset / GPU / full budget
   (which is human-gated for the unattended runner)? Prefer small, decisive claims.
6. **Propose candidates.** Append to the **Candidates** section of
   `research/BACKLOG.md`, one item each: a one-line falsifiable claim, the paper
   (title + link + year), the bucket (replication/extension/novelty), the Spyx
   modules involved, and the rough cost / whether a GPU run is needed. Do **not**
   promote candidates to `ready` — that is the human's triage.

## Guardrails

- **Cite real, findable sources** (title + link + venue/year). Never invent a paper,
  a number, or a claim. Mark clearly what the paper *claims* vs what the study would
  *test* — they are not the same.
- Flag when a claim is contested or when reproduction is known to be hard.
- Do not start studies here — scouting only fills the backlog. The
  [`/research-study`](research-study.md) runner picks up from there.
- Keep proposals small and falsifiable; a vague "explore X" is not a backlog item.
