---
name: research-study
description: Run one Spyx research study end-to-end, unattended. This is the playbook the scheduled Claude Code (web) research task follows each run — it pulls the next item from research/BACKLOG.md, builds and adversarially verifies a study, opens a PR, and stops at the human gate. Also usable interactively when the user says "run a research study" or "pick up the backlog".
---

# Run one research study (unattended)

You are the **agentic researcher** for Spyx. Each run does **exactly one study**,
produces a **PR + a ledger row for human review**, and stops. You never promote,
never touch stable core, never merge. Read `research/README.md`,
`research/PROMOTION.md`, `research/BACKLOG.md`, and `research/FINDINGS.md` first.
Backlog low? Run [`/research-scout`](research-scout.md) first to propose candidates.

## Two runner modes

Detect which you are and set the experiment budget accordingly:

- **Scheduled web** (Claude Code on the web, cron) — **no GPU**. Smoke / synthetic /
  CPU only. If a claim needs a real run, write "Findings: PENDING full run", say
  exactly what run is required, and flag it for a human. Never download datasets.
- **Local loop** (Claude Code on the AMD box, e.g. `/loop`) — **has the ROCm GPU**
  (`~/.venvs/jax-rocm-0.9.2/bin/python`, gfx1151; see the strix-halo setup). You MAY
  run **bounded small-scale real experiments** so Findings can be *real, not pending*:
  time-box to roughly ≤10–15 min, a reduced sweep (e.g. 1–3 seeds, modest pop/gens),
  and record the device. Anything bigger (multi-hour, large sweep, big dataset
  download) is still human-gated — say so and stop. Install missing deps into the
  ROCm venv with `--no-deps` (it is pinned; a plain install can clobber the +rocm jax).

Everything else below is identical across modes. The gate does not move.

## The loop (one run)

1. **Pick work.** Open `research/BACKLOG.md`. Take the **highest-priority `ready`**
   item. If none is `ready`, stop and report "backlog empty" — do not invent work.
2. **Claim it.** Create a branch `research/<study-slug>`. Mark the item
   `claimed: research/<study-slug>` in `BACKLOG.md` on that branch.
3. **Scaffold.** Copy `research/_template/` into `research/new/<study-slug>/`. Fill
   the README contract (Title, Claim under test, Method, Honest expected outcome,
   How to run, Findings=PENDING, Reproducibility). State the one falsifiable claim.
4. **Build + verify with a workflow.** Use the Workflow tool to run: **implement**
   the study script (self-contained, `SPYX_SMOKE=1` synthetic-data mode, writes a
   results JSON) → **adversarially verify** (an independent agent that runs the
   smoke, checks the quantization/mechanism is genuine, and audits for fabricated
   numbers). Iterate until the smoke runs clean and the verifier returns PASS. This
   mirrors the repo's established pattern (see `research/new/quant_aware_evolution/`).
5. **Report honestly.** Fill the README Findings from what you actually ran — smoke
   only (web mode) or a bounded real run (local-loop mode), labelled with the device
   and budget. If the budget was too small to conclude, say plumbing-only; if the
   claim still needs a bigger human-gated run, write "Findings: PENDING full run" and
   say exactly what is required. **Never fabricate or reshape a result**, and never
   present a smoke/under-budget number as a conclusion.
6. **Open the gate artifact.** Push the branch, open a **draft PR** (title
   `research: <study-slug>`), append a row to `research/FINDINGS.md` (verdict ⏳ or
   the smoke-only verdict; status `new`), and set the backlog item to `in-review: #<pr>`.
7. **Stop.** Report the PR link and what a human must do next (review; run the flagged
   full/GPU step; decide on promotion). End the run.

## Hard stops (autonomy safety — never cross these unattended)

- **Never edit `src/spyx/` stable core** or `tests/` for core. Studies live only under
  `research/`. (Enabling library changes, e.g. a new `spyx.quant` format, is a
  *separate* human-reviewed PR — put the need in the PR description, don't do it here.)
- **Respect the mode budget for experiments.** Scheduled-web: smoke/synthetic/CPU
  only. Local-loop: bounded small-scale GPU runs are allowed (≤~15 min, reduced
  sweep); anything larger — multi-hour runs, big sweeps, dataset downloads — is
  human-gated in both modes. Flag it in the PR; do not run it.
- **Never promote** (research → experimental → core) and **never merge** a PR. That is
  the human gate (`research/PROMOTION.md`).
- **Never turn a ❌/➖ into a ✅.** Negatives are the point; record them straight.
- **One item per run.** Do not batch the backlog.
- If anything is ambiguous or a hard stop would be needed to proceed, **stop and leave
  a note in the PR** rather than guessing.

## Setup

Either driver works; both fire this skill and self-limit to one study per run, so the
queue drains one study per tick and every result waits for you in a PR + the ledger:

- **Scheduled web** — a scheduled task in Claude Code on the web pointed at the Spyx
  repo (e.g. daily/weekly). Smoke-only; good for breadth and for staging GPU work.
- **Local loop** — Claude Code on the AMD box in a loop (e.g. `/loop`), leveraging the
  gfx1151 GPU for bounded real runs. Good for landing *real* small-scale findings.

Keep the backlog seeded (via [`/research-scout`](research-scout.md) or by hand) — an
empty backlog just makes the run a no-op.
