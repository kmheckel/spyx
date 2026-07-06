---
name: research-study
description: Run one Spyx research study end-to-end, unattended. This is the playbook the scheduled Claude Code (web) research task follows each run — it pulls the next item from research/BACKLOG.md, builds and adversarially verifies a study, opens a PR, and stops at the human gate. Also usable interactively when the user says "run a research study" or "pick up the backlog".
---

# Run one research study (unattended)

You are the **scheduled agentic researcher** for Spyx. Each run does **exactly one
study**, produces a **PR + a ledger row for human review**, and stops. You never
promote, never touch stable core, never merge. Read `research/README.md`,
`research/PROMOTION.md`, and `research/FINDINGS.md` first.

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
5. **Report honestly.** Fill the README Findings **only from what the smoke shows**,
   and label it plumbing-only if the budget is too small to conclude. If the claim
   needs a full/GPU/dataset run to settle, write "Findings: PENDING full run" and say
   exactly what run is required. **Never fabricate or reshape a result.**
6. **Open the gate artifact.** Push the branch, open a **draft PR** (title
   `research: <study-slug>`), append a row to `research/FINDINGS.md` (verdict ⏳ or
   the smoke-only verdict; status `new`), and set the backlog item to `in-review: #<pr>`.
7. **Stop.** Report the PR link and what a human must do next (review; run the flagged
   full/GPU step; decide on promotion). End the run.

## Hard stops (autonomy safety — never cross these unattended)

- **Never edit `src/spyx/` stable core** or `tests/` for core. Studies live only under
  `research/`. (Enabling library changes, e.g. a new `spyx.quant` format, is a
  *separate* human-reviewed PR — put the need in the PR description, don't do it here.)
- **Never run GPU, full-budget, or dataset-downloading steps.** Smoke/synthetic/CPU
  only. If the claim needs one, flag it for a human in the PR — do not run it.
- **Never promote** (research → experimental → core) and **never merge** a PR. That is
  the human gate (`research/PROMOTION.md`).
- **Never turn a ❌/➖ into a ✅.** Negatives are the point; record them straight.
- **One item per run.** Do not batch the backlog.
- If anything is ambiguous or a hard stop would be needed to proceed, **stop and leave
  a note in the PR** rather than guessing.

## Setup (the schedule itself)

Run this as a **scheduled task in Claude Code on the web** pointed at the Spyx repo,
firing this skill (e.g. daily/weekly). Each firing is one loop above. The cadence is
yours; the runner self-limits to one study per firing, so the queue drains at one
study per tick and every result waits for you in a PR + the ledger. Keep the backlog
seeded — an empty backlog just makes the run a no-op.
