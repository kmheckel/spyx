---
name: promote-finding
description: Promote a research finding up the Spyx ladder (research → spyx.experimental → stable core). Use when the user decides to promote a study, says "graduate this", "move X into experimental/core", or "run the promotion checklist". Runs the PROMOTION.md gate and stages the promotion PR — the human still decides.
---

# Promote a finding

The human has decided a `research/` finding earns a home in the library. Your job is
to **run the gate checklist, extract the API cleanly, and stage a PR** — not to
decide. Read `research/PROMOTION.md` (the criteria) and `research/FINDINGS.md` (the
study's row) first, and confirm which rung: `research → experimental` or
`experimental → core`.

## Steps

1. **Confirm the target and the study.** Which study, which rung. If the study's
   Findings are still smoke-only or PENDING, **stop** — it is not promotable yet; say
   what run is missing.
2. **Run the checklist** for the target rung from `research/PROMOTION.md`. Report each
   item pass/fail with evidence. If any fails, stop and list what is missing.
3. **Extract the API** (only if the checklist passes):
   - *To `spyx.experimental`:* create/extend the owning module under
     `src/spyx/experimental/`, lift the minimal reusable surface out of the study
     script, export it from `spyx/experimental/__init__.py`, and add unit tests under
     `tests/` (not network-gated). Keep the study folder as the provenance record.
   - *To stable core:* move the symbol into the owning core module (`spyx.nn` / `ssm`
     / `quant` / …), add reference + explanation/how-to docs, update the **stable-core
     list in `CLAUDE.md` and `AGENTS.md`**, and add a deprecation note if it replaces
     something.
4. **Verify locally:** `uv run pytest -m "not network"`, `uv run ruff check`,
   `uv run ty check`, and (for core) `uv run mkdocs build --strict`. All must pass.
5. **Update the ledger.** Flip the study's Status in `research/FINDINGS.md` to
   `experimental` or `core` and note the landing module.
6. **Stage the PR** (branch + push + open PR) with the checklist results in the body.
   **Do not merge** — the human reviews and merges. Promotion is their signature.

## Guardrails

- Do not promote a ❌ or ➖ finding, or one whose Findings are smoke-only/PENDING.
- Do not change the finding's verdict to justify promotion.
- Experimental keeps the unstable-API contract; only core promotion implies support.
- If extraction would require reshaping the result or weakening a test to pass, stop
  and report — the finding is not ready.
