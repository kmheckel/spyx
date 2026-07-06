# Promotion gate

How a finding moves from a `research/` study into the library. The ladder is
deliberately slow: research is cheap and disposable, `spyx.experimental` is a
promise of "tested and usable", and stable core is a promise of "supported API".
**Every rung up is a human decision** — agents prepare, they do not promote.

## The ladder

```
research/new/          any honest study (positive, negative, or null)
      │  ── human gate 1 ──▶
spyx.experimental.*     a useful finding with a minimal, tested API
      │  ── human gate 2 ──▶
stable core             a stable, documented, fully-tested API
```

Honest **negatives and nulls stop at `research/`** — they are recorded in
[FINDINGS.md](FINDINGS.md) and kept forever so the question is not re-litigated.
They are never promoted and never deleted.

## Gate 1 — `research/` → `spyx.experimental`

Promote when a study lands a **useful, reusable** result. Checklist:

- [ ] The study README Findings are filled from a **real run** (not smoke), with
      numbers and seed spread — and the verdict is genuinely ✅ (or a ➖ with a
      clearly useful artifact, e.g. a working primitive).
- [ ] The result was **adversarially verified** (an independent pass that tried to
      refute it — correctness *and* honesty).
- [ ] A **minimal API** can be extracted (one module / a few functions or a class),
      importable from `spyx.experimental`.
- [ ] **Unit tests** added under `tests/` covering the new surface; not network-gated.
- [ ] **No stable-core API change** and no new hard dependency on unmerged work.
- [ ] `uv run pytest -m "not network"`, `ruff`, and `ty` all green.
- [ ] Row in [FINDINGS.md](FINDINGS.md) updated to `experimental` with the landing module.

## Gate 2 — `spyx.experimental` → stable core

Promote when the experimental API has proven itself and you are ready to support it.

- [ ] API reviewed and judged **stable** (naming, signature, the `(x, state) ->
      (out, new_state)` contract for neurons / layers).
- [ ] **Full test coverage** including edge cases, on CPU (and ideally a GPU spot-check).
- [ ] **Reference docs** (`docs/reference/`) and at least one **explanation** or
      how-to page; `mkdocs build --strict` green.
- [ ] Moved out of `spyx.experimental` into the owning core module; the
      **stable-core list** in `CLAUDE.md` and `AGENTS.md` updated.
- [ ] CI green on **all** supported Python versions; `ty` clean.
- [ ] A **deprecation note** for anything it replaces, if applicable.
- [ ] [FINDINGS.md](FINDINGS.md) row flipped to `core`.

## Who does what

- **Agents** (incl. the scheduled runner, see [research-study skill](../.claude/skills/research-study.md))
  produce studies + ledger rows + PRs, and — when you ask — run the checklist and
  prepare the promotion PR. They **stop at every gate**.
- **You** review the PR, decide, and merge. Promotion is your signature, not the
  agent's. Use [`/promote-finding`](../.claude/skills/promote-finding.md) to have an
  agent run the checklist and stage the promotion PR for that decision.
