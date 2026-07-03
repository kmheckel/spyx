# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repository.
The full project overview lives in [AGENTS.md](AGENTS.md) — read it for the
module map, coding standards, and architecture. This file is the quick
operational reference.

## Environment

Spyx uses [uv](https://github.com/astral-sh/uv). In Claude Code on the web the
`SessionStart` hook (`.claude/hooks/session-start.sh`) runs `uv sync --extra
quant` for you, so the commands below work immediately. Locally, run
`uv sync --extra quant` once first.

## Commands

```bash
uv run pytest -m "not network"     # full suite minus dataset-downloading tests
uv run pytest tests/test_nn_nnx.py # a single test file
uv run ruff check                  # lint
uv run ruff format                 # auto-format
uv run mkdocs build --strict       # build docs, fail on broken references
uv run python scripts/smoke_notebook_apis.py   # catch tutorial API drift
uv run python scripts/check_install.py         # end-to-end install smoke test
```

CI (`.github/workflows/ci.yml`) runs `lint`, `test` (py3.11 + py3.12), `docs`,
and `smoke` on every PR. Match it locally before pushing.

## Conventions

- **Every stateful piece is a `flax.nnx.Module`** with plain `nnx.Param` nodes.
  New neurons/layers follow the `(x, state) -> (out, new_state)` contract so
  they drop into `spyx.nn.Sequential` and `spyx.nn.run`.
- **Surrogate gradients are `jax.custom_gradient` factories** in `spyx.axn`.
- **flax 0.11+ API**: `nnx.Optimizer(model, tx, wrt=nnx.Param)` and
  `optimizer.update(model, grads)`. Older tutorials that skip `wrt=` are wrong.
- **Tests that download data** carry `@pytest.mark.network` and are excluded
  from CI. Keep new network-dependent tests behind that marker.
- **Ruff** is the linter/formatter (line length 88). `src/` and `tests/` are
  linted; `research/`, `docs/`, `scripts/` are excluded.

## Repository layout

- `src/spyx/` — library (`nn`, `axn`, `fn`, `data`, `optimize`, `nir`, `quant`,
  `ssm`, `phasor`, `experimental`).
- `tests/` — pytest suite (`conftest.py` pins JAX to CPU + seeds fixtures).
- `docs/` — MkDocs, organized by the [Diátaxis](https://diataxis.fr) framework
  (tutorials / how-to / reference / explanation).
- `scripts/` — smoke tests, demos, release automation.
- `.claude/skills/` — task playbooks for agents (see the skill files there).
