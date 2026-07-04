<!--
Thanks for contributing to Spyx! Fill in the sections below. Delete any that
don't apply. See AGENTS.md / CLAUDE.md for conventions and CI expectations.
-->

## What's in this PR

<!-- One-paragraph summary of the change and why it's needed. Link issues with
"Closes #NN". -->

## Type of change

- [ ] Bug fix
- [ ] New feature (neuron model, layer, loss, loader, etc.)
- [ ] Documentation
- [ ] Tooling / CI
- [ ] Refactor (no behavior change)

## Changes

<!-- Bullet list of the concrete changes: files, modules, new public API. -->

## Verification

<!-- Paste the commands you ran and their outcome. CI runs all of these. -->

```
uv run ruff check                    #
uv run pytest -m "not network"       #
uv run mkdocs build --strict         #   (if docs changed)
uv run python scripts/smoke_notebook_apis.py   #   (if a tutorial's API changed)
```

## Checklist

- [ ] New public API has docstrings (mkdocstrings renders them in the reference).
- [ ] Added / updated tests; network-dependent tests carry `@pytest.mark.network`.
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] Docs updated if user-facing behavior changed.
