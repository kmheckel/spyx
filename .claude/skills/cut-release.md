---
name: cut-release
description: Cut a new Spyx release — bump the version, tag, and publish to PyPI. Use when the user asks to "cut a release", "bump the version", "publish to PyPI", "tag a new version", or "ship v0.x".
---

# Cut a Spyx release

The version lives in `pyproject.toml` (`project.version`). Tags are `v<version>`.
`scripts/release.py` automates build + tag + GitHub release + PyPI publish; the
`Upload Python Package` workflow also publishes automatically when a GitHub
release is **published**.

## 1. Pre-flight — everything green

Never tag a red tree. Run the full CI-equivalent locally first:

```bash
uv sync --extra quant
uv run ruff check && uv run ruff format --check
uv run pytest -m "not network"
uv run mkdocs build --strict
uv run python scripts/smoke_notebook_apis.py
```

Also sanity-check the install end to end:

```bash
uv run python scripts/check_install.py
```

## 2. Bump the version

Edit `project.version` in `pyproject.toml` (semver). Update `uv.lock` if the
bump changes it:

```bash
uv lock
```

Commit the bump on its own:

```bash
git add pyproject.toml uv.lock
git commit -m "Release vX.Y.Z"
```

## 3. Dry-run the release script

Always dry-run first — it prints every command without executing:

```bash
uv run python scripts/release.py --dry-run --github --pypi
```

Confirm the tag (`vX.Y.Z`) and commands look right.

## 4. Release

Two paths — pick based on how the repo publishes:

**A. Let CI publish (preferred).** Push the commit, create a GitHub release for
the tag; the `Upload Python Package` workflow builds and pushes to PyPI on
`release: published` (uses the `PYPI_API_TOKEN` secret).

```bash
git push origin <branch>
# then create the release + tag (via GitHub UI or the release script's --github)
uv run python scripts/release.py --github
```

**B. Publish directly from the release script:**

```bash
uv run python scripts/release.py --github --pypi
```

Note: the script shells out to `gh` for the GitHub release and `uv publish` for
PyPI. In an environment without the `gh` CLI, do the GitHub release through the
web UI (or the GitHub tools) and let the publish workflow handle PyPI.

## 5. Verify

- The tag exists on the remote and the GitHub release is published.
- The PyPI page shows the new version, and `uv pip install spyx==X.Y.Z` (in a
  scratch venv) resolves.
- The `Upload Python Package` Actions run is green.

## Don'ts

- Don't tag before tests/docs are green.
- Don't reuse or move an existing tag — cut a new patch version instead.
- Don't hand-edit `_version.py` if the project derives it; the source of truth
  is `pyproject.toml`'s `project.version`.
