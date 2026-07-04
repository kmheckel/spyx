#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs Spyx and its dev + optional dependencies so `uv run pytest`,
# `uv run ruff check`, and `uv run mkdocs build` work the moment a session
# starts — no manual `uv sync` first.
#
# Runs synchronously (no `{"async": true}` line) so the environment is ready
# before the agent's first turn. It's idempotent: uv resolves from uv.lock and
# no-ops when everything is already present, so re-running on resume is cheap.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment. Local users
# manage their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Ensure uv is on PATH (installed to ~/.local/bin by the standard installer).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# `--extra quant` pulls qwix (from GitHub) so the quantization + SSM-quant
# tests run instead of skipping. Dev tooling (pytest, ruff, mkdocs) comes from
# the default dependency-group. The [loaders] extra is intentionally left out:
# tonic's tests download datasets and are gated behind the `network` marker.
uv sync --extra quant

# Persist uv's bin dir for the rest of the session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
fi

echo "Spyx environment ready: uv run pytest -m 'not network' | uv run ruff check"
