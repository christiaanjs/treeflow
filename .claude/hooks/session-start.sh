#!/bin/bash

# Only run in remote Claude Code web sessions
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Idempotent PATH fix: only append if not already present
_fix_path() {
  if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    grep -qF '/usr/local/bin:$PATH' "$CLAUDE_ENV_FILE" 2>/dev/null || \
      echo 'export PATH="/usr/local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
  fi
}

# Fast path: if already installed (cached container), fix PATH and exit synchronously.
# This avoids any race condition on subsequent sessions.
if python -c "import treeflow, pytest, pandas, allpairspy" 2>/dev/null; then
  _fix_path
  exit 0
fi

# Slow path (fresh container): go async so the session starts immediately
# while install runs in background. The race condition here is acceptable —
# CLAUDE.md documents manual setup as a fallback.
echo '{"async": true, "asyncTimeout": 300000}'

LOG="$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.log"

{
  echo "[$(date -u)] Session start hook starting (fresh install)"

  pip install --upgrade setuptools --ignore-installed -q \
    && echo "[$(date -u)] setuptools upgraded" \
    || echo "[$(date -u)] WARNING: setuptools upgrade failed"

  pip install -e "$CLAUDE_PROJECT_DIR[test]" -q \
    && echo "[$(date -u)] treeflow installed" \
    || echo "[$(date -u)] WARNING: pip install failed"

  _fix_path

  echo "[$(date -u)] Session start hook complete"
} >> "$LOG" 2>&1
