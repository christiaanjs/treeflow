#!/bin/bash
set -euo pipefail

# Only run in remote Claude Code web sessions
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Fast path: if already installed (cached container), fix PATH and exit synchronously.
# This avoids any race condition on subsequent sessions.
if python -c "import treeflow, pytest, pandas, allpairspy" 2>/dev/null; then
  if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo 'export PATH="/usr/local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
  fi
  exit 0
fi

# Slow path (fresh container): go async so the session starts immediately
# while install runs in background. The race condition here is acceptable —
# CLAUDE.md documents manual setup as a fallback.
echo '{"async": true, "asyncTimeout": 300000}'

LOG="$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.log"

{
  echo "[$(date -u)] Session start hook starting (fresh install)"

  pip install --upgrade setuptools --ignore-installed -q
  pip install -e "$CLAUDE_PROJECT_DIR/[test]" -q

  if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo 'export PATH="/usr/local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
  fi

  echo "[$(date -u)] Session start hook complete"
} >> "$LOG" 2>&1
