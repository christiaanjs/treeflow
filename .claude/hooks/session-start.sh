#!/bin/bash
set -eo pipefail

LOG="${CLAUDE_PROJECT_DIR:-/tmp}/.claude/hooks/session-start.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Session start hook starting"

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "[$(date)] Not a remote session, skipping"
  exit 0
fi

echo "[$(date)] CLAUDE_PROJECT_DIR=${CLAUDE_PROJECT_DIR:-<unset>}"
echo "[$(date)] CLAUDE_ENV_FILE=${CLAUDE_ENV_FILE:-<unset>}"

cd "${CLAUDE_PROJECT_DIR:-.}"

# Upgrade setuptools first to fix legacy package build issues (ete3, silence_tensorflow)
echo "[$(date)] Upgrading setuptools..."
pip install --upgrade setuptools --ignore-installed

echo "[$(date)] Installing treeflow with test extras..."
pip install -e ".[test]"

# Ensure the pip-installed pytest (/usr/local/bin/pytest) takes precedence
# over any uv-tool-installed pytest that may be on PATH
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PATH="/usr/local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
fi

echo "[$(date)] Session start hook complete"
