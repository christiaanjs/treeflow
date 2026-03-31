#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Upgrade setuptools first to fix legacy package build issues (ete3, silence_tensorflow)
pip install --upgrade setuptools --ignore-installed

pip install -e ".[test]"

# Ensure the pip-installed pytest (/usr/local/bin/pytest) takes precedence
# over any uv-tool-installed pytest that may be on PATH
echo 'export PATH="/usr/local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
