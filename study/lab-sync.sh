#!/usr/bin/env bash
# Commit and push whatever Claude has written into the private lab repo.
# Runs automatically on SessionEnd (installed by lab-setup.sh); safe to run by hand.
set -uo pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || exit 0
LAB_DIR=${VLLM_LAB_DIR:-$(dirname "$REPO_ROOT")/vllm-lab}
[ -d "$LAB_DIR/.git" ] || exit 0
cd "$LAB_DIR" || exit 0

git add -A
git diff --cached --quiet && exit 0
git commit -q -m "sessions: $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ)" || exit 0
timeout 60 git pull --rebase -q origin HEAD >/dev/null 2>&1
timeout 60 git push -q origin HEAD >/dev/null 2>&1 || echo "lab-sync: push failed, commit is local" >&2
