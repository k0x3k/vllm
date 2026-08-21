#!/usr/bin/env bash
# Commit and push whatever Claude has written into the private lab repo.
# Runs automatically on SessionEnd (installed by lab-setup.sh); safe to run by hand.
#
# A Codespaces token is scoped to the repo that owns the Codespace, so it cannot
# push to the lab repo. Set VLLM_LAB_TOKEN (a fine-grained PAT with Contents:
# read+write on the lab repo) as a Codespaces secret and it is used for the push.
set -uo pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || exit 0
LAB_REPO=${VLLM_LAB_REPO:-k0x3k/vllm-lab}
LAB_DIR=${VLLM_LAB_DIR:-$(dirname "$REPO_ROOT")/vllm-lab}
[ -d "$LAB_DIR/.git" ] || exit 0
cd "$LAB_DIR" || exit 0

git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "sessions: $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ)" || exit 0
fi
git -c core.logAllRefUpdates=true rev-parse @{u} >/dev/null 2>&1 || true

if [ -n "${VLLM_LAB_TOKEN:-}" ]; then
  REMOTE="https://x-access-token:${VLLM_LAB_TOKEN}@github.com/${LAB_REPO}.git"
else
  REMOTE=origin
fi

timeout 60 git pull --rebase -q "$REMOTE" main >/dev/null 2>&1
if ! timeout 60 git push -q "$REMOTE" HEAD:main >/dev/null 2>&1; then
  if [ -z "${VLLM_LAB_TOKEN:-}" ]; then
    echo "lab-sync: push failed. Commits are local only." >&2
    echo "lab-sync: set VLLM_LAB_TOKEN (PAT with Contents:write on $LAB_REPO)." >&2
  else
    echo "lab-sync: push failed with VLLM_LAB_TOKEN set; check the token's scope." >&2
  fi
fi
