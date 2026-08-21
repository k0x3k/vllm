#!/usr/bin/env bash
# Bootstrap Claude session persistence for this checkout.
#
# Claude Code writes transcripts and memory to ~/.claude/projects/<slug>/, which is
# ephemeral and machine-local. This points that directory at a clone of the private
# lab repo, so sessions survive rebuilds and travel between machines.
#
# Run once per machine:  ./study/lab-setup.sh
set -euo pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
LAB_REPO=${VLLM_LAB_REPO:-k0x3k/vllm-lab}
LAB_DIR=${VLLM_LAB_DIR:-$(dirname "$REPO_ROOT")/vllm-lab}
# Claude Code's project-directory slug: the absolute path with every non-alphanumeric
# character replaced by a dash.
SLUG=$(printf '%s' "$REPO_ROOT" | sed 's/[^a-zA-Z0-9]/-/g')
SRC="$HOME/.claude/projects/$SLUG"
DST="$LAB_DIR/sessions/$SLUG"

if [ ! -d "$LAB_DIR/.git" ]; then
  if [ -n "${VLLM_LAB_TOKEN:-}" ]; then
    echo "==> cloning $LAB_REPO into $LAB_DIR (using VLLM_LAB_TOKEN)"
    git clone -q "https://x-access-token:${VLLM_LAB_TOKEN}@github.com/${LAB_REPO}.git" "$LAB_DIR"
    git -C "$LAB_DIR" remote set-url origin "https://github.com/$LAB_REPO.git"
  elif gh repo view "$LAB_REPO" >/dev/null 2>&1; then
    echo "==> cloning $LAB_REPO into $LAB_DIR"
    gh repo clone "$LAB_REPO" "$LAB_DIR"
  else
    echo "==> $LAB_REPO does not exist yet; starting a local repo at $LAB_DIR"
    echo "    create it later with: gh repo create $LAB_REPO --private"
    mkdir -p "$LAB_DIR"
    git -C "$LAB_DIR" init -q -b main
    git -C "$LAB_DIR" remote add origin "https://github.com/$LAB_REPO.git"
    printf '%s\n' "# vllm-lab" "" \
      "Private Claude Code session transcripts and memory for the \`vllm\` study fork." "" \
      "\`sessions/<slug>/\` mirrors \`~/.claude/projects/<slug>/\` on each machine;" \
      "the slug is the checkout path with non-alphanumerics replaced by dashes." "" \
      "Set up by \`study/lab-setup.sh\` in the fork; synced by \`study/lab-sync.sh\`." \
      > "$LAB_DIR/README.md"
  fi
fi
mkdir -p "$DST"

if [ -L "$SRC" ]; then
  echo "==> $SRC is already a symlink -> $(readlink "$SRC")"
elif [ -d "$SRC" ]; then
  echo "==> importing existing transcripts from $SRC"
  cp -an "$SRC/." "$DST/"
  rm -rf "$SRC"
  ln -s "$DST" "$SRC"
else
  mkdir -p "$(dirname "$SRC")"
  ln -s "$DST" "$SRC"
fi
echo "==> sessions for $REPO_ROOT now live in $DST"

echo "==> installing SessionEnd sync hook"
SYNC="$REPO_ROOT/study/lab-sync.sh"
python3 - "$HOME/.claude/settings.json" "$SYNC" <<'PY'
import json, os, sys
path, sync = sys.argv[1], sys.argv[2]
cfg = {}
if os.path.exists(path):
    with open(path) as f:
        cfg = json.load(f) or {}
hooks = cfg.setdefault("hooks", {}).setdefault("SessionEnd", [])
if not any(h.get("command") == sync for e in hooks for h in e.get("hooks", [])):
    hooks.append({"hooks": [{"type": "command", "command": sync}]})
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(f"    SessionEnd -> {sync}")
PY

"$SYNC" || true
echo "==> done"
