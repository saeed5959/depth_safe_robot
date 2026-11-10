#!/usr/bin/env bash
# Replace the __REPO_ROOT__ placeholder (absolute paths Nav2 needs, e.g. the behavior-tree XML)
# with the actual location of this repository. Safe to run again after moving the repo.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LIST="$ROOT/scripts/.paths_to_configure"
[ -f "$LIST" ] || { echo "nothing to configure"; exit 0; }
while read -r f; do
  [ -f "$ROOT/$f" ] || continue
  sed -i "s#__REPO_ROOT__#$ROOT#g" "$ROOT/$f"
  echo "configured: $f"
done < "$LIST"
