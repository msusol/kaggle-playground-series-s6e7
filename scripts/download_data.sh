#!/usr/bin/env zsh
# Download competition data. Requires ~/.kaggle/kaggle.json and accepted rules.
#
# Rule acceptance is a hard prerequisite: Kaggle refuses to serve any competition
# file until you've clicked "I Understand and Accept" on the competition's rules
# page. This script halts with clear instructions instead of a raw API error if
# that hasn't happened yet.
set -euo pipefail

COMP="playground-series-s6e7"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"

STDERR_LOG="$(mktemp)"
if ! kaggle competitions download -c "$COMP" -p "$DEST" 2> "$STDERR_LOG"; then
  # kaggle CLI 2.x returns a bare "403 Client Error: Forbidden" for this endpoint with
  # no mention of "rules" — older 1.x CLIs spelled out "You must accept this
  # competition's rules...". Match both since either can show up depending on version.
  if grep -qiE "rules|403|forbidden" "$STDERR_LOG"; then
    echo ""
    echo "HALTED: competition rules not yet accepted for $COMP (or you have not"
    echo "joined the competition)."
    echo "  1. Visit https://www.kaggle.com/competitions/$COMP/rules"
    echo "  2. Click \"I Understand and Accept\""
    echo "  3. Re-run: zsh scripts/download_data.sh"
    rm -f "$STDERR_LOG"
    exit 1
  fi
  cat "$STDERR_LOG" >&2
  rm -f "$STDERR_LOG"
  exit 1
fi
rm -f "$STDERR_LOG"

unzip -o "$DEST/${COMP}.zip" -d "$DEST"
rm -f "$DEST/${COMP}.zip"

echo "Downloaded to $DEST:"
ls -lh "$DEST"
