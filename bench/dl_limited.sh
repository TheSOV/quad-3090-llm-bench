#!/bin/bash
# Rate-limited, resumable download of a Hugging Face repo into /data/models/<name>.
# Files already complete in the HF cache are hard-linked instead of downloaded again.
# Usage: dl_limited.sh REPO REVISION NAME [RATE]   (RATE for curl --limit-rate, default 20M = 20 MB/s)
set -u
REPO=$1 REV=$2 NAME=$3 RATE=${4:-20M}
DST=/data/models/$NAME
SNAP=/data/huggingface/hub/models--${REPO//\//--}/snapshots/$REV
mkdir -p "$DST"
curl -s "https://huggingface.co/api/models/$REPO/revision/$REV" | python3 -c "
import json,sys
for s in json.load(sys.stdin)['siblings']:
    f = s['rfilename']
    if not f.startswith(('original/', 'metal/')):
        print(f)" > "$DST/.files"
while read -r f; do
    mkdir -p "$DST/$(dirname "$f")"
    if [ -s "$DST/$f" ] && [ ! -e "$DST/$f.part" ]; then
        continue
    fi
    if [ -e "$SNAP/$f" ]; then
        ln -f "$(readlink -f "$SNAP/$f")" "$DST/$f" && echo "$(date +%T) linked $f" && continue
    fi
    echo "$(date +%T) get $f"
    touch "$DST/$f.part"
    until curl -sfL --limit-rate "$RATE" -C - -o "$DST/$f" "https://huggingface.co/$REPO/resolve/$REV/$f"; do
        echo "$(date +%T) retry $f"; sleep 10
    done
    rm -f "$DST/$f.part"
done < "$DST/.files"
echo "$(date +%T) DONE $(du -sh "$DST" | cut -f1)"
