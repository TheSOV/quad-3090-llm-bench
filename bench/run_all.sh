#!/bin/bash
# Run every configuration in order; skip the ones already completed (results.json "complete": true).
cd "$(dirname "$0")"
QUEUE="llama70b-awq-tp4 llama70b-awq-tp2pp2 gptoss120b-tp4 qwen38-27b-bf16-tp4 qwen38-27b-awq-tp4 qwen38-27b-awq-dp2tp2 qwen36-35b-awq-tp4 qwen36-35b-awq-dp2tp2"
# Optional: TAG and extra bench.py args, e.g.  TAG=rapida ./run_all.sh "" --scenarios medio --conc 1,8,32
TAG=${TAG:-}
LOGS=${LOG_DIR:-../logs}
mkdir -p "$LOGS"
LIST=${1:-$QUEUE}; shift || true
for n in $LIST; do
    R=/data/benchmarks/$n${TAG:+-$TAG}/results.json
    if [ -f "$R" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$R')).get('complete') else 1)"; then
        echo "$(date +%T) skip $n (complete)"; continue
    fi
    echo "$(date +%T) run $n"
    ~/vllm-env/bin/python -I bench.py "$n" ${TAG:+--tag $TAG} "$@" > "$LOGS"/bench-$n${TAG:+-$TAG}.txt 2>&1
    echo "$(date +%T) $n exit $?"
done
echo "$(date +%T) ALL DONE"
