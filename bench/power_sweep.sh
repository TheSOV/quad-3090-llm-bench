#!/bin/bash
# Power-limit sweep. Run ONCE with sudo:  sudo ./power_sweep.sh
# - waits for the quick round (run_all.sh / bench.py) to finish
# - for each limit: nvidia-smi -pl, then bench.py as the normal user (root only changes the power limit)
# - always restores RESTORE_W at the end, also on error or Ctrl+C
set -u
USER_NAME=${SUDO_USER:?run with sudo}          # bench.py runs as this user
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
BENCH=$(cd "$(dirname "$0")" && pwd)
LOGS=${LOG_DIR:-$BENCH/../logs}
mkdir -p "$LOGS"
LOG=$LOGS/power_sweep.txt
LIMITS=${LIMITS:-"280 250 220 200"}
CONFIGS=${CONFIGS:-"llama70b-awq-tp4 qwen36-35b-awq-dp2tp2"}
RESTORE_W=${RESTORE_W:-280}

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
exec >> "$LOG" 2>&1
restore() { nvidia-smi -pl "$RESTORE_W" > /dev/null && echo "$(date +%T) power limit restored to $RESTORE_W W"; }
trap restore EXIT

echo "$(date +%T) sweep: limits [$LIMITS], configs [$CONFIGS]; waiting for other benchmarks to finish"
while pgrep -f "run_all[.]sh|bench[.]py" > /dev/null; do sleep 30; done

for W in $LIMITS; do
    nvidia-smi -pl "$W" > /dev/null || { echo "$(date +%T) could not set $W W"; exit 1; }
    echo "$(date +%T) limit $W W: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"
    for c in $CONFIGS; do
        echo "$(date +%T)   run $c at $W W"
        sudo -u "$USER_NAME" -H "$HOME_DIR/vllm-env/bin/python" -I "$BENCH/bench.py" "$c" \
            --power "$W" --tag rapida --scenarios medio --conc 1,8,32 \
            > "$LOGS/bench-$c-${W}W-rapida.txt" 2>&1
        echo "$(date +%T)   $c exit $?"
    done
done
echo "$(date +%T) SWEEP DONE"
