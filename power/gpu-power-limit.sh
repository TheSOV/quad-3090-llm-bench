#!/bin/sh
# Set the power limit on every GPU to $WATTS if it differs (run by gpu-power-limit.service).
set -eu
WATTS=${WATTS:-220}
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits | while IFS=', ' read -r i cur; do
    cur=${cur%%.*}
    if [ "$cur" != "$WATTS" ]; then
        nvidia-smi -i "$i" -pl "$WATTS" > /dev/null
        echo "GPU $i: power limit $cur W -> $WATTS W"
    fi
done
