#!/bin/sh
# Set the power limit on every GPU to $WATTS if it differs (run by gpu-power-limit.service).
# WATTS_BY_UUID overrides it per card, by UUID prefix (the card keeps its UUID when it changes slot):
#   WATTS_BY_UUID="GPU-25a0c115=260 GPU-xxxxxxxx=240"
set -eu
WATTS=${WATTS:-220}
WATTS_BY_UUID=${WATTS_BY_UUID:-}
nvidia-smi --query-gpu=index,uuid,power.limit --format=csv,noheader,nounits | while IFS=', ' read -r i uuid cur; do
    cur=${cur%%.*}
    want=$WATTS
    for pair in $WATTS_BY_UUID; do
        case "$uuid" in "${pair%%=*}"*) want=${pair#*=} ;; esac
    done
    if [ "$cur" != "$want" ]; then
        nvidia-smi -i "$i" -pl "$want" > /dev/null
        echo "GPU $i ($uuid): power limit $cur W -> $want W"
    fi
done
