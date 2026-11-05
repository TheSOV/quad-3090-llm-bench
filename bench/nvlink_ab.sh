#!/bin/bash
# A/B: Flash-Next on Merlin (profile fn-quad-3090-nop2p, TP2 x PP2) with the default GPU order vs. the NVLink-bridged
# pair as a TP pair, NCCL P2P over NVLink only. Needs patches/merlin-nccl-p2p-nvl.patch applied to merlin-engine.
# The inference service must be stopped (RAM and GPUs). Run as the normal user:
#   sudo systemctl stop flashnext && bench/nvlink_ab.sh ; sudo systemctl start flashnext
set -u
cd "$(dirname "$0")"
if systemctl is-active --quiet flashnext; then echo "flashnext is still running: sudo systemctl stop flashnext"; exit 1; fi
SC=${SC:-prefill-4k,prefill-32k,medio,largo}
CONC=${CONC:-1,4,8}
COMMON=(flashnext-merlin-nop2p --power 220 --scenarios "$SC" --conc "$CONC" --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=INIT)

echo "== A: order 0,1,2,3 (TP (0,1)(2,3), no P2P) $(date +%T)"
python3 bench.py "${COMMON[@]}" --tag nvab-A-0123 --env CUDA_VISIBLE_DEVICES=0,1,2,3 || { echo "A failed: stopping"; exit 1; }

echo "== B: order 0,2,1,3 (TP (0,2) over NVLink, (1,3) no P2P) $(date +%T)"
python3 bench.py "${COMMON[@]}" --tag nvab-B-0213-nvl --env CUDA_VISIBLE_DEVICES=0,2,1,3 --env MERLIN_NCCL_P2P_NVL=1 \
  || echo "B failed: see /data/benchmarks/flashnext-merlin-nop2p-220W-nvab-B-0213-nvl/vllm.log"

for t in A-0123 B-0213-nvl; do
  d=/data/benchmarks/flashnext-merlin-nop2p-220W-nvab-$t
  echo "== $t: NCCL channels"; grep -hoE 'via (P2P|SHM|NET)[^ ]*' "$d/vllm.log" | sort | uniq -c
done
