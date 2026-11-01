#!/bin/sh
# Install the GPU power limit timer. Run with sudo from this directory.
set -eu
cd "$(dirname "$0")"
install -m 755 gpu-power-limit.sh /usr/local/sbin/gpu-power-limit.sh
install -m 644 gpu-power-limit.service gpu-power-limit.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gpu-power-limit.timer
systemctl start gpu-power-limit.service
journalctl -u gpu-power-limit --no-pager -n 5
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
