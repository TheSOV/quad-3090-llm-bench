# GPU power limit at boot

`nvidia-smi -pl` is lost on reboot, and without persistence mode the driver can also drop it when no process has the
GPUs open. The timer runs `gpu-power-limit.sh` 20 s after boot and every 5 minutes; it only changes GPUs whose limit
differs from `WATTS`.

```bash
sudo ./install.sh                                   # installs, enables and runs it once
journalctl -u gpu-power-limit                       # one line per change
```

Change the limit: edit `Environment=WATTS=220` in `/etc/systemd/system/gpu-power-limit.service`, then
`sudo systemctl daemon-reload` (applied within 5 minutes).
