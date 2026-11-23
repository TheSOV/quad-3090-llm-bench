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

A different limit for one card: `WATTS_BY_UUID`, matched by UUID prefix so it follows the card if it moves
to another slot. A drop-in keeps it out of the unit file:

```ini
# /etc/systemd/system/gpu-power-limit.service.d/card-25a0c115.conf
[Service]
Environment="WATTS_BY_UUID=GPU-25a0c115=260"
```

Used here from 2026-11-18 to 11-23 on the card with recurring Xid 62; 260 W did not stop the failures
(four in five days), so that card is back at 220 W (see the
[MC62-G40 notes](https://github.com/TheSOV/mc62-g40-quad-3090/blob/main/docs/gpu-xid62-pmu-halt.md)).
