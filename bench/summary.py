#!/usr/bin/env python3
"""Table of benchmark results. Usage: summary.py [TAG] [DIR]
(TAG e.g. rapida, '' = untagged runs; DIR default /data/benchmarks, e.g. ../results/2026-11-01-quick)"""
import glob
import json
import os
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else ""
root = sys.argv[2] if len(sys.argv) > 2 else "/data/benchmarks"
rows = []
for f in sorted(glob.glob(os.path.join(root, "*/results.json"))):
    name = os.path.basename(os.path.dirname(f))
    if tag and not name.endswith("-" + tag):
        continue
    if not tag and name.rsplit("-", 1)[-1] in ("rapida", "osl256"):
        continue
    r = json.load(open(f))
    for m in r.get("levels", []):
        rows.append((name.removesuffix("-" + tag) if tag else name, r.get("complete", False), m))

hdr = f"{'config':<24} {'scenario':<11} {'usr':>3} {'out t/s':>7} {'tot t/s':>7} {'t/s/usu':>7} " \
      f"{'TTFT s':>6} {'ITL ms':>6} {'W':>5} {'t/J':>5} {'core':>4} {'hot':>3} {'vram':>4} {'F4/5':>4} {'cache%':>6}"
print(hdr)
print("-" * len(hdr))
prev = None
for name, complete, m in rows:
    if prev and prev != name:
        print()
    prev = name
    f = lambda k, d=0: "-" if m.get(k) is None else f"{m[k]:.{d}f}"
    ttft = "-" if m.get("ttft_ms_avg") is None else f"{m['ttft_ms_avg'] / 1000:.1f}"
    print(f"{name + ('' if complete else '*'):<24} {m.get('scenario', 'medio'):<11} {m['concurrency']:>3} "
          f"{f('out_tok_s'):>7} {f('total_tok_s'):>7} {f('user_tok_s', 1):>7} {ttft:>6} {f('itl_ms_avg', 1):>6} "
          f"{f('avg_total_W'):>5} {f('tok_per_J', 2):>5} {f('max_core_C'):>4} {f('max_hotspot_C'):>3} "
          f"{f('max_vram_C'):>4} {f('max_case_fan_duty'):>4} {f('prefix_hit_pct'):>6}")
print("\n* = run incomplete or interrupted")
