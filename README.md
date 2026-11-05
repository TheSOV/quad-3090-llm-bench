# quad-3090-llm-bench

vLLM throughput, latency, power and temperature measurements on **4× RTX 3090 with no PCIe P2P**,
plus the scripts that produced them. One pair has an NVLink bridge since 2026-11-04 (see [NVLink A/B](#nvlink-ab-flash-next-on-merlin)).

The machine, its fan control and its cooling quirks are documented in
[mc62-g40-quad-3090](https://github.com/TheSOV/mc62-g40-quad-3090). Hotspot and GDDR6X temperatures come from
that repo's `gpu-fancontrol` daemon (`/run/gpu-fancontrol/temps.json`), because NVML does not report them on GeForce.

## Setup

| | |
|---|---|
| GPUs | 4× RTX 3090 24 GB: 2 Founders Edition cards converted to blower coolers, 1 Gigabyte Turbo (blower), 1 Founders Edition |
| Interconnect | PCIe 4.0, **no PCIe P2P on any pair**, Resizable BAR off (BAR1 256 MB). NVLink bridge (NV4, 4 × 14.06 GB/s) between GPUs 0 and 2 from 2026-11-04; not used in the 2026-11-01 runs |
| Board / CPU | Gigabyte MC62-G40, Threadripper PRO 5965WX, 128 GB RAM |
| Software | Ubuntu 26.04, kernel 7.0, NVIDIA 615.71.09 (open), vLLM 0.32.0, torch 2.13.0+cu132, FlashInfer 0.7.0.post1, AIPerf 0.13.0 |
| Case fans | front fans driven by `gpu-fancontrol`, 40–100 % |

## Method

- [`bench/bench.py`](bench/bench.py) starts `vllm serve` for one configuration, sends a short smoke-test prompt
  (in Spanish, to catch broken non-English output), then runs AIPerf at each concurrency level.
- Synthetic prompts, streaming chat endpoint, `ignore_eos` + `min_tokens`, so every request produces exactly the
  requested number of tokens. Requests per level: max(2 × concurrency, 8).
- While AIPerf runs: NVML every 0.5 s (power, core temperature, SM clock) and the daemon's readings every 2 s (hotspot,
  GDDR6X, case-fan duty). Power is the **sum of the 4 GPUs** (board power from NVML), not wall power.
- `--max-model-len 32768`, `--gpu-memory-utilization 0.92`, everything else at vLLM defaults.

This round ("quick"): **1024-token prompt, 1024-token answer, 1, 8 and 32 concurrent users**. Columns:

- **t/s per user**: output tokens per second seen by each user (decode speed).
- **total**: output tokens per second of the whole server.
- **TTFT**: average time to the first token.
- **tokens/J**: output tokens per joule of GPU energy.

## Results

### Models and parallel layouts (280 W per GPU)

| Model | Layout | 1 user: t/s | 8 users: t/s per user / total | 32 users: t/s per user / total | TTFT at 32 | GPU W at 32 | tokens/J at 32 | max VRAM °C |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B AWQ (MoE, ~3B active) | TP4 | 176 | 107 / 791 | 44 / 1286 | 2.2 s | 783 | 1.64 | 88 |
| Qwen3.6-35B-A3B AWQ | DP2×TP2 | 143 | 95 / 705 | 47 / 1402 | 1.7 s | 1044 | 1.34 | 90 |
| gpt-oss-120b MXFP4 (MoE, ~5B active) | TP4 | 180 | 91 / 666 | 38 / 1052 | 4.1 s | 1042 | 1.01 | 92 |
| Qwen3.8-27B AWQ (dense) | TP4 | 88 | 51 / 359 | 18 / 488 | 9.3 s | 1022 | 0.48 | 96 |
| Qwen3.8-27B AWQ | DP2×TP2 | 65 | 47 / 325 | 27 / 700 | 5.4 s | 1075 | 0.65 | 98 |
| Qwen3.8-27B BF16 (dense) | TP4 | 45 | 33 / 239 | 16 / 433 | 9.5 s | 1060 | 0.41 | 102 |
| Llama 3.3 70B AWQ (dense) | TP4 | 51 | 32 / 219 | 11 / 286 | 19.5 s | 1062 | 0.27 | 100 |
| Llama 3.3 70B AWQ | TP2+PP2 ¹ | 34 | 22 / 164 | 12 / 319 | 14.7 s | 1072 | 0.30 | – |

¹ The power limit changed from 350 to 280 W during this run, and per-GPU hotspot/VRAM logging was not in place yet.
Llama 3.3 70B TP4 at 350 W is in [`results/2026-11-01-quick/llama70b-awq-tp4`](results/2026-11-01-quick/llama70b-awq-tp4):
the same speed as at 280 W (±2 %) for ~100–225 W more.

- **MoE models are 2–4× faster than dense ones here**, and the most efficient. Few active parameters per token means
  little memory traffic per token and little data exchanged between GPUs over PCIe.
- **TP4 vs. two TP2 replicas (DP2×TP2):** for dense models, replicas win with many users (+43 % at 32 users for
  Qwen3.8-27B AWQ) and lose with one. For the MoE the gain at 32 users is small (+9 %) and costs 33 % more energy.
- **Pipeline parallel (TP2+PP2)** only helped at high concurrency. Qwen3.8 is not tested with PP (vLLM issue #63505 on sm_86).
- Speed is not quality: these numbers say nothing about how well each model answers.

### Power limit sweep

Same workload, `nvidia-smi -pl` on all 4 GPUs. Temperatures: highest reading on any GPU (always the Gigabyte Turbo).
The case fans follow the GPUs, so lower power also means slower fans.

**Llama 3.3 70B AWQ, TP4**

| Limit | 1 user t/s | 8 users t/s/user | 32 users total t/s | GPU W at 32 | tokens/J at 32 | max core / hotspot / VRAM °C | case fans |
|---|---|---|---|---|---|---|---|
| 280 W | 51.0 | 31.6 | 286 | 1062 | 0.27 | 80 / 96 / 100 | 100 % |
| 250 W | 49.7 (−2 %) | 31.2 (−1 %) | 283 (−1 %) | 968 | 0.29 | 76 / 92 / 98 | 100 % |
| 220 W | 45.4 (−11 %) | 30.8 (−3 %) | 277 (−3 %) | 862 | 0.32 | 73 / 88 / 94 | 88 % |
| 200 W | 37.1 (−27 %) | 29.9 (−6 %) | 270 (−6 %) | 788 | 0.34 | 70 / 85 / 90 | 80 % |

**Qwen3.6-35B-A3B AWQ, DP2×TP2**

| Limit | 1 user t/s | 8 users t/s/user | 32 users total t/s | GPU W at 32 | tokens/J at 32 | max core / hotspot / VRAM °C | case fans |
|---|---|---|---|---|---|---|---|
| 280 W | 142.7 | 95.2 | 1393 | 1045 | 1.33 | 79 / 94 / 92 | 100 % |
| 250 W | 141.0 (−1 %) | 94.0 (−1 %) | 1392 (0 %) | 960 | 1.45 | 75 / 90 / 90 | 92–100 % |
| 220 W | 136.8 (−4 %) | 91.8 (−4 %) | 1380 (−1 %) | 856 | 1.61 | 72 / 87 / 88 | 80–88 % |
| 200 W | 132.2 (−7 %) | 89.6 (−6 %) | 1349 (−3 %) | 781 | 1.73 | 70 / 85 / 86 | 78 % |

- **250 W is nearly free** (0–2 %) and 3–4 °C cooler than 280 W.
- **220 W** costs 1–4 % (11 % for a single Llama user) and takes 7–8 °C off the core and hotspot.
- **200 W** only makes sense for the MoE: a single Llama user loses 27 %.
- GDDR6X temperature drops much less than the core with lower power (4–10 °C from 280 to 200 W).
- This machine now runs at **220 W** ([`power/`](power/)). The fan curves were made more aggressive after this
  sweep, so later runs are not directly comparable in temperature.

### NVLink A/B: Flash-Next on Merlin

2026-11-04, one pass per configuration, 220 W, 0 errors. Qwen3.8-Flash-Next served by
[Merlin](https://github.com/halt95/merlin-engine) (a vLLM 0.32.0 fork), profile `fn-quad-3090-nop2p`: TP2 × PP2 with
expert parallel, `NCCL_P2P_DISABLE=1`.

- **A:** `CUDA_VISIBLE_DEVICES=0,1,2,3`: TP pairs (0,1) and (2,3), no P2P, every NCCL channel over shared memory.
- **B:** `CUDA_VISIBLE_DEVICES=0,2,1,3` + [`patches/merlin-nccl-p2p-nvl.patch`](patches/merlin-nccl-p2p-nvl.patch)
  (`MERLIN_NCCL_P2P_NVL=1` drops `NCCL_P2P_DISABLE` and sets `NCCL_P2P_LEVEL=NVL`): the NVLink pair (0,2) is a TP
  pair and NCCL uses P2P only over NVLink. The NCCL log shows the 0–2 channels `via P2P/CUMEM`, the rest `via SHM`.

Merlin classes a host with P2P on only 1 of 6 pairs as "P2P off" and refuses its P2P profiles, hence the patch.

| Scenario (prompt / answer tokens) | Users | Total output t/s, A → B | TTFT, A → B | Prefill, 1 user (median tok/s), B vs. A |
|---|---|---|---|---|
| prefill-4k (4096 / 64) | 1 | 46 → 49 (+5 %) | 0.92 → 0.89 s | 4181 (+5 %) |
| | 8 | 105 → 113 (+8 %) | 2.38 → 2.14 s | |
| prefill-32k (32768 / 64) | 1 | 9.7 → 10.5 (+8 %) | 6.18 → 5.67 s | 5080 (+9 %) |
| | 8 | 17.6 → 19.4 (+10 %) | 22.4 → 20.1 s | |
| medio (1024 / 1024) | 1 | 127 → 122 (−4 %) | 0.36 → 0.33 s | 2860 (+9 %) |
| | 4 | 307 → 298 (−3 %) | 0.52 → 0.46 s | |
| | 8 | 417 → 454 (+9 %) | 0.80 → 0.72 s | |
| largo (8192 / 1024) | 1 | 112 → 109 (−3 %) | 1.68 → 1.55 s | 4704 (+9 %) |
| | 4 | 245 → 258 (+5 %) | 2.59 → 2.32 s | |
| | 8 | 300 → 347 (+15 %) | 4.46 → 3.97 s | |

- **Prefill +5 to +9 %, TTFT 3–12 % lower in every test, +9 to +15 % total throughput at 8 users.**
- **Single-user decode 4–6 % slower** (inter-token latency 7.51 → 7.88 ms on medio). One pass with MTP speculative
  decoding, so it may be noise, or NCCL P2P latency on small messages. Not confirmed yet.
- Power and temperatures: the same.
- B is what the server runs now (a systemd drop-in with the two environment variables).

Files: [`bench/nvlink_ab.sh`](bench/nvlink_ab.sh), [`results/2026-11-04-nvlink-ab`](results/2026-11-04-nvlink-ab).

## Files

| Path | What |
|---|---|
| [`bench/bench.py`](bench/bench.py) | One configuration: vLLM + smoke test + AIPerf per level + power/temperature sampling. Configurations in `CONFIGS` |
| [`bench/run_all.sh`](bench/run_all.sh) | Queue of configurations, skips completed ones |
| [`bench/power_sweep.sh`](bench/power_sweep.sh) | Power-limit sweep (sudo; bench.py runs as the calling user), restores the limit at the end |
| [`bench/summary.py`](bench/summary.py) | Table of results: `python3 bench/summary.py "" results/2026-11-01-quick` |
| [`bench/realtext.py`](bench/realtext.py) | Decode speed on real prompts (code, explanation, prose), for models with MTP; `bench.py --realtext 1,2,5` |
| [`bench/nvlink_ab.sh`](bench/nvlink_ab.sh) | NVLink A/B on Flash-Next (Merlin) |
| [`patches/merlin-nccl-p2p-nvl.patch`](patches/merlin-nccl-p2p-nvl.patch) | Merlin opt-in: NCCL P2P over NVLink only on a P2P-off profile |
| [`bench/dl_limited.sh`](bench/dl_limited.sh) | Rate-limited, resumable Hugging Face download (a full-speed download saturated the link here) |
| [`power/`](power/) | systemd timer that sets the GPU power limit at boot and re-checks it every 5 min |
| `results/<round>/<config>/results.json` | Metrics per concurrency level (AIPerf + power + temperatures + prefix-cache hits) |
| `results/<round>/<config>/gpu_samples.csv` | NVML samples, 0.5 s: power, core °C, SM clock per GPU |
| `results/<round>/<config>/daemon_samples.jsonl` | Daemon readings, 2 s: core/hotspot/VRAM per GPU and case-fan duty |

vLLM logs and raw AIPerf output are not included.

## Notes

- **FlashInfer JIT needs `nvcc`.** With pip packages only: `nvidia-cuda-nvcc`, `nvidia-cuda-crt` and `nvidia-nvvm`
  pinned to **13.2.78** (13.4's NVVM emits PTX 9.4, which the 13.2 `ptxas` rejects), a `libcudart.so` symlink in
  `nvidia/cu13/lib64`, and `CUDA_HOME` pointing at `site-packages/nvidia/cu13` (see `ENV` in `bench.py`).
- Prefix-cache hits are non-zero even with random prompts (6–37 %): the chat template and AIPerf's synthetic text repeat.
