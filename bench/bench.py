#!/usr/bin/env python3
"""Benchmark one vLLM configuration with AIPerf, sampling GPU power and temperatures.

Usage: bench.py NAME [--conc 1,8,32,64] [--isl 1024] [--osl 256] [--power W]
NAME is a key of CONFIGS below. Results go to /data/benchmarks/<NAME>[-<W>W]/.

Each concurrency level: AIPerf with synthetic prompts (ISL in, OSL out, ignore_eos so every request
produces exactly OSL tokens), streaming chat endpoint. Meanwhile NVML is sampled every 0.5 s for
power, core temperature, fan and SM clock on the GPUs in use.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import pynvml as nv

VLLM = os.path.expanduser("~/vllm-env/bin/vllm")
AIPERF = os.path.expanduser("~/aiperf-env/bin/aiperf")
OUT = "/data/benchmarks"
TEMPS_FILE = "/run/gpu-fancontrol/temps.json"   # core/hotspot/VRAM of every GPU, from gpu-fancontrol (2 s)
CUDA_HOME = os.path.expanduser("~/vllm-env/lib/python3.12/site-packages/nvidia/cu13")  # pip nvcc 13.2 (FlashInfer JIT)
ENV = {"HF_HOME": "/data/huggingface", "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
       "CUDA_HOME": CUDA_HOME,
       "PATH": CUDA_HOME + "/bin:" + os.path.expanduser("~/vllm-env/bin") + ":" + os.environ.get("PATH", "")}

# name: (model, gpus, vllm args)
# GPU pairs for TP2: 2+3 share a PCIe host bridge (PHB), 0+1 go through the CPU. No P2P on any pair.
# Avoid PP with Qwen3.8 (GDN): vLLM issue #63505 (garbage / Xid 31 with PP on sm_86, 0.28-0.32).
CONFIGS = {
    "llama70b-awq-tp4":   ("casperhansen/llama-3.3-70b-instruct-awq", "0,1,2,3",
                           ["-tp", "4", "--max-model-len", "32768"]),
    "llama70b-awq-tp2pp2": ("casperhansen/llama-3.3-70b-instruct-awq", "0,1,2,3",
                           ["-tp", "2", "-pp", "2", "--max-model-len", "32768"]),
    "gptoss120b-tp4":     ("/data/models/gpt-oss-120b", "0,1,2,3",
                           ["-tp", "4", "--max-model-len", "32768"]),
    "qwen38-27b-bf16-tp4": ("Qwen/Qwen3.8-27B", "0,1,2,3",
                           ["-tp", "4", "--max-model-len", "32768", "--language-model-only"]),
    "qwen38-27b-awq-tp4": ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "0,1,2,3",
                           ["-tp", "4", "--max-model-len", "32768", "--language-model-only"]),
    "qwen38-27b-awq-dp2tp2": ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "0,1,2,3",
                           ["-tp", "2", "-dp", "2", "--max-model-len", "32768", "--language-model-only"]),
    "qwen36-35b-awq-tp4": ("QuantTrio/Qwen3.6-35B-A3B-AWQ", "0,1,2,3",
                           ["-tp", "4", "--max-model-len", "32768", "--language-model-only"]),
    "qwen36-35b-awq-dp2tp2": ("QuantTrio/Qwen3.6-35B-A3B-AWQ", "0,1,2,3",
                           ["-tp", "2", "-dp", "2", "--max-model-len", "32768", "--language-model-only"]),
    "qwen38-27b-awq-tp2": ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "0,1",
                           ["-tp", "2", "--max-model-len", "32768", "--language-model-only"]),
    "qwen36-35b-awq-tp2": ("QuantTrio/Qwen3.6-35B-A3B-AWQ", "2,3",
                           ["-tp", "2", "--max-model-len", "32768", "--language-model-only"]),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Sampler(threading.Thread):
    """NVML samples for the given GPU indices until stopped."""

    def __init__(self, gpus, path):
        super().__init__(daemon=True)
        self.h = [(i, nv.nvmlDeviceGetHandleByIndex(i)) for i in gpus]
        self.f = open(path, "w")
        self.f.write("t,gpu,W,core_C,fan_rpm,sm_mhz\n")
        self.stop_ev, self.rows = threading.Event(), []
        self.daemon_rows = []   # (t, duty, [(gpu, core, hotspot, vram)]) from gpu-fancontrol
        self.daemon_f = open(path.replace("gpu_samples.csv", "daemon_samples.jsonl"), "w")
        self.last_daemon_t = 0

    def run(self):
        while not self.stop_ev.is_set():
            t = time.time()
            for i, h in self.h:
                w = nv.nvmlDeviceGetPowerUsage(h) / 1000
                c = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                try:
                    rpm = nv.nvmlDeviceGetFanSpeedRPM(h, 0).speed
                except Exception:
                    rpm = -1
                mhz = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
                self.rows.append((t, i, w, c, rpm, mhz))
                self.f.write(f"{t:.1f},{i},{w:.1f},{c},{rpm},{mhz}\n")
            self.f.flush()
            try:
                d = json.load(open(TEMPS_FILE))
                if d["t"] != self.last_daemon_t:
                    self.last_daemon_t = d["t"]
                    self.daemon_rows.append((d["t"], d.get("duty"),
                                             [(g["gpu"], g["core"], g["hotspot"], g["vram"]) for g in d["gpus"]]))
                    self.daemon_f.write(json.dumps(d) + "\n")
                    self.daemon_f.flush()
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.5)

    def summary(self, t0, t1):
        rows = [r for r in self.rows if t0 <= r[0] <= t1]
        if not rows:
            return {}
        ts = sorted({r[0] for r in rows})
        total_w = sum(r[2] for r in rows) / len(ts)
        extra = {}
        drows = [r for r in self.daemon_rows if t0 <= r[0] <= t1]
        if drows:
            mx = lambda i: max((g[i] for r in drows for g in r[2] if g[i] is not None), default=None)
            extra = {"max_hotspot_C": mx(2), "max_vram_C": round(mx(3)) if mx(3) is not None else None,
                     "max_case_fan_duty": max((r[1] for r in drows if r[1] is not None), default=None),
                     "per_gpu_max": {g[0]: [max(x[1] for r in drows for x in r[2] if x[0] == g[0]),
                                            max((x[2] or 0) for r in drows for x in r[2] if x[0] == g[0]),
                                            round(max((x[3] or 0) for r in drows for x in r[2] if x[0] == g[0]))]
                                     for g in drows[0][2]}}
        return {**extra, "avg_total_W": round(total_w, 1),
                "max_core_C": max(r[3] for r in rows),
                "max_fan_rpm": max(r[4] for r in rows),
                "avg_sm_mhz": round(sum(r[5] for r in rows) / len(rows)),
                "energy_J": round(total_w * (t1 - t0))}

    def close(self):
        self.stop_ev.set()
        self.join(timeout=3)
        self.f.close()
        self.daemon_f.close()


def wait_ready(port, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return time.time() - t0
        except Exception:
            time.sleep(5)
    raise TimeoutError("vLLM did not become ready")


def smoke(port, model):
    """Deterministic Spanish + arithmetic answer, to catch garbage output before measuring speed."""
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                          "Responde en español en dos frases: ¿cuál es la capital de Francia y cuánto es 17 por 23?"}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        return r["choices"][0]["message"].get("content") or r["choices"][0]["message"].get("reasoning_content") or ""
    except Exception as ex:
        return f"SMOKE FAILED: {ex}"


def prefix_cache_counters(port):
    """(hits, queries) summed over vLLM's prefix-cache counters (tokens)."""
    try:
        txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    except Exception:
        return None
    hits = queries = 0.0
    for line in txt.splitlines():
        if line.startswith("#") or "prefix_cache" not in line or "external" in line:
            continue
        name, val = line.split("{")[0], float(line.rsplit(" ", 1)[1])
        if name.endswith(("prefix_cache_hits_total", "prefix_cache_hits")):
            hits += val
        elif name.endswith(("prefix_cache_queries_total", "prefix_cache_queries")):
            queries += val
    return hits, queries


# name: (prompt tokens unique to each request, shared prefix tokens, output tokens, concurrency levels)
SCENARIOS = {
    "corto":       (256, 0, 1024, [1, 8, 32, 64]),
    "medio":       (1024, 0, 1024, [1, 8, 32, 64]),
    "largo":       (8192, 0, 1024, [1, 4, 16]),
    "largo-cache": (512, 7680, 1024, [1, 4, 16, 32]),
}


def aiperf_metrics(d):
    """Pick the main numbers from AIPerf's JSON export."""
    path = next((os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
                 if f == "profile_export_aiperf.json"), None)
    if not path:
        return {}
    j = json.load(open(path))
    def g(key, stat="avg"):
        v = j.get(key)
        return round(v[stat], 2) if isinstance(v, dict) and v.get(stat) is not None else None
    return {"out_tok_s": g("output_token_throughput"),
            "in_tok_s": g("input_token_throughput"),
            "total_tok_s": g("total_token_throughput"),
            "prefill_tok_s_user": g("prefill_throughput_per_user"),
            "req_s": g("request_throughput"),
            "ttft_ms_avg": g("time_to_first_token"), "ttft_ms_p99": g("time_to_first_token", "p99"),
            "itl_ms_avg": g("inter_token_latency"), "itl_ms_p99": g("inter_token_latency", "p99"),
            "user_tok_s": g("output_token_throughput_per_user"),
            "requests": g("request_count"), "errors": g("error_request_count")}


def _sigterm(*_):
    raise KeyboardInterrupt  # run the finally: block, which stops vLLM


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=CONFIGS)
    ap.add_argument("--scenarios", default="corto,medio,largo,largo-cache",
                    help="comma list of: " + ", ".join(SCENARIOS))
    ap.add_argument("--conc", default="", help="override concurrency levels, e.g. 1,8")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--power", type=int, help="label only; set the limit beforehand with nvidia-smi -pl")
    ap.add_argument("--extra", default="", help="extra vllm serve args, e.g. \"--kv-cache-dtype fp8\"")
    ap.add_argument("--env", action="append", default=[], help="extra env for vllm, KEY=VALUE")
    ap.add_argument("--tag", default="", help="suffix for the results directory")
    a = ap.parse_args()

    model, gpus, args = CONFIGS[a.name]
    args = args + a.extra.split()
    out = os.path.join(OUT, a.name + (f"-{a.power}W" if a.power else "") + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out, exist_ok=True)
    nv.nvmlInit()
    gl = [int(x) for x in gpus.split(",")]

    env = dict(os.environ, **ENV, CUDA_VISIBLE_DEVICES=gpus, **dict(e.split("=", 1) for e in a.env))
    cmd = [VLLM, "serve", model, "--port", str(a.port), "--gpu-memory-utilization", "0.92", *args]
    log("starting:", " ".join(cmd[1:]), f"(GPUs {gpus})")
    srv_log = open(os.path.join(out, "vllm.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True)
    results = {"name": a.name, "model": model, "gpus": gpus, "vllm_args": args,
               "power_label_W": a.power, "env": a.env, "levels": []}
    sampler = Sampler(gl, os.path.join(out, "gpu_samples.csv"))
    sampler.start()
    try:
        results["startup_s"] = round(wait_ready(a.port, proc))
        log(f"ready in {results['startup_s']} s")
        results["smoke"] = smoke(a.port, model)
        log("smoke:", results["smoke"][:300].replace("\n", " "))
        for sc in a.scenarios.split(","):
            isl, prefix, osl, levels = SCENARIOS[sc]
            if a.conc:
                levels = [int(x) for x in a.conc.split(",")]
            for c in levels:
                d = os.path.join(out, sc, f"c{c}")
                n = max(2 * c, 8)   # 1024-token outputs: ~1000 ITL samples per request
                acmd = [AIPERF, "profile", "--model", model, "--tokenizer", model,
                        "--url", f"http://127.0.0.1:{a.port}", "--endpoint-type", "chat", "--streaming",
                        "--synthetic-input-tokens-mean", str(isl), "--synthetic-input-tokens-stddev", "0",
                        "--output-tokens-mean", str(osl), "--output-tokens-stddev", "0",
                        "--extra-inputs", "ignore_eos:true", "--extra-inputs", f"min_tokens:{osl}",
                        "--concurrency", str(c), "--request-count", str(n),
                        "--warmup-request-count", str(min(c, 8)),
                        "--artifact-dir", d, "--ui", "none", "--no-gpu-telemetry", "--no-server-metrics"]
                if prefix:
                    acmd += ["--shared-system-prompt-length", str(prefix)]
                pc0 = prefix_cache_counters(a.port)
                t0 = time.time()
                r = subprocess.run(acmd, env=dict(os.environ, HF_HOME=ENV["HF_HOME"]),
                                   capture_output=True, text=True)
                t1 = time.time()
                pc1 = prefix_cache_counters(a.port)
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, "aiperf.log"), "w").write(r.stdout + r.stderr)
                m = {"scenario": sc, "isl": isl + prefix, "shared_prefix": prefix, "osl": osl,
                     "concurrency": c, "wall_s": round(t1 - t0), **aiperf_metrics(d), **sampler.summary(t0, t1)}
                if pc0 and pc1 and pc1[1] > pc0[1]:
                    m["prefix_hit_pct"] = round(100 * (pc1[0] - pc0[0]) / (pc1[1] - pc0[1]), 1)
                if m.get("avg_total_W"):
                    if m.get("out_tok_s"):
                        m["tok_per_J"] = round(m["out_tok_s"] / m["avg_total_W"], 3)
                    if m.get("total_tok_s"):
                        m["total_tok_per_J"] = round(m["total_tok_s"] / m["avg_total_W"], 3)
                results["levels"].append(m)
                log(json.dumps(m))
                if r.returncode != 0:
                    log(f"aiperf exited {r.returncode}; see {d}/aiperf.log")
                json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
        results["complete"] = True
    finally:
        sampler.close()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
        log("saved", os.path.join(out, "results.json"))


if __name__ == "__main__":
    main()
