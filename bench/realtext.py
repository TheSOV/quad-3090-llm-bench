#!/usr/bin/env python3
"""Decode speed on real prompts (code, explanation, prose), for models with speculative decoding (MTP).

Random-token prompts (bench.py / AIPerf) give MTP almost nothing to predict, so they understate its gain.
Here each request streams a real answer; per request we take decode tok/s = (completion_tokens - 1) /
(time of last token - time of first token), so time to first token (prefill) is left out.

Usage: realtext.py --port 8000 --model NAME [--conc 1,2,5] [--max-tokens 1024] [--effort medium] --out FILE.json
The API key, if any, is read from the VLLM_API_KEY environment variable.
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    ("code", "Escribe en Python una clase LRUCache con get y put en O(1), usando un diccionario y una lista "
             "doblemente enlazada propia (sin OrderedDict). Incluye docstrings y un pequeño bloque de pruebas."),
    ("code", "Implement in TypeScript a debounced async search hook for React (useDebouncedSearch) that cancels "
             "stale requests with AbortController. Include types and a usage example."),
    ("explain", "Explica cómo funciona la atención en un transformer: consultas, claves, valores, la escala por "
                "la raíz de d_k y por qué hace falta la máscara causal en un modelo de lenguaje."),
    ("explain", "Explain the difference between TCP congestion control algorithms Reno, CUBIC and BBR, and when "
                "each one behaves better or worse."),
    ("prose", "Escribe un ensayo de unos 800 palabras sobre cómo cambió el trabajo de oficina con la llegada del "
              "correo electrónico, con introducción, tres apartados y conclusión."),
]


def one(port, model, prompt, max_tokens, effort, key):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": effort},
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
    n = (usage or {}).get("completion_tokens", 0)
    if not first or n < 2 or last <= first:
        return None
    return {"ttft_s": round(first - t0, 3), "tokens": n, "decode_tok_s": round((n - 1) / (last - first), 1)}


def level(port, model, conc, max_tokens, effort, key, rounds):
    """conc users at once, each sending the prompts in turn; `rounds` prompts per user."""
    out, lock = [], threading.Lock()

    def user(u):
        for i in range(rounds):
            kind, p = PROMPTS[(u + i) % len(PROMPTS)]
            try:
                m = one(port, model, p, max_tokens, effort, key)
            except Exception as ex:
                m = {"error": str(ex)}
            if m:
                with lock:
                    out.append({"kind": kind, **m})

    t0 = time.perf_counter()
    ts = [threading.Thread(target=user, args=(u,)) for u in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    ok = [m for m in out if "decode_tok_s" in m]
    res = {"concurrency": conc, "requests": len(out), "errors": len(out) - len(ok), "wall_s": round(wall, 1)}
    if ok:
        per = [m["decode_tok_s"] for m in ok]
        res.update({"decode_tok_s_user_median": round(statistics.median(per), 1),
                    "decode_tok_s_user_min": min(per),
                    "aggregate_out_tok_s": round(sum(m["tokens"] for m in ok) / wall, 1),
                    "by_kind": {k: round(statistics.median([m["decode_tok_s"] for m in ok if m["kind"] == k]), 1)
                                for k in sorted({m["kind"] for m in ok})}})
    res["samples"] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", default="1,2,5")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--rounds", type=int, default=3, help="prompts per user and level")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY")
    one(a.port, a.model, PROMPTS[0][1], 64, a.effort, key)  # warm-up
    results = {"model": a.model, "max_tokens": a.max_tokens, "effort": a.effort, "levels": []}
    for c in [int(x) for x in a.conc.split(",")]:
        r = level(a.port, a.model, c, a.max_tokens, a.effort, key, a.rounds)
        print(json.dumps({k: v for k, v in r.items() if k != "samples"}), flush=True)
        results["levels"].append(r)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
