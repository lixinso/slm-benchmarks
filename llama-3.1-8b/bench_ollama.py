#!/usr/bin/env python3

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional, Tuple


OLLAMA_HOST_DEFAULT = "http://localhost:11434"


def _iter_json_lines(resp) -> Iterable[Dict[str, Any]]:
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        yield json.loads(line)


def ollama_chat_stream(
    host: str,
    model: str,
    prompt: str,
    options: Dict[str, Any],
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Stream a chat completion from Ollama.

    Returns:
      (ttft_seconds, final_stats_json)

    final_stats_json is the chunk where done==true, which includes eval durations/counts.
    """

    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": options,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    final: Optional[Dict[str, Any]] = None

    with urllib.request.urlopen(req, timeout=600) as resp:
        for obj in _iter_json_lines(resp):
            if first_token_time is None:
                msg = obj.get("message") or {}
                content = msg.get("content")
                if content:
                    first_token_time = time.perf_counter()
            if obj.get("done") is True:
                final = obj
                break

    if first_token_time is None:
        first_token_time = time.perf_counter()

    return first_token_time - start, final


def load_prompts(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ns_to_s(ns: Optional[int]) -> Optional[float]:
    if ns is None:
        return None
    return ns / 1e9


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0]
    # statistics.quantiles with n=20 gives 5% increments; index 18 is ~95th percentile.
    return statistics.quantiles(values, n=20)[18]


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark an Ollama model via the local HTTP API")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default=OLLAMA_HOST_DEFAULT)
    ap.add_argument("--prompts", default="llama-3.1-8b/prompts.jsonl")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--num_predict", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_predict": args.num_predict,
        "seed": args.seed,
    }

    results: list[Dict[str, Any]] = []

    try:
        for item in load_prompts(args.prompts):
            pid = item.get("id", "unknown")
            prompt = item["prompt"]

            for r in range(args.runs):
                wall_start = time.perf_counter()
                ttft, final = ollama_chat_stream(args.host, args.model, prompt, options)
                wall = time.perf_counter() - wall_start

                prompt_eval_count = final.get("prompt_eval_count") if final else None
                prompt_eval_s = ns_to_s(final.get("prompt_eval_duration")) if final else None
                eval_count = final.get("eval_count") if final else None
                eval_s = ns_to_s(final.get("eval_duration")) if final else None

                gen_toks_per_s = safe_div(float(eval_count) if eval_count is not None else None, eval_s)
                prompt_toks_per_s = safe_div(
                    float(prompt_eval_count) if prompt_eval_count is not None else None,
                    prompt_eval_s,
                )

                row = {
                    "id": pid,
                    "run": r,
                    "ttft_s": ttft,
                    "wall_s": wall,
                    "prompt_eval_count": prompt_eval_count,
                    "prompt_eval_s": prompt_eval_s,
                    "eval_count": eval_count,
                    "eval_s": eval_s,
                    "prompt_toks_per_s": prompt_toks_per_s,
                    "gen_toks_per_s": gen_toks_per_s,
                }
                results.append(row)
                print(json.dumps(row, ensure_ascii=False))

    except urllib.error.URLError as e:
        raise SystemExit(
            f"Failed to reach Ollama at {args.host}. Is Ollama running? Error: {e}"
        ) from e

    def summarize(key: str) -> None:
        vals = [x[key] for x in results if x.get(key) is not None]
        if not vals:
            print(f"\nsummary.{key}: no data")
            return
        vals_f = [float(v) for v in vals]
        print(
            f"\nsummary.{key}: n={len(vals_f)} "
            f"mean={statistics.mean(vals_f):.4f} "
            f"p50={statistics.median(vals_f):.4f} "
            f"p95={_p95(vals_f):.4f}"
        )

    summarize("ttft_s")
    summarize("wall_s")
    summarize("gen_toks_per_s")
    summarize("prompt_toks_per_s")


if __name__ == "__main__":
    main()
