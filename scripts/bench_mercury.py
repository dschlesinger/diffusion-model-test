#!/usr/bin/env python3
"""Throughput benchmark for Inception Labs' Mercury diffusion LLM.

Measures TTFT, decode tokens/sec and end-to-end tokens/sec over N runs.
Stdlib only -- no pip install required.

  python3 scripts/bench_mercury.py                          # default: 5 streamed runs
  python3 scripts/bench_mercury.py --runs 10 --max-tokens 1024
  python3 scripts/bench_mercury.py --diffusing              # Mercury's diffusion-style streaming
  python3 scripts/bench_mercury.py --no-stream              # e2e latency only
  python3 scripts/bench_mercury.py --list-models
"""

import argparse
import json
import os
import statistics
import sys
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_MODEL = "mercury-2"
DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how masked diffusion language "
    "models differ from autoregressive transformers. Cover the training "
    "objective, the decoding loop, and the throughput implications."
)


def load_env_file(path):
    """Minimal .env loader; does not overwrite already-set env vars."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            os.environ.setdefault(key, val)


def get_api_key():
    for name in ("MERCURY_KEY", "MERCURY_API_KEY", "INCEPTION_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    sys.exit("No API key found. Set MERCURY_KEY in .env or the environment.")


class Conn:
    """Keep-alive HTTPS connection, so the ~100 ms TLS handshake is paid once
    for the whole benchmark instead of once per request."""

    def __init__(self, base_url):
        self.parts = urllib.parse.urlsplit(base_url)
        self.path_prefix = self.parts.path.rstrip("/")
        self._c = None

    def _connect(self):
        if self._c is None:
            self._c = http.client.HTTPSConnection(
                self.parts.hostname, self.parts.port or 443, timeout=300
            )
        return self._c

    def reset(self):
        if self._c is not None:
            try:
                self._c.close()
            except Exception:
                pass
            self._c = None

    def post(self, path, key, body, stream):
        payload = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Connection": "keep-alive",
        }
        for attempt in (1, 2):
            try:
                c = self._connect()
                c.request("POST", self.path_prefix + path, body=payload, headers=headers)
                resp = c.getresponse()
                if resp.status >= 400:
                    detail = resp.read().decode(errors="replace")[:800]
                    self.reset()
                    raise RuntimeError(f"HTTP {resp.status}: {detail}")
                return resp
            except (http.client.HTTPException, OSError):
                # A pooled connection the server already closed; retry once fresh.
                self.reset()
                if attempt == 2:
                    raise

    def warm(self):
        """Pay DNS + TCP + TLS before timing starts."""
        self._connect().connect()


def list_models(base_url, key):
    req = urllib.request.Request(
        f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    for m in data.get("data", []):
        print(m.get("id"))


def run_streamed(args, key, prompt, conn):
    """One streamed request. Returns a metrics dict."""
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.diffusing:
        body["diffusing"] = True
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort

    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    text_parts = []
    usage = None
    server_ms = None

    resp = conn.post("/chat/completions", key, body, stream=True)
    with resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                # Drain anything trailing so the socket can be reused.
                try:
                    resp.read()
                except Exception:
                    conn.reset()
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("server_timing"):
                server_ms = obj["server_timing"].get("server_latency_ms")

            for choice in obj.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    text_parts.append(piece)

    total = time.perf_counter() - t0
    # In diffusing mode chunks may be whole-sequence refreshes rather than
    # appends, so trust usage.completion_tokens when the API reports it.
    out_tokens = (usage or {}).get("completion_tokens")
    token_source = "usage"
    if not out_tokens:
        out_tokens = chunks
        token_source = "chunks (usage unavailable -- approximate)"

    decode_time = total - ttft if ttft is not None else total
    details = (usage or {}).get("completion_tokens_details") or {}
    return {
        "ttft_s": ttft,
        "total_s": total,
        "decode_s": decode_time,
        "out_tokens": out_tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "server_latency_ms": server_ms,
        "chunks": chunks,
        "token_source": token_source,
        "decode_tps": out_tokens / decode_time if decode_time > 0 else 0.0,
        "e2e_tps": out_tokens / total if total > 0 else 0.0,
        "text": "".join(text_parts),
    }


def run_blocking(args, key, prompt, conn):
    """One non-streamed request. Returns a metrics dict."""
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort

    t0 = time.perf_counter()
    with conn.post("/chat/completions", key, body, stream=False) as resp:
        obj = json.loads(resp.read().decode())
    total = time.perf_counter() - t0

    usage = obj.get("usage") or {}
    out_tokens = usage.get("completion_tokens") or 0
    text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return {
        "ttft_s": None,
        "total_s": total,
        "decode_s": total,
        "out_tokens": out_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "server_latency_ms": None,
        "chunks": 0,
        "token_source": "usage",
        "decode_tps": out_tokens / total if total > 0 else 0.0,
        "e2e_tps": out_tokens / total if total > 0 else 0.0,
        "text": text,
    }


def summarize(values, unit, label):
    if not values:
        return
    mean = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"  {label:<24} mean {mean:8.2f} {unit}   "
        f"median {med:8.2f}   sd {sd:6.2f}   "
        f"min {min(values):8.2f}   max {max(values):8.2f}"
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    load_env_file(os.path.join(here, "..", ".env"))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=os.environ.get("MERCURY_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", help="read the prompt from a file instead")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="untimed runs first")
    p.add_argument("--no-stream", dest="stream", action="store_false", default=True)
    p.add_argument("--diffusing", action="store_true",
                   help="enable Mercury's diffusing stream mode (implies --stream)")
    p.add_argument("--reasoning-effort", choices=["instant", "low", "medium", "high"],
                   help="Mercury 2 reasoning budget; 'instant' for raw speed")
    p.add_argument("--json", dest="json_out", help="write per-run metrics to this file")
    p.add_argument("--show-output", action="store_true", help="print the last completion")
    p.add_argument("-v", "--verbose", action="store_true", help="per-run lines and full stats")
    p.add_argument("--bare", action="store_true", help="print only the number, nothing else")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    key = get_api_key()

    if args.list_models:
        list_models(args.base_url, key)
        return

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as fh:
            prompt = fh.read()

    conn = Conn(args.base_url)
    conn.warm()
    runner = run_streamed if (args.stream or args.diffusing) else run_blocking
    mode = "streaming" + (" + diffusing" if args.diffusing else "") if runner is run_streamed else "blocking"

    if args.verbose:
        print(f"model={args.model}  mode={mode}  max_tokens={args.max_tokens}  "
              f"reasoning_effort={args.reasoning_effort or 'default'}")
        print(f"endpoint={args.base_url}  runs={args.runs} (+{args.warmup} warmup)\n")

    try:
        for i in range(args.warmup):
            runner(args, key, prompt, conn)
            if args.verbose:
                print(f"warmup {i + 1}/{args.warmup} done")

        results = []
        for i in range(args.runs):
            r = runner(args, key, prompt, conn)
            results.append(r)
            if args.verbose:
                ttft = f"{r['ttft_s'] * 1000:7.0f} ms" if r["ttft_s"] is not None else "      n/a"
                print(
                    f"run {i + 1:>2}/{args.runs}  ttft {ttft}  "
                    f"total {r['total_s']:6.2f} s  out {r['out_tokens']:>5} tok  "
                    f"chunks {r['chunks']:>3}  e2e {r['e2e_tps']:7.1f} tok/s"
                )
    except RuntimeError as e:
        sys.exit(str(e))
    except (http.client.HTTPException, OSError) as e:
        sys.exit(f"Connection failed: {e}")
    finally:
        conn.reset()

    tps = statistics.mean([r["e2e_tps"] for r in results])

    if args.bare:
        print(f"{tps:.1f}")
    elif args.verbose:
        print(f"\n--- summary over {len(results)} runs ---")
        summarize([r["e2e_tps"] for r in results], "tok/s", "end-to-end tok/s")
        summarize([r["total_s"] for r in results], "s    ", "total latency")
        ttfts = [r["ttft_s"] * 1000 for r in results if r["ttft_s"] is not None]
        if ttfts:
            summarize(ttfts, "ms   ", "time to first token")
        summarize([float(r["out_tokens"]) for r in results], "tok  ", "output tokens")
        reas = [float(r["reasoning_tokens"]) for r in results if r.get("reasoning_tokens")]
        if reas:
            summarize(reas, "tok  ", "  of which reasoning")
        srv = [r["server_latency_ms"] for r in results if r.get("server_latency_ms")]
        if srv:
            summarize(srv, "ms   ", "server-reported latency")
        print(f"\n{tps:.1f} tok/s")
    else:
        print(f"{tps:.1f} tok/s   ({args.model}, {len(results)} runs)")

    if args.show_output:
        print("\n--- last completion ---")
        print(results[-1]["text"])

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {
                    "config": {k: v for k, v in vars(args).items() if k != "prompt"},
                    "prompt": prompt,
                    "runs": [{k: v for k, v in r.items() if k != "text"} for r in results],
                },
                fh,
                indent=2,
            )
        if args.verbose:
            print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
