#!/usr/bin/env python3
"""Smoke-test a running diffusion_http_server.py.

  python3 scripts/test_diffusion_http.py --url http://localhost:8000
  python3 scripts/test_diffusion_http.py --url http://localhost:8000 --prompt "Explain diffusion models in 3 sentences."
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def get(url, timeout):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def post(url, body, timeout):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="base URL of diffusion_http_server.py")
    ap.add_argument("--prompt", default="Explain diffusion models in 3 sentences.")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120, help="seconds, generation can be slow")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"GET  {base}/healthz")
    try:
        status, body = get(f"{base}/healthz", args.timeout)
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"FAILED to reach server: {e}", file=sys.stderr)
        print("Is diffusion_http_server.py running, and is the tunnel/port forward up?", file=sys.stderr)
        sys.exit(1)
    print(f"  {status} {body}")
    if status != 200:
        sys.exit(1)

    print(f"\nPOST {base}/v1/chat/completions")
    print(f"  prompt: {args.prompt!r}")
    t0 = time.perf_counter()
    status, body = post(f"{base}/v1/chat/completions", {
        "messages": [{"role": "user", "content": args.prompt}],
        "n_blocks": args.n_blocks,
    }, args.timeout)
    elapsed = time.perf_counter() - t0

    if status != 200:
        print(f"  FAILED ({status}): {body}", file=sys.stderr)
        sys.exit(1)

    reply = body["choices"][0]["message"]["content"]
    diag = body.get("diffusion", {})
    print(f"  {status} in {elapsed:.1f}s (finish_reason={body['choices'][0]['finish_reason']})")
    print(f"  stats: {diag.get('stats')}")
    print(f"\n--- reply ---\n{reply}\n-------------")


if __name__ == "__main__":
    main()
