#!/usr/bin/env python3
"""HTTP front end for llama.cpp's llama-diffusion-gemma-visual-server.

That binary only speaks a stdin/stdout line protocol (see eval_mmlu.py), so nothing
outside the local machine can call it. This wraps one persistent instance of it and
exposes an OpenAI-compatible /v1/chat/completions endpoint over HTTP instead. The
underlying binary handles one request at a time (it's a single subprocess reading
one line of stdin at a time), so requests here are serialized behind a lock too.

  python3 scripts/diffusion_http_server.py --model /path/to/model-Q8_0.gguf

Port defaults to 0 -- the OS picks a free one, printed on startup (and to
diffusion_http.log under qsub) as "serving on http://<hostname>:<port>". Shared
SCC nodes may already have a fixed port like 8000 taken by someone else's job,
so don't assume you got the port you asked for -- read it back from the log.

Then, from any machine that can reach host:port (see docs/diffusiongemma-scc.md for
tunneling from an SCC compute node):

  curl http://HOST:PORT/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Explain diffusion models in 3 sentences."}]}'
"""

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from eval_mmlu import DiffusionServer, split_reply

server_lock = threading.Lock()
diffusion_server = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"bad json: {e}"})
            return

        messages = req.get("messages")
        if not messages:
            self._send_json(400, {"error": "\"messages\" is required"})
            return
        n_blocks = req.get("n_blocks", 8)
        seed = req.get("seed", 0)

        t0 = time.perf_counter()
        with server_lock:
            raw, stats, error = diffusion_server.generate(messages, n_blocks, seed)
        elapsed = time.perf_counter() - t0
        final, had_thought, truncated = split_reply(raw)

        if error and not final:
            self._send_json(502, {"error": error, "stats": stats})
            return

        self._send_json(200, {
            "id": f"diffusiongemma-{int(t0 * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "diffusiongemma"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final},
                "finish_reason": "length" if truncated else "stop",
            }],
            "diffusion": {
                "had_thought": had_thought,
                "truncated": truncated,
                "error": error,
                "stats": stats,
                "wall_seconds": elapsed,
            },
        })


def main():
    global diffusion_server

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the DiffusionGemma GGUF")
    ap.add_argument("--server", default="llama-diffusion-gemma-visual-server",
                     help="path to the server binary (default: from PATH)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0,
                     help="0 (default) = let the OS pick a free port; shared SCC nodes may already "
                          "have your requested port taken by another job")
    ap.add_argument("--ngl", type=int, default=99, help="layers on GPU")
    ap.add_argument("--maxtok", type=int, default=0, help="context budget (0 = auto-size to VRAM)")
    args = ap.parse_args()

    diffusion_server = DiffusionServer(args.server, args.model, args.ngl, args.maxtok, "diffusion_http.log")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    bound_host, bound_port = httpd.server_address
    print(f"model loaded, serving on http://{socket.gethostname()}:{bound_port}", file=sys.stderr)
    print(f"tunnel from your machine: ssh -L {bound_port}:{socket.gethostname()}:{bound_port} "
          f"$USER@scc1.bu.edu", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        diffusion_server.close()


if __name__ == "__main__":
    main()
