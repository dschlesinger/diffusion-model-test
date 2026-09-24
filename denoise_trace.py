#!/usr/bin/env python3
"""Watch Mercury edit its own past.

With `diffusing: true` the API streams full-sequence snapshots of the denoising
trajectory instead of left-to-right increments. This tool captures those
snapshots and reports, for each step, how much of the ALREADY-EMITTED text the
model went back and rewrote -- i.e. non-causal revision, the thing an
autoregressive model structurally cannot do.

  python3 denoise_trace.py "Explain diffusion models in 3 sentences."
  python3 denoise_trace.py --model mercury-2 --show-snapshots
  python3 denoise_trace.py --max-tokens 800 --json trace.json
"""

import argparse
import difflib
import json
import os
import shutil
import sys
import time

from bench_mercury import Conn, get_api_key, load_env_file, DEFAULT_BASE_URL

if sys.stdout.isatty():
    DIM, BOLD, GREEN, RED, YELLOW, OFF = (
        "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")
else:
    DIM = BOLD = GREEN = RED = YELLOW = OFF = ""


def capture(conn, key, args, on_snapshot=None):
    """Return [(t_seconds, snapshot_text), ...] for one diffusing generation.

    on_snapshot(index, t, text) is called as each snapshot lands, so --viz can
    draw frames while the response is still streaming.
    """
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "diffusing": True,
    }
    if args.effort:
        body["reasoning_effort"] = args.effort

    t0 = time.perf_counter()
    snaps = []
    resp = conn.post("/chat/completions", key, body, stream=True)
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                try:
                    resp.read()
                except Exception:
                    conn.reset()
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece is not None and piece != "":
                    t = time.perf_counter() - t0
                    snaps.append((t, piece))
                    if on_snapshot:
                        on_snapshot(len(snaps) - 1, t, piece)
    return snaps


def lay_out(prev, cur, width):
    """Wrap `cur` to `width`, flagging each char that differs from `prev`.

    Positions line up index-for-index because denoising revises in place, so a
    positional comparison is the honest one here.
    -> list of lines, each a list of (char, changed).
    """
    lines, line = [], []
    for i, ch in enumerate(cur):
        changed = i >= len(prev) or prev[i] != ch
        if ch == "\n":
            lines.append(line)
            line = []
            continue
        if len(line) >= width:
            lines.append(line)
            line = []
        line.append((ch, changed))
    lines.append(line)
    return lines


def paint(lines, color=True):
    """Render lay_out() output, highlighting changed runs."""
    out = []
    for line in lines:
        buf, active = [], None
        for ch, changed in line:
            if color and changed != active:
                buf.append(YELLOW if changed else OFF)
                active = changed
            buf.append(ch)
        if color and active:
            buf.append(OFF)
        out.append("".join(buf))
    return out


def print_steps(snaps, width, color=True):
    """Print every snapshot in full, stacked in order, so the trajectory is
    scrollable and diffable rather than overwritten in place."""
    prev = ""
    for i, (t, cur) in enumerate(snaps):
        kept, rewritten, appended = analyze(prev, cur)
        bar = "\u2500" * max(8, min(width, 64))
        head = (f"{DIM}{bar}{OFF}\n"
                f"{BOLD}step {i}{OFF}  {DIM}{t * 1000:.0f} ms  {len(cur)} chars  "
                f"{kept} kept{OFF}  "
                f"{(RED if rewritten else DIM)}{rewritten} rewritten{OFF}"
                f"{DIM}  +{appended} new{OFF}")
        print(head)
        for ln in paint(lay_out(prev, cur, width), color):
            print(ln)
        print()
        prev = cur


class Viz:
    """Redraws the denoising sequence in place using ANSI cursor movement.

    Characters that changed since the previous frame are highlighted, which is
    what makes in-place revision visible rather than just implied.
    """

    def __init__(self, width=None, height=None, delay=0.35, color=True):
        term = shutil.get_terminal_size((80, 24))
        self.width = width or max(20, term.columns - 2)
        self.max_lines = max(4, (height or term.lines) - 6)
        self.delay = delay
        self.color = color
        self.drawn = 0
        self.prev = ""
        self.last_t = None

    # -- layout --

    def _lay_out(self, cur):
        return lay_out(self.prev, cur, self.width)

    def _paint(self, lines):
        return paint(lines, self.color)

    # -- drawing --

    def frame(self, step, t, cur, stats):
        if self.delay and self.last_t is not None:
            elapsed = time.perf_counter() - self.last_t
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

        lines = self._lay_out(cur)
        truncated = len(lines) > self.max_lines
        if truncated:
            lines = lines[: self.max_lines]

        kept, rewritten, appended = stats
        header = (f"{BOLD}step {step}{OFF}  {DIM}{t * 1000:.0f} ms  "
                  f"{len(cur)} chars  {OFF}"
                  f"{RED if rewritten else DIM}{rewritten} rewritten{OFF}"
                  f"{DIM}  +{appended} new{OFF}")

        body = self._paint(lines)
        if truncated:
            body.append(f"{DIM}... ({len(self._lay_out(cur)) - self.max_lines} "
                        f"more lines){OFF}")

        buf = []
        if self.drawn:
            buf.append(f"\033[{self.drawn}A")   # cursor up to the frame's top
            buf.append("\033[J")                # clear everything below
        buf.append(header + "\n")
        buf.append("\n".join(body) + "\n")
        sys.stdout.write("".join(buf))
        sys.stdout.flush()

        self.drawn = len(body) + 1
        self.prev = cur
        self.last_t = time.perf_counter()

    def finish(self, final):
        """Repaint the last frame with no highlighting, so the result reads clean."""
        if not self.drawn:
            return
        lines = self._lay_out(final)
        truncated = len(lines) > self.max_lines
        sys.stdout.write(f"\033[{self.drawn}A\033[J")
        self.drawn = 0
        if truncated:
            sys.stdout.write(f"{DIM}(final text below){OFF}\n")
        sys.stdout.flush()


def analyze(prev, cur):
    """Compare two snapshots over their shared prefix region.

    Returns (kept, rewritten, appended): characters of prev that survived,
    characters of prev that were CHANGED, and net new characters.
    """
    overlap = min(len(prev), len(cur))
    sm = difflib.SequenceMatcher(None, prev[:overlap], cur[:overlap], autojunk=False)
    kept = sum(b.size for b in sm.get_matching_blocks())
    rewritten = overlap - kept
    appended = max(0, len(cur) - len(prev))
    return kept, rewritten, appended


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    load_env_file(os.path.join(here, ".env"))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="*", default=None)
    p.add_argument("--model", default="mercury-2.5")
    p.add_argument("--base-url", default=os.environ.get("MERCURY_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--effort", choices=("instant", "low", "medium", "high"),
                   default="instant")
    p.add_argument("--steps", action="store_true",
                   help="print every snapshot in full, stacked in order, with "
                        "characters changed since the previous step highlighted")
    p.add_argument("--show-snapshots", action="store_true",
                   help="one truncated preview line per snapshot (compact form "
                        "of --steps)")
    p.add_argument("--width", type=int, default=72, help="snapshot preview width")
    p.add_argument("--json", dest="json_out", help="save the raw trace")
    p.add_argument("--viz", action="store_true",
                   help="redraw the text in place as it denoises, highlighting "
                        "characters that changed since the last frame")
    p.add_argument("--delay", type=float, default=0.35,
                   help="minimum seconds per frame in --viz (snapshots often "
                        "arrive ~1 ms apart, too fast to see); 0 for no pacing")
    args = p.parse_args()

    args.prompt = " ".join(args.prompt).strip() if args.prompt else (
        "List the first 10 prime numbers with one clause about each.")

    key = get_api_key()
    conn = Conn(args.base_url)
    conn.warm()

    viz = None
    on_snapshot = None
    if args.viz:
        if not sys.stdout.isatty():
            sys.exit("--viz needs a terminal (it repaints with ANSI cursor moves).")
        viz = Viz(delay=args.delay)
        seen = {"prev": ""}

        def on_snapshot(i, t, text):
            viz.frame(i, t, text, analyze(seen["prev"], text))
            seen["prev"] = text

        print(f"{BOLD}{args.model}{OFF} {DIM}| denoising live | "
              f"{YELLOW}yellow{OFF}{DIM} = changed since last frame{OFF}\n")
        sys.stdout.write("\033[?25l")  # hide cursor

    try:
        snaps = capture(conn, key, args, on_snapshot=on_snapshot)
    finally:
        conn.reset()
        if args.viz:
            sys.stdout.write("\033[?25h")  # restore cursor
            sys.stdout.flush()
    if viz and snaps:
        viz.finish(snaps[-1][1])

    if not snaps:
        sys.exit("No snapshots captured -- the server may not have honored diffusing mode.")

    if not args.viz:
        print(f"{BOLD}{args.model}{OFF} {DIM}| {len(snaps)} denoising snapshots | "
              f"prompt: {args.prompt[:50]!r}{OFF}\n")
        print(f"{DIM}{'step':>4}  {'t(ms)':>7}  {'chars':>6}  {'kept':>6}  "
              f"{'REWROTE':>8}  {'new':>6}{OFF}")

    prev = ""
    total_rewritten = 0
    rows = []
    for i, (t, cur) in enumerate(snaps):
        kept, rewritten, appended = analyze(prev, cur)
        total_rewritten += rewritten
        if not args.viz:
            flag = f"{RED}{rewritten:>8}{OFF}" if rewritten else f"{DIM}{0:>8}{OFF}"
            print(f"{i:>4}  {t * 1000:>7.0f}  {len(cur):>6}  {kept:>6}  "
                  f"{flag}  {appended:>6}")
        rows.append({"step": i, "t_ms": t * 1000, "chars": len(cur),
                     "kept": kept, "rewritten": rewritten, "appended": appended})
        if args.show_snapshots and not args.viz:
            preview = cur[:args.width].replace("\n", "\\n")
            print(f"      {DIM}{preview}{OFF}")
        prev = cur

    if args.steps:
        term_w = shutil.get_terminal_size((80, 24)).columns
        print()
        print_steps(snaps, max(20, min(args.width, term_w - 2)),
                    color=sys.stdout.isatty())

    final = snaps[-1][1]
    print(f"\n{BOLD}Total characters rewritten after first being emitted: "
          f"{total_rewritten}{OFF}")
    if total_rewritten:
        print(f"{GREEN}=> Mercury revised already-generated positions. "
              f"An autoregressive model cannot do this.{OFF}")
    else:
        print(f"{YELLOW}=> No revision observed in this run; snapshots were "
              f"purely additive.{OFF}")

    print(f"\n{BOLD}Final output:{OFF}\n{final}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"model": args.model, "prompt": args.prompt,
                       "steps": rows,
                       "snapshots": [{"t_ms": t * 1000, "text": c} for t, c in snaps],
                       "final": final}, fh, indent=2)
        print(f"\n{DIM}wrote {args.json_out}{OFF}")


if __name__ == "__main__":
    main()
