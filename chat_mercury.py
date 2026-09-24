#!/usr/bin/env python3
"""Interactive chat with Inception Labs' Mercury diffusion LLM.

Streams responses, keeps conversation history, and reuses one TLS connection.
Stdlib only.

  python3 chat_mercury.py
  python3 chat_mercury.py --model mercury-2.5 --effort high
  python3 chat_mercury.py --system "You are a terse Rust expert."
  echo "explain diffusion LMs" | python3 chat_mercury.py   # one-shot from a pipe

In-chat commands: /help /reset /undo /system /model /effort /temp /diffuse
                  /tokens /save /history /exit
"""

import argparse
import json
import os
import sys
import time

from bench_mercury import Conn, get_api_key, load_env_file, DEFAULT_BASE_URL

try:
    import readline  # noqa: F401  -- gives arrow-key editing and input history
except ImportError:
    pass

EFFORTS = ("instant", "low", "medium", "high")

# ANSI, disabled when not a tty so piped output stays clean.
if sys.stdout.isatty():
    DIM, BOLD, CYAN, YELLOW, RED, OFF = (
        "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m")
else:
    DIM = BOLD = CYAN = YELLOW = RED = OFF = ""


class Chat:
    def __init__(self, args, key):
        self.args = args
        self.key = key
        self.conn = Conn(args.base_url)
        self.messages = []
        self.system = args.system
        self.total_in = 0
        self.total_out = 0
        self.last_stats = None

    # ---------- networking ----------

    def _body(self):
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.extend(self.messages)
        body = {
            "model": self.args.model,
            "messages": msgs,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.args.diffusing:
            body["diffusing"] = True
        if self.args.effort:
            body["reasoning_effort"] = self.args.effort
        return body

    def ask(self, prompt):
        """Send one turn and stream the reply to stdout. Returns the reply text."""
        self.messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        ttft = None
        parts = []
        usage = None
        interrupted = False

        try:
            resp = self.conn.post("/chat/completions", self.key, self._body(), stream=True)
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
                            self.conn.reset()
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            if self.args.diffusing:
                                # Each chunk is a full snapshot of the sequence
                                # mid-denoise, so it replaces rather than extends.
                                # Intermediate states are partly noise -- show a
                                # progress marker and print only the final text.
                                parts = [piece]
                                sys.stdout.write(".")
                                sys.stdout.flush()
                            else:
                                parts.append(piece)
                                sys.stdout.write(piece)
                                sys.stdout.flush()
        except KeyboardInterrupt:
            interrupted = True
            self.conn.reset()  # socket has unread data; don't reuse it
            sys.stdout.write(f"\n{YELLOW}[interrupted]{OFF}")
        except Exception as e:
            self.conn.reset()
            self.messages.pop()  # roll back the user turn so history stays clean
            print(f"\n{RED}error: {e}{OFF}", file=sys.stderr)
            return None

        total = time.perf_counter() - t0
        text = "".join(parts)

        if self.args.diffusing and text:
            sys.stdout.write("\r" + " " * 40 + "\r" + text)
            sys.stdout.flush()

        if text:
            self.messages.append({"role": "assistant", "content": text})
        else:
            self.messages.pop()

        if usage:
            self.total_in += usage.get("prompt_tokens") or 0
            self.total_out += usage.get("completion_tokens") or 0

        out_tok = (usage or {}).get("completion_tokens")
        self.last_stats = {
            "ttft_s": ttft, "total_s": total, "out_tokens": out_tok,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "tps": (out_tok / total) if (out_tok and total) else None,
            "interrupted": interrupted,
        }

        if self.args.stats and text:
            s = self.last_stats
            bits = [f"{total:.2f}s"]
            if ttft is not None:
                bits.append(f"ttft {ttft * 1000:.0f}ms")
            if out_tok:
                bits.append(f"{out_tok} tok")
            if s["tps"]:
                bits.append(f"{s['tps']:.0f} tok/s")
            sys.stdout.write(f"\n{DIM}[{'  '.join(bits)}]{OFF}")
        return text

    # ---------- commands ----------

    def command(self, line):
        """Handle a /command. Returns False to quit, True otherwise."""
        cmd, _, arg = line[1:].partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        a = self.args

        if cmd in ("exit", "quit", "q"):
            return False

        elif cmd in ("help", "h", "?"):
            print(f"""{DIM}/reset          clear conversation history
/undo           drop the last exchange
/system [text]  show or set the system prompt (empty arg clears)
/model [name]   show or switch model (mercury-2, mercury-2.5)
/effort [lvl]   reasoning effort: {', '.join(EFFORTS)}
/temp [n]       sampling temperature (0.5-1.0)
/diffuse        toggle Mercury's diffusing stream mode
/tokens         session token usage
/history        print the conversation
/save [file]    write the transcript to JSON
/exit           quit{OFF}""")

        elif cmd == "reset":
            self.messages.clear()
            print(f"{DIM}history cleared{OFF}")

        elif cmd == "undo":
            dropped = 0
            while self.messages and dropped < 2:
                self.messages.pop()
                dropped += 1
            print(f"{DIM}dropped {dropped} message(s); {len(self.messages)} left{OFF}")

        elif cmd == "system":
            if arg:
                self.system = arg
                print(f"{DIM}system prompt set{OFF}")
            elif line.strip() == "/system":
                print(f"{DIM}{self.system or '(none)'}{OFF}")
            else:
                self.system = None
                print(f"{DIM}system prompt cleared{OFF}")

        elif cmd == "model":
            if arg:
                a.model = arg
                print(f"{DIM}model -> {arg}{OFF}")
            else:
                print(f"{DIM}{a.model}{OFF}")

        elif cmd == "effort":
            if not arg:
                print(f"{DIM}{a.effort or 'default'}{OFF}")
            elif arg in EFFORTS:
                a.effort = arg
                print(f"{DIM}effort -> {arg}{OFF}")
            else:
                print(f"{RED}pick one of: {', '.join(EFFORTS)}{OFF}")

        elif cmd == "temp":
            if not arg:
                print(f"{DIM}{a.temperature}{OFF}")
            else:
                try:
                    a.temperature = float(arg)
                    print(f"{DIM}temperature -> {a.temperature}{OFF}")
                except ValueError:
                    print(f"{RED}not a number: {arg}{OFF}")

        elif cmd == "diffuse":
            a.diffusing = not a.diffusing
            print(f"{DIM}diffusing -> {a.diffusing}{OFF}")

        elif cmd == "tokens":
            print(f"{DIM}session: {self.total_in} in / {self.total_out} out "
                  f"/ {self.total_in + self.total_out} total{OFF}")

        elif cmd == "history":
            if self.system:
                print(f"{DIM}system: {self.system}{OFF}")
            for m in self.messages:
                who = CYAN + "you" + OFF if m["role"] == "user" else BOLD + "mercury" + OFF
                print(f"{who}: {m['content']}")

        elif cmd == "save":
            path = arg or f"chat-{int(time.time())}.json"
            with open(path, "w") as fh:
                json.dump({"model": a.model, "system": self.system,
                           "messages": self.messages}, fh, indent=2)
            print(f"{DIM}saved {len(self.messages)} messages to {path}{OFF}")

        else:
            print(f"{RED}unknown command: /{cmd}{OFF}  {DIM}try /help{OFF}")

        return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    load_env_file(os.path.join(here, ".env"))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mercury-2")
    p.add_argument("--base-url", default=os.environ.get("MERCURY_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--system", help="system prompt")
    p.add_argument("--effort", choices=EFFORTS, help="reasoning effort")
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--diffusing", action="store_true", help="diffusing stream mode")
    p.add_argument("--no-stats", dest="stats", action="store_false", default=True,
                   help="hide the timing line after each reply")
    p.add_argument("prompt", nargs="*", help="one-shot prompt; omit for interactive")
    args = p.parse_args()

    key = get_api_key()
    chat = Chat(args, key)

    # One-shot: argv prompt or piped stdin.
    oneshot = " ".join(args.prompt).strip()
    if not oneshot and not sys.stdin.isatty():
        oneshot = sys.stdin.read().strip()
    if oneshot:
        chat.conn.warm()
        chat.ask(oneshot)
        print()
        chat.conn.reset()
        return

    chat.conn.warm()
    print(f"{BOLD}mercury{OFF} {DIM}({args.model}, effort={args.effort or 'default'})  "
          f"/help for commands, /exit to quit{OFF}\n")

    while True:
        try:
            line = input(f"{CYAN}you>{OFF} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.startswith("/"):
            if not chat.command(line):
                break
            continue

        sys.stdout.write(f"\n{BOLD}mercury>{OFF} ")
        sys.stdout.flush()
        chat.ask(line)
        print("\n")

    chat.conn.reset()


if __name__ == "__main__":
    main()
