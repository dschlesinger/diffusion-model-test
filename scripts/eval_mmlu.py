#!/usr/bin/env python3
"""Evaluate DiffusionGemma on MMLU (generative, answer-extraction scoring).

Drives llama.cpp's llama-diffusion-gemma-visual-server (PR #24423), which loads the
GGUF once and applies the model's own chat template, so each question costs only
generation time. Results are appended to a JSONL file as they finish; rerunning with
the same --out resumes where it left off.

Usage (on a GPU node, see docs/diffusiongemma-scc.md):
  python3 scripts/eval_mmlu.py --model /path/to/model-Q8_0.gguf --limit 50
  python3 scripts/eval_mmlu.py --model ... --per-subject 20 --out mmlu_q8.jsonl
  python3 scripts/eval_mmlu.py --summarize mmlu_q8.jsonl

Requires: pip install --user datasets
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

LETTERS = "ABCD"
SERVER_BIN = "llama-diffusion-gemma-visual-server"


# ---------------------------------------------------------------- data

def load_mmlu(subjects=None):
    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    wanted = set(subjects) if subjects else None
    rows = []
    counters = defaultdict(int)
    for r in test:
        s = r["subject"]
        idx = counters[s]
        counters[s] += 1
        if wanted is None or s in wanted:
            rows.append({"id": f"{s}/{idx}", "subject": s, "question": r["question"],
                         "choices": r["choices"], "answer": LETTERS[r["answer"]]})
    shots = defaultdict(list)
    for r in dev:
        shots[r["subject"]].append(r)
    return rows, shots


def format_question(question, choices):
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def build_prompt(row, shots, n_shots):
    subject = row["subject"].replace("_", " ")
    parts = [f"The following are multiple choice questions about {subject}. "
             "Reason if you need to, then end your reply with a line of the form "
             "'Answer: X', where X is A, B, C, or D."]
    for ex in shots[row["subject"]][:n_shots]:
        parts.append(format_question(ex["question"], ex["choices"])
                     + f"\nAnswer: {LETTERS[ex['answer']]}")
    parts.append(format_question(row["question"], row["choices"]))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- answer extraction

THOUGHT_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
SPECIAL_RE = re.compile(r"<\|[^<>]*>|<[^<>]*\|>")
ANSWER_RES = [
    re.compile(r"answer\s*(?:is|:)?\s*[*_]*\s*\(?([ABCD])\b", re.IGNORECASE),
    re.compile(r"^\s*[*_]*\(?([ABCD])[)\.:]?[*_]*\s*$", re.MULTILINE),
    re.compile(r"\(([ABCD])\)"),
]


def split_reply(text):
    """Return (final_text, had_thought, truncated) from the raw server text (special tokens kept)."""
    had_thought = "<|channel>" in text
    final = THOUGHT_RE.sub("", text)
    truncated = "<|channel>" in final  # thought block opened but never closed
    if truncated:
        final = final.split("<|channel>", 1)[0]
    final = SPECIAL_RE.sub("", final).strip()
    return final, had_thought, truncated


def extract_answer(final):
    for pat in ANSWER_RES:
        matches = pat.findall(final)
        if matches:
            return matches[-1].upper()
    return None


# ---------------------------------------------------------------- server

class DiffusionServer:
    def __init__(self, binary, model, ngl, maxtok, log_path):
        env = dict(os.environ, NGL=str(ngl))
        if maxtok:
            env["MAXTOK"] = str(maxtok)
        self.log = open(log_path, "a")
        self.tmp = tempfile.mkdtemp(prefix="mmlu_req_")
        self.proc = subprocess.Popen([binary, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1, env=env)
        line = self._readline()
        while not line.startswith("READY"):
            line = self._readline()
        _, n_vocab, maxtok = line.split()
        print(f"server ready: n_vocab={n_vocab} MAXTOK={maxtok} (log: {log_path})", file=sys.stderr)

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"server exited (code {self.proc.poll()}); see {self.log.name}")
        return line.rstrip("\n")

    def generate(self, messages, n_blocks, seed):
        path = os.path.join(self.tmp, "req.json")
        with open(path, "w") as f:
            json.dump({"seed": seed, "n_blocks": n_blocks, "messages": messages}, f)
        self.proc.stdin.write(path + "\n")
        self.proc.stdin.flush()

        text, stats, error = "", {}, None
        while True:
            line = self._readline()
            if line == "DONE":
                break
            if line.startswith("C "):
                text = json.loads(line.split(" ", 2)[2])
            elif line.startswith("STATS "):
                for kv in line.split()[1:]:
                    k, v = kv.split("=", 1)
                    stats[k] = float(v) if "." in v else int(v)
            elif line.startswith("ERR "):
                error = line[4:]
                # "toolong" and "gen" are followed by DONE; any other ERR ends the request
                if not (error.startswith("toolong") or error.startswith("gen")):
                    break
            # "F ..." per-step frames are ignored
        return text, stats, error

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.log.close()


# ---------------------------------------------------------------- reporting

def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        print("no results yet")
        return
    by_subj = defaultdict(list)
    for r in recs:
        by_subj[r["subject"]].append(r["correct"])
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / n
    macro = sum(sum(v) / len(v) for v in by_subj.values()) / len(by_subj)
    no_ans = sum(r["pred"] is None for r in recs)
    trunc = sum(r["truncated"] for r in recs)
    thought = sum(r["had_thought"] for r in recs)
    errs = sum(r["error"] is not None for r in recs)
    gen_tok = sum(r["stats"].get("predicted_n", 0) for r in recs)
    dec_ms = sum(r["stats"].get("decode_ms", 0) for r in recs)
    steps = sum(r["stats"].get("steps", 0) for r in recs)

    print(f"results:          {path}")
    print(f"questions:        {n} across {len(by_subj)} subjects")
    print(f"accuracy (micro): {acc:.4f}")
    print(f"accuracy (macro): {macro:.4f}   (mean of per-subject accuracy)")
    print(f"no answer found:  {no_ans} ({no_ans / n:.1%})   truncated thinking: {trunc}   server errors: {errs}")
    print(f"used thinking:    {thought} ({thought / n:.1%})")
    if dec_ms:
        print(f"generation:       {gen_tok / n:.0f} tok/question avg, {gen_tok / (dec_ms / 1000):.1f} tok/s, "
              f"{gen_tok / max(steps, 1):.1f} tok/step")
    print("\nper subject (worst first):")
    for s, v in sorted(by_subj.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {sum(v) / len(v):.3f}  {len(v):4d}  {s}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to the DiffusionGemma GGUF")
    ap.add_argument("--server", default=shutil.which(SERVER_BIN) or SERVER_BIN,
                    help=f"path to {SERVER_BIN} (default: from PATH)")
    ap.add_argument("--out", default="mmlu_results.jsonl", help="JSONL results file (resumable)")
    ap.add_argument("--subjects", nargs="+", help="only these subjects (e.g. college_physics virology)")
    ap.add_argument("--per-subject", type=int, help="first N questions of each subject")
    ap.add_argument("--limit", type=int, help="random sample of N questions overall (after --per-subject)")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples from the dev split (0-5)")
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--n-blocks", type=int, default=8,
                    help="max 256-token blocks per answer, thinking included (default 8 = 2048 tok)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ngl", type=int, default=99, help="layers on GPU")
    ap.add_argument("--maxtok", type=int, default=0, help="server context budget (0 = auto-size to VRAM)")
    ap.add_argument("--summarize", metavar="JSONL", help="only print the summary of an existing results file")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.model:
        ap.error("--model is required")

    rows, shots = load_mmlu(args.subjects)
    if args.per_subject:
        seen = defaultdict(int)
        kept = []
        for r in rows:
            if seen[r["subject"]] < args.per_subject:
                kept.append(r)
                seen[r["subject"]] += 1
        rows = kept
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} questions selected, {len(done & {r['id'] for r in rows})} already done, "
          f"{len(todo)} to run", file=sys.stderr)

    if todo:
        server = DiffusionServer(args.server, args.model, args.ngl, args.maxtok,
                                 os.path.splitext(args.out)[0] + ".server.log")
        t0 = time.time()
        n_correct = 0
        try:
            with open(args.out, "a") as out:
                for i, row in enumerate(todo, 1):
                    messages = []
                    if args.system:
                        messages.append({"role": "system", "content": args.system})
                    messages.append({"role": "user", "content": build_prompt(row, shots, args.shots)})

                    raw, stats, error = server.generate(messages, args.n_blocks, args.seed)
                    final, had_thought, truncated = split_reply(raw)
                    pred = extract_answer(final)
                    correct = pred == row["answer"]
                    n_correct += correct

                    out.write(json.dumps({
                        "id": row["id"], "subject": row["subject"], "gold": row["answer"], "pred": pred,
                        "correct": correct, "had_thought": had_thought, "truncated": truncated,
                        "error": error, "stats": stats, "final": final, "raw": raw,
                    }) + "\n")
                    out.flush()

                    rate = i / (time.time() - t0)
                    print(f"[{i}/{len(todo)}] {row['id']:<45} gold={row['answer']} pred={pred} "
                          f"{'ok ' if correct else 'BAD'} run_acc={n_correct / i:.3f} "
                          f"eta={(len(todo) - i) / rate / 60:.0f}m", file=sys.stderr)
        finally:
            server.close()

    print()
    summarize(args.out)


if __name__ == "__main__":
    main()
