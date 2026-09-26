# Running DiffusionGemma (Gemma 4 diffusion) on BU SCC

This guide runs Google's **DiffusionGemma** (`diffusiongemma-26B-A4B-it`) on the SCC. It uses Unsloth's GGUF quants and llama.cpp's `llama-diffusion-cli`.

DiffusionGemma is a Gemma 4 mixture-of-experts model. It has 25.2B total parameters, of which 3.8B are active per token. Instead of writing text left to right, it writes a 256-token block ("canvas") at a time and fills in many tokens in parallel at each step.

---

## 1. Pick a GPU and a quant

| Quant (unsloth/diffusiongemma-26B-A4B-it-GGUF) | File size | Fits on |
|---|---|---|
| `Q4_K_M` | ~18 GB | A40, A6000, L40, L40S, RTX6000ada (48 GB) |
| `Q8_0` | ~27 GB | Same 48 GB cards, with room left for context |
| bf16 (original) | ~52 GB | A100-80G, H200 only |

**Recommended: `Q8_0` on an L40S.** The SCC has the most L40S cards and they are usually free. The rest of this guide uses that setup.

CUDA architecture codes for the build (every card you want to run on must be listed):

| Arch | GPUs |
|---|---|
| `80` | A100, A100-80G |
| `86` | A40, A6000 |
| `89` | L40S, L40, RTX6000ada |
| `90` | H200 |

## 2. Storage

- **`/projectnb/<project>`**: large and not backed up. Put the model weights (27 GB) here.
- **`/project/<project>`**: backed up, small quota. Fine for the llama.cpp source and build.
- **`$HOME`**: tiny. Keep the Hugging Face cache out of it (see step 4).

The examples below use project `buaisociety`. Replace it with your own.

## 3. Build `llama-diffusion-cli`

DiffusionGemma support has not been merged into llama.cpp yet. It lives in **PR #24423**. The build job in this repo checks out that PR and builds only the diffusion runner.

> Don't build on a login node (`scc1`, `scc2`). The build runs out of memory and gets `Killed`, and login nodes stop heavy processes anyway. Submit it as a batch job.

```bash
cd /project/buaisociety/diffusion-model-test
qsub -P buaisociety -v CLEAN=1,CUDA_MODULE=cuda/12.5 scripts/build_llamacpp.qsub
qstat -u $USER                  # wait for it to finish
tail -f build_llamacpp.log      # should end with "done: .../llama-diffusion-cli"
```

Options (passed comma-separated to `-v`):

- `CUDA_ARCH="86;89"` builds for more GPU types. The default is `89`. Building for fewer architectures is faster.
- `CLEAN=1` deletes `build/` first. You need it on the first build and whenever you change `CUDA_ARCH`.
- `LLAMA_DIR=...` sets the checkout location. The default is `./llama.cpp`.

The job uses 8 cores and 32 GB of RAM, and takes roughly 10–20 minutes.

### Put it on your PATH

Add this to `~/.bashrc`:

```bash
export PATH=/project/buaisociety/diffusion-model-test/llama.cpp/build/bin:$PATH
```

or symlink it:

```bash
mkdir -p ~/bin
ln -s /project/buaisociety/diffusion-model-test/llama.cpp/build/bin/llama-diffusion-cli ~/bin/
```

## 4. Download the model

```bash
module load python3
pip install --user -U huggingface_hub

export HF_HOME=/projectnb/buaisociety/$USER/hf-cache    # keep the cache out of $HOME
mkdir -p /projectnb/buaisociety/$USER/models

hf download unsloth/diffusiongemma-26B-A4B-it-GGUF \
    --include "*Q8_0*" \
    --local-dir /projectnb/buaisociety/$USER/models/diffusiongemma
```

The download is about 27 GB. If it's slow or gets killed on the login node, run it inside a `qrsh` session.

## 5. Get a GPU node

The binary needs `libcuda.so.1`, which only exists on GPU nodes. On a login node you'll see:

```
error while loading shared libraries: libcuda.so.1: cannot open shared object file
```

Request an interactive L40S:

```bash
qrsh -P buaisociety -l gpus=1 -l gpu_type=L40S -pe omp 4 -l h_rt=02:00:00
```

Then, on the node:

```bash
module load gcc cuda/12.5
nvidia-smi    # confirm you have the GPU
```

## 6. Run it

```bash
llama-diffusion-cli \
    -m /projectnb/buaisociety/$USER/models/diffusiongemma/<file>-Q8_0.gguf \
    -ngl 99 -cnv \
    -n 4096 -c 8192 -b 8192 -ub 8192 \
    --diffusion-visual
```

| Flag | Meaning |
|---|---|
| `-ngl 99` | Put every layer on the GPU |
| `-cnv` | Interactive chat mode |
| `-n` | Maximum tokens to generate per reply. This includes the model's thinking. |
| `-c` | Context size. Must be at least the conversation so far plus `-n`. |
| `-b`, `-ub` | Batch and micro-batch size. **Must be at least prompt + 256.** See below. |
| `--diffusion-visual` | Shows the canvas being filled in, like the Google demos. Leave it off for clean output. |

**Why `-ub` has to be large:** at every denoising step, the model re-runs the whole `[prompt | 256-token canvas]` in a single micro-batch. If `-ub` is smaller than that, you get:

```
this diffusion model needs the whole [prompt | canvas] in one ubatch;
set -ub and -c >= n_input + canvas_length
```

Setting `-c`, `-b`, and `-ub` all to the same value is the simplest fix. Use 4096 for short chats and 8192 for long ones.

## 7. Reading the stats

Example output:

```
total time: 17208.44ms, time per step: 77.87ms (221 steps over 6 blocks, entropy-bound)
throughput: 89.3 tok/s (1536 tok in 17208.44ms), in-step parallel 3288 tok/s
```

- **blocks**: how many 256-token canvases were generated (6 × 256 = 1536 tokens).
- **steps**: how many denoising passes it took. The number of steps per block varies ("entropy-bound"). Easy text locks in many tokens per step, and hard text needs more steps. Here it averaged about 37 steps per block, or about 7 tokens per step.
- **throughput**: real speed as a user sees it (tokens ÷ wall time). **This is the number to compare** against autoregressive models.
- **in-step parallel**: 256 ÷ time per step. This is the rate at which one step processes the canvas, not how fast finished tokens come out.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Build ends with `Killed` | You built on a login node, or with too many parallel jobs. Use `scripts/build_llamacpp.qsub`. |
| `libcuda.so.1: cannot open shared object file` | You're on a login node. Use `qrsh` to get a GPU node (step 5). |
| `needs the whole [prompt \| canvas] in one ubatch` | Raise `-ub`, `-b`, and `-c` (step 6). |
| Reply stops mid-sentence after exactly 256 tokens / 1 block | `-n` is too small or missing. Set `-n 2048` or higher. Thinking tokens count against `-n`, so the answer may not have started yet. |
| `no kernel image is available for execution on the device` | The binary wasn't built for this GPU's architecture. Rebuild with `CUDA_ARCH` including it, plus `CLEAN=1`. |
| Out of GPU memory | Use `Q4_K_M`, or lower `-c`/`-ub`. |

## 9. Evaluating on MMLU

`scripts/eval_mmlu.py` runs MMLU through `llama-diffusion-gemma-visual-server`. That server is built by the same qsub job and loads the model once for the whole run. Each question is answered as generated text, including any thinking, and the script pulls the letter out of an `Answer: X` line.

```bash
pip install --user datasets    # one time, with `module load python3`

# quick smoke test on an interactive GPU node
python3 scripts/eval_mmlu.py --model <gguf> --limit 20

# batch job: 20 questions per subject (1,140 total), results resumable in mmlu_q8.jsonl
qsub -P buaisociety -v MODEL=<gguf>,OUT=mmlu_q8.jsonl,EVAL_ARGS="--per-subject 20" scripts/eval_mmlu.qsub

# summary of a results file (micro/macro accuracy, parse failures, tok/s, per-subject)
python3 scripts/eval_mmlu.py --summarize mmlu_q8.jsonl
```

Useful flags: `--shots 5` (the standard 5-shot setup, using the dev split), `--n-blocks` (the cap on tokens per answer, in 256-token blocks, default 8), `--subjects`, and `--system`. Every result row stores the raw output, so you can check why an answer was marked wrong.

## 10. Calling it over HTTP

`llama-diffusion-gemma-visual-server` only speaks stdin/stdout (no socket), so nothing off the node can reach it directly. `scripts/diffusion_http_server.py` wraps one instance of it and exposes an OpenAI-shaped `/v1/chat/completions` endpoint over HTTP.

On the GPU node:

```bash
python3 scripts/diffusion_http_server.py --model <gguf> --host 0.0.0.0 --port 8000
```

SCC compute nodes aren't reachable from off-campus directly, so tunnel through the login node from your own machine:

```bash
ssh -L 8000:<compute-node-hostname>:8000 <user>@scc1.bu.edu
```

Then, from your own computer:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Explain diffusion models in 3 sentences."}]}'
```

Response shape:

```json
{
  "id": "diffusiongemma-...",
  "object": "chat.completion",
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "diffusion": {"had_thought": false, "truncated": false, "error": null, "stats": {...}, "wall_seconds": 4.2}
}
```

`choices[0].message.content` is the final answer (thinking stripped, same as `eval_mmlu.py`'s scoring path). `diffusion.stats` carries the block/step/tok-per-second numbers from step 7. The server handles one request at a time — it's one subprocess behind a lock — so concurrent callers queue rather than run in parallel. There's no streaming yet: each call blocks until the full reply is generated.

Body fields: `messages` (required, OpenAI chat format), `n_blocks` (optional, default 8), `seed` (optional, default 0).
