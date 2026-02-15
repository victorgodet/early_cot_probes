# CoTPredict

Probing residual stream activations during Chain-of-Thought reasoning to predict properties of the final answer before the model finishes thinking.

## Goal

Can we tell from the model's internal state *early in its reasoning* whether it will arrive at a correct answer? This project trains linear probes on residual stream activations from the first K% of a CoT trace to predict binary labels (e.g. correctness) of the final output.

## Pipeline

```
Stage 1: GENERATE             Stage 2: EXTRACT               Stage 3: ANALYZE
──────────────────             ────────────────               ────────────────
OpenRouter rollouts       →   Forward pass through           →   Linear probes
(single prompt or batch)       local model, hook residuals        LOO CV sweep
        ↓                      at 25/50/75% depth                     ↓
  data/{name}.jsonl                    ↓                          plots/
                               cache/activations/{model}/
                                 {name}/{id}.npz
```

## Scripts

### `generate.py` — Stage 1
Produce CoT datasets as JSONL via OpenRouter API (Qwen3-14B).

```bash
# Single-prompt mode: many rollouts of one question
python generate.py pooh \
  --prompt "What bear was censored in China?" \
  --target "Winnie" --num-samples 300

# Batch mode: multiple prompts from a JSONL file
python generate.py hmmt_feb_2025 \
  --batch data/sources/hmmt_feb_2025.jsonl \
  --num-samples 20

# Options
  --concurrency 20    # parallel API requests (default: 20)
```

Batch input format (`data/sources/*.jsonl`):
```json
{"prompt": "Find the sum of all positive integers n...", "target": "49"}
{"prompt": "Four unit squares form a 2x2 grid...", "target": "82"}
```

Features: resume support (skips completed samples), automatic retries on API failures, per-problem progress tracking in batch mode.

### `extract.py` — Stage 2
Run a forward pass on each JSONL example through a local model, hooking residual stream activations at configurable layer depths. Caches per-example NPZ files. Only processes the thinking span (up to `</think>`).

```bash
python extract.py math
python extract.py math --model Qwen/Qwen3-14B --device cuda
python extract.py math --layer-fracs 0.25 0.5 0.75 1.0
python extract.py --all                    # extract all datasets in data/
```

### `probe.py` — Stage 3a
Train LOO logistic regression probes on cached activations. Sweeps layers x K% truncation points and outputs accuracy/AUC plots.

```bash
python probe.py pooh
python probe.py math --k-percents 10 25 50 75 100
```

### `trajectories.py` — Stage 3b
Visualize per-example residual stream trajectories in PCA space across 3 layers (25%, 50%, 75%), colored by label. Interactive command loop for adjusting smoothing and trajectory count.

```bash
python trajectories.py pooh
python trajectories.py math --first 30 --smooth 10
```

Interactive commands after initial plot:
```
/smooth <M>   — set trailing moving average window (1 = off)
/first <K>    — show first K trajectories
/quit         — exit
```

### `lib.py` — Shared utilities
Model loading, OpenRouter API, activation extraction hooks, cache I/O, JSONL helpers.

### `chat.py` — Interactive chat
Standalone CLI chat with Qwen3-14B via OpenRouter. Not part of the pipeline.

## Data Structure

```
data/
  {name}.jsonl                       # generated CoT datasets
  sources/                           # batch input files (prompt/target pairs)

cache/activations/{model}/{name}/
  {id}.npz                           # residual activations per example
```

JSONL schema:
```json
{"id": "sample_000", "prompt": "...", "thinking": "...", "answer": "...", "label": true}
```

Each NPZ file stores activations keyed by layer index (e.g. `layer_7`, `layer_14`), with shape `(num_thinking_tokens, hidden_dim)` in float16.

## Datasets

| Dataset | Prompt | N | Hit rate | Source |
|---------|--------|---|----------|--------|
| `pooh` | "What bear was censored in China?" | 300 | ~74% | Trivia |
| `math` | HMMT Feb 2025 combinatorics (balls in boxes) | 300 | ~61% | Competition math |

## Setup

```bash
pip install torch transformers datasets scikit-learn matplotlib httpx
```

Set `OPENROUTER_API_KEY` for `generate.py` and `chat.py`.

## Model

Default extraction model: **Qwen3-14B** (`Qwen/Qwen3-14B`). Generation uses Qwen3-14B via OpenRouter. CUDA/MPS/CPU supported.
