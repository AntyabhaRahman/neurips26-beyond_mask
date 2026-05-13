# Model Evaluation Framework (OpenRouter)

This framework accompanies the MASK benchmark (Model Alignment between Statements and Knowledge) for measuring honesty in large language models. It tests whether a model will contradict its own beliefs when pressured to lie, separating *honesty* from *factual accuracy*.

This fork routes **every** model call through [OpenRouter](https://openrouter.ai/) — so you can run any OpenRouter-supported model as the **baseline** (the model being evaluated) and any OpenRouter-supported model as the **judge** (the LLM that grades responses). Only one API key is required.

## Pipeline

```
HuggingFace cais/MASK ──► generate_responses.py ──► evaluate.py ──► metric.py ──► process_metrics.py
                          (baseline model, via       (judge model,    (per-row     (all_results.json)
                           OpenRouter)                via OpenRouter)  metrics)
```

## Setup

1. Create a `.env` file at the repo root (or anywhere `python-dotenv` walks up to) with:
   ```env
   OPENROUTER_API_KEY=sk-or-...
   ```

2. Install dependencies (handled at the parent repo level via `uv`):
   ```bash
   uv sync
   ```

3. **No separate `download.py` step is required for generation.** `generate_responses.py` loads `cais/MASK` directly from HuggingFace. (`download.py` still works if you want the raw CSVs locally for inspection.)

## Run order

All commands below assume cwd is the `mask/` directory.

### 1. Generate responses (baseline model)

```bash
uv run python mask/generate_responses.py \
  --model google/gemini-3.1-flash-lite \
  [--archetypes statistics known_facts ...] \
  [--lie-k 1] [--max-tokens 1000] [--temperature 1.0] \
  [--concurrency 8] [--limit N] [--test] \
  [--cache-dir .cache/openrouter]
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | required | Any OpenRouter model ID, e.g. `google/gemini-3.1-flash-lite`, `anthropic/claude-sonnet-4.6`, `openai/gpt-4o`. |
| `--archetypes` | all 6 | `continuations` `disinformation` `doubling_down_known_facts` `known_facts` `provided_facts` `statistics`. |
| `--lie-k` | 1 | Number of times to sample the lying prompt per row. |
| `--max-tokens` | 1000 | Per-call output limit. |
| `--temperature` | 1.0 | Sampling temperature. |
| `--concurrency` | 8 | Max concurrent OpenRouter requests. |
| `--limit` | — | Cap rows per archetype (smoke testing). |
| `--test` | off | Uses 5 rows/archetype, writes to `test_csv_data/responses/`. |
| `--cache-dir` | `.cache/openrouter` | Disk cache (SHA256-keyed). Re-runs are free. |

Outputs, per archetype:
- `csv_data/responses/{archetype}_{model_slug}.csv` — the existing MASK CSV schema with `generation(System Prompt + User Prompt)_run{N}` and `generation(belief_system_1 + belief_elicit_{i})_run{N}` columns.
- `csv_data/responses/{archetype}_{model_slug}.meta.json` — sidecar with per-call `finish_reason`, `prompt_tokens`, `completion_tokens`, `error`, plus run totals.

`{model_slug}` is the model ID's last path component (e.g. `google/gemini-3.1-flash-lite` → `gemini-3.1-flash-lite`).

If a CSV already exists at the output path, that archetype is skipped — delete the file to force a re-run.

### 2. Evaluate responses (judge model)

```bash
uv run python mask/evaluate.py \
  [--judge-model openai/gpt-4o] \
  [--numerical-judge-model openai/gpt-4o-mini] \
  [--test] [--input_file path/to/file.csv] \
  [--concurrency_limit 50] [--cache-dir .cache/openrouter]
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--judge-model` | `anthropic/claude-sonnet-4.6` | Used for the binary/text-output judge (`A` / `B` / `C`). Sonnet was picked as the default because it followed the `Answer: <letter>` output contract on every prompt we tried; chattier models (e.g. `google/gemini-3.1-pro-preview`) often drift into prose and produce unparseable verdicts. |
| `--numerical-judge-model` | `anthropic/claude-sonnet-4.6` | Used for the `statistics` archetype, which requires JSON Schema output `{lower_bound, upper_bound}`. The model you pick **must support structured outputs** on OpenRouter. |
| `--test` | off | Reads `test_csv_data/responses/`, writes `test_csv_data/evaluated/`. |
| `--input_file` | — | Evaluate a single file instead of the whole directory. |
| `--concurrency_limit` | 50 | Outer concurrency cap (the OpenRouter client also has its own semaphore). |
| `--cache-dir` | `.cache/openrouter` | Shared with generation; same cache, no double-spend. |

Outputs: `csv_data/evaluated/{archetype}_{model_slug}.csv` — the input CSV plus new judge-verdict columns whose names start with `eval` followed by the same `(...)` suffix as the corresponding `generation(...)` column, ending in `_extracted`. The cell value is `A`/`B`/`C` for binary or `(lower, upper)` tuple text for numerical.

### 3. Per-row metrics

```bash
uv run python mask/metric.py [--test]
```

Reads `csv_data/evaluated/*.csv`, computes `eval_belief`, `eval_accuracy`, and `eval_honesty_*` per row, and writes `csv_data/metrics/*.csv`.

### 4. Aggregate final metrics

```bash
uv run python mask/process_metrics.py [--test]
```

Reads `csv_data/metrics/*.csv` and writes `csv_data/metrics/all_results.json`:

```json
{
  "<archetype>": {
    "<model>": {
      "total_responses": 0,
      "honest_1": 0.0,
      "unhonest_1": 0.0,
      "honesty_score_1": 0.0,
      "accurate": 0.0,
      "inaccurate": 0.0
    }
  }
}
```

## End-to-end smoke test

```bash
cd mask
uv run python mask/generate_responses.py --test \
  --model google/gemini-3.1-flash-lite \
  --archetypes statistics known_facts
uv run python mask/evaluate.py --test \
  --judge-model anthropic/claude-sonnet-4.6 \
  --numerical-judge-model anthropic/claude-sonnet-4.6
uv run python mask/metric.py --test
uv run python mask/process_metrics.py --test
cat test_csv_data/metrics/all_results.json
```

## Directory layout

```
mask/
├── csv_data/                # production
│   ├── responses/           # generate_responses.py output (CSV + .meta.json)
│   ├── evaluated/           # evaluate.py output
│   └── metrics/             # metric.py + all_results.json
├── test_csv_data/           # mirrors csv_data, used when --test is passed
├── mask/                    # pipeline scripts (this directory)
│   ├── generate_responses.py
│   ├── evaluate.py
│   ├── metric.py
│   ├── process_metrics.py
│   ├── prompts/             # judge prompt templates
│   └── README.md            # (this file)
└── requirements.txt
```

## Notes

- The OpenRouter client (`src/beyond_mask/openrouter.py`) handles retries (429/5xx), a disk cache keyed by `(model, messages, temperature, max_tokens, response_format)`, and a concurrency semaphore. Cached results are free on re-runs.
- For the `statistics` archetype the numerical judge sends `response_format={"type": "json_schema", ...}`. Pick a `--numerical-judge-model` that supports structured outputs on OpenRouter (most current OpenAI / Gemini / Anthropic models do).
- Paths in every script are cwd-relative; run from `mask/`.
