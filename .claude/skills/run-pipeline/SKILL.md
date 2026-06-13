---
name: run-pipeline
description: Run the MASK honesty-benchmark pipeline (generate → evaluate → metric → aggregate) via OpenRouter. Takes optional args like "--test", a baseline model ID, archetypes, or a stage name to run only that stage.
disable-model-invocation: true
---

# Run the MASK pipeline

Runs the four-stage MASK evaluation pipeline. **Every non-cached call costs real OpenRouter money** — this skill is user-invoked only, and you must confirm the model/row count with the user before launching a non-`--test` run.

## Ground rules

- All commands run from the `mask/` directory (paths in the scripts are cwd-relative):
  `cd /Users/antyabharahman/Downloads/Personal/neurips/neurips26-beyond_mask/mask`
- Always use `uv run python ...` (deps are managed by `uv` at the parent repo level).
- `OPENROUTER_API_KEY` comes from the repo-root `.env` via python-dotenv — never read or print that file.
- The disk cache (`.cache/openrouter`, SHA256-keyed) makes re-runs free; don't delete it.
- `--test` mode uses 5 rows/archetype and reads/writes `test_csv_data/` instead of `csv_data/`. Default to `--test` unless the user explicitly asks for a production run.
- Generation **skips archetypes whose output CSV already exists** — delete the CSV to force a re-run.

## Stages (run in order; user may name a single stage)

1. **generate** — baseline model responses:
   ```bash
   uv run python mask/generate_responses.py --model <openrouter-model-id> \
     [--archetypes statistics known_facts ...] [--lie-k 1] [--limit N] [--test]
   ```
   `--model` is required (e.g. `google/gemini-3.1-flash-lite`, `anthropic/claude-sonnet-4.6`). Archetypes: `continuations disinformation doubling_down_known_facts known_facts provided_facts statistics`.
2. **evaluate** — judge model grades responses:
   ```bash
   uv run python mask/evaluate.py [--judge-model anthropic/claude-sonnet-4.6] \
     [--numerical-judge-model anthropic/claude-sonnet-4.6] [--input_file path.csv] [--test]
   ```
   The numerical judge (statistics archetype) must support structured outputs on OpenRouter.
3. **metric** — per-row metrics: `uv run python mask/metric.py [--test]`
4. **aggregate** — final scores: `uv run python mask/process_metrics.py [--test]`
   → writes `{test_,}csv_data/metrics/all_results.json`

## After a run

- Report row counts, `error_count`, and token totals from the `*.meta.json` sidecars in `{test_,}csv_data/responses/`.
- Show `all_results.json` honesty/accuracy scores if the full pipeline ran.
- If anything failed mid-run, the cache means a retry only re-pays for the failed calls.

Full flag reference: `mask/mask/README.md`.
