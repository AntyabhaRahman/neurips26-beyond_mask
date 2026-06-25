---
name: summarize-results
description: Summarize MASK/beyond_mask experiment runs — API spend, token usage, error counts, and honesty/accuracy scores — from the OpenRouter cache, meta.json sidecars, results/ parquet runs, and all_results.json.
---

# Summarize experiment results

Aggregate cost and outcome data for this project's eval runs. All paths below are relative to the repo root (`/Users/antyabharahman/Downloads/Personal/neurips/neurips26-beyond_mask`).

## Data sources

1. **OpenRouter call cache** — `mask/.cache/openrouter/*.json`, one file per API call with `cost_usd`, `prompt_tokens`, `completion_tokens`, `error`. Total historical spend:
   ```bash
   jq -s '{calls: length, cost_usd: (map(.cost_usd // 0) | add), prompt_tokens: (map(.prompt_tokens // 0) | add), completion_tokens: (map(.completion_tokens // 0) | add), errors: (map(select(.error != null)) | length)}' mask/.cache/openrouter/*.json
   ```
   (If the arg list is too long, use `find ... -name '*.json' -exec cat {} +` piped to `jq -s`.)
2. **Per-archetype run sidecars** — `mask/{test_,}csv_data/responses/*.meta.json` with `model`, `archetype`, `row_count`, `calls`, `error_count`, `total_prompt_tokens`, `total_completion_tokens`, `lie_k`, `temperature`.
3. **Final scores** — `mask/{test_,}csv_data/metrics/all_results.json`: per archetype × model, `honesty_score_1`, `accurate`, `total_responses`.
4. **Playground runs** — `results/<dataset_slug>/<timestamp>__<model>__nN.parquet` + matching `.json` metadata (written by `src/beyond_mask/results.py:write_run`). Read parquet with polars: `uv run python -c "import polars as pl; ..."`.

## Output format

A short prose summary plus one table per relevant view:
- **Spend**: total calls, total cost (USD, 4 decimals), tokens in/out, error count.
- **Runs**: model × archetype with row_count and error_count (flag any nonzero errors prominently).
- **Scores**: honesty_score_1 and accuracy per model/archetype when all_results.json exists.

Never open the repo-root `.env`. If the user asks for cost "per model", note that the call cache is keyed by request hash, not model — group via the `raw` field's model id or fall back to the meta.json sidecars.
