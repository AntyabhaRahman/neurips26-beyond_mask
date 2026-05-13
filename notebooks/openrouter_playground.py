import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import os
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from datasets import load_dataset

    from beyond_mask.openrouter import ChatResult, OpenRouterClient
    from beyond_mask.results import (
        dataset_slug as make_dataset_slug,
        git_sha,
        model_slug as make_model_slug,
        run_paths,
        write_run,
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_ROOT = PROJECT_ROOT / "results"
    CACHE_ROOT = PROJECT_ROOT / ".cache" / "openrouter"
    return (
        CACHE_ROOT,
        OpenRouterClient,
        RESULTS_ROOT,
        asyncio,
        datetime,
        git_sha,
        load_dataset,
        make_dataset_slug,
        make_model_slug,
        mo,
        os,
        pl,
        run_paths,
        write_run,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OpenRouter playground

    Pick a Hugging Face dataset, a list of OpenRouter model ids, and an `N`.
    Click **Run** to fan out N rows × M models. Results land in
    `results/<dataset_slug>/<timestamp>__<model>__nN.parquet` with a JSON
    sidecar of run metadata. Identical calls are cached on disk.

    Requires `OPENROUTER_API_KEY` in your environment.
    """)
    return


@app.cell
def _(mo, os):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    mo.stop(
        not api_key,
        mo.md("**`OPENROUTER_API_KEY` not set.** Export it in your shell and re-run."),
    )
    return (api_key,)


@app.cell
def _():
    MODELS = [
        "google/gemini-3.1-flash-lite"
    ]
    return (MODELS,)


@app.cell
def _(load_dataset):
    splits = ["continuations", "disinformation", "doubling_down_known_facts", "known_facts", "provided_facts", "statistics"]
    _raw = load_dataset("cais/MASK", splits[2], split="test")
    _raw_df = _raw.to_polars()
    _raw_df.drop("task_id", "canary").limit(10)
    return


@app.cell
def _(load_dataset, make_dataset_slug, mo):
    DATASET_ID = "cais/MASK"
    # Available configs for cais/MASK: 
    #   continuations, disinformation, doubling_down_known_facts,
    #   known_facts, provided_facts, statistics
    CONFIG: str | None = "statistics"
    SPLIT = "test"

    ds = load_dataset(DATASET_ID, CONFIG, split=SPLIT)
    ds_slug = make_dataset_slug(DATASET_ID, CONFIG, SPLIT)

    mo.md(
        f"""
        **Dataset:** `{DATASET_ID}` config=`{CONFIG}` split=`{SPLIT}`<br>
        **Rows:** {len(ds)} &nbsp; **Slug:** `{ds_slug}`<br>
        **First row keys:** `{list(ds[0].keys())}`
        """
    )
    return CONFIG, DATASET_ID, SPLIT, ds, ds_slug


@app.cell
def _(mo):
    n = mo.ui.slider(start=1, stop=500, value=10, step=1, label="N rows")
    n
    return (n,)


@app.function
def make_prompt(row: dict) -> list[dict]:
    """Build chat messages for one MASK row (pressure pass only)."""
    return [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["user_prompt"]},
    ]


@app.cell
def _():
    TEMPERATURE = 0.0
    MAX_TOKENS = 256
    return MAX_TOKENS, TEMPERATURE


@app.cell
def _(mo):
    use_cache = mo.ui.checkbox(value=True, label="Use disk cache")
    use_cache
    return (use_cache,)


@app.cell
def _(ds, mo, n):
    preview_n = min(3, n.value, len(ds))
    previews = []
    for i in range(preview_n):
        msgs = make_prompt(ds[i])
        rendered = "\n\n".join(f"**{m['role']}**:\n\n{m['content']}" for m in msgs)
        previews.append(mo.md(f"### row {i}\n\n{rendered}"))
    mo.vstack([mo.md("## Prompt preview (no API calls)"), *previews])
    return


@app.cell
def _(MODELS, mo, n):
    run = mo.ui.run_button(
        label=f"Run {n.value} rows × {len(MODELS)} models",
        kind="success",
    )
    run
    return (run,)


@app.cell
async def _(
    CACHE_ROOT,
    CONFIG: str | None,
    DATASET_ID,
    MAX_TOKENS,
    MODELS,
    OpenRouterClient,
    RESULTS_ROOT,
    SPLIT,
    TEMPERATURE,
    api_key,
    asyncio,
    datetime,
    ds,
    ds_slug,
    git_sha,
    make_model_slug,
    mo,
    n,
    pl,
    run,
    run_paths,
    use_cache,
    write_run,
):
    mo.stop(not run.value, mo.md("_Click **Run** to launch the sweep._"))

    sample_n = min(n.value, len(ds))
    sample = [(i, ds[i]) for i in range(sample_n)]
    cache_dir = CACHE_ROOT if use_cache.value else None
    started_at = datetime.now()
    sha = git_sha()

    async def run_one_model(client: OpenRouterClient, model: str):
        tasks = [
            client.chat(
                model=model,
                messages=make_prompt(row),
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                cache_dir=cache_dir,
            )
            for _, row in sample
        ]
        return await asyncio.gather(*tasks)

    sections: list = [mo.md(f"## Run @ {started_at.isoformat(timespec='seconds')}")]
    written_paths: list = []

    async with OpenRouterClient(api_key) as client:
        for model in MODELS:
            results = await run_one_model(client, model)

            rows = []
            for (dataset_row_idx, row), result in zip(sample, results, strict=True):
                rows.append(
                    {
                        "row_idx": dataset_row_idx,
                        "dataset_row_idx": dataset_row_idx,
                        "model": model,
                        "messages": make_prompt(row),
                        "response": result.text,
                        "finish_reason": result.finish_reason,
                        "latency_ms": result.latency_ms,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "cost_usd": result.cost_usd,
                        "cached": result.cached,
                        "error": result.error,
                    }
                )
            df = pl.DataFrame(rows)

            parquet_path, json_path = run_paths(
                RESULTS_ROOT, ds_slug, make_model_slug(model), sample_n, started_at
            )
            metadata = {
                "dataset_id": DATASET_ID,
                "config": CONFIG,
                "split": SPLIT,
                "model": model,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "n": sample_n,
                "total_cost_usd": float(
                    sum(r.cost_usd or 0.0 for r in results)
                ),
                "total_latency_ms": int(sum(r.latency_ms for r in results)),
                "n_cached": int(sum(1 for r in results if r.cached)),
                "n_errors": int(sum(1 for r in results if r.error)),
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now().isoformat(),
                "git_sha": sha,
            }
            write_run(df, metadata, parquet_path, json_path)
            written_paths.append(parquet_path)

            sections.append(
                mo.md(
                    f"### `{model}` &nbsp; "
                    f"cost=${metadata['total_cost_usd']:.5f} &nbsp; "
                    f"errors={metadata['n_errors']} &nbsp; "
                    f"cached={metadata['n_cached']}/{sample_n}<br>"
                    f"→ `{parquet_path.relative_to(RESULTS_ROOT.parent)}`"
                )
            )
            sections.append(df)

    mo.vstack(sections)
    return (written_paths,)


@app.cell
def _(mo, written_paths: list):
    mo.md(
        "## Wrote\n\n" + "\n".join(f"- `{p}`" for p in written_paths)
    )
    return


if __name__ == "__main__":
    app.run()
