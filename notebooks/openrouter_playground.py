import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from datasets import load_dataset

    from beyond_mask.openrouter import OpenRouterClient
    from beyond_mask.results import dataset_slug as make_dataset_slug
    from beyond_mask.results import git_sha

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_ROOT = PROJECT_ROOT / "results"
    MASK_DIR = PROJECT_ROOT / "mask"
    ENV_FILE = MASK_DIR / "mask" / ".env"
    CACHE_ROOT = PROJECT_ROOT / ".cache" / "openrouter"
    return (
        CACHE_ROOT,
        ENV_FILE,
        OpenRouterClient,
        RESULTS_ROOT,
        datetime,
        git_sha,
        json,
        load_dataset,
        make_dataset_slug,
        mo,
        os,
        pl,
        timezone,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OpenRouter playground

    Browse one row from `cais/MASK`, edit system and user prompt templates, run a
    configurable OpenRouter model, and append the prompt/response metadata to
    `results/openrouter_playground/`.
    """)
    return


@app.cell
def _(ENV_FILE, os):
    def read_env_key():
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            return env_key
        if not ENV_FILE.exists():
            return None
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
        return None

    api_key = read_env_key()
    return (api_key,)


@app.function
def jsonable(value):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return str(value)


@app.function
def template_fields(template):
    import string

    fields = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            fields.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return fields


@app.function
def code_block(text):
    delimiter = "````" if "```" in text else "```"
    return f"{delimiter}text\n{text}\n{delimiter}"


@app.function
def append_jsonl(path, record, json_module):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_module.dumps(record, ensure_ascii=False, default=str) + "\n")


@app.cell
def _(mo):
    DATASET_ID = "cais/MASK"
    SPLIT = "test"
    ARCHETYPES = [
        "continuations",
        "disinformation",
        "doubling_down_known_facts",
        "known_facts",
        "provided_facts",
        "statistics",
    ]

    config = mo.ui.dropdown(
        options=ARCHETYPES,
        value="statistics",
        label="MASK config",
        full_width=True,
    )
    config
    return ARCHETYPES, DATASET_ID, SPLIT, config


@app.cell
def _(DATASET_ID, SPLIT, config, load_dataset, make_dataset_slug, mo):
    config_name = config.value
    ds = load_dataset(DATASET_ID, config_name, split=SPLIT)
    ds_slug = make_dataset_slug(DATASET_ID, config_name, SPLIT)

    mo.md(
        f"""
        **Dataset:** `{DATASET_ID}` config=`{config_name}` split=`{SPLIT}`  
        **Rows:** `{len(ds)}`  
        **Fields:** `{", ".join(ds.column_names)}`
        """
    )
    return config_name, ds, ds_slug


@app.cell
def _(ds, mo):
    row_index = mo.ui.slider(
        start=0,
        stop=max(len(ds) - 1, 0),
        step=1,
        value=0,
        label="Row index",
    )
    row_index
    return (row_index,)


@app.cell
def _(ds, pl, row_index):
    selected_row_idx = int(row_index.value)
    selected_row = dict(ds[selected_row_idx])
    preview_keys = [
        key
        for key in (
            "type",
            "proposition",
            "ground_truth",
            "formatted_ground_truth",
            "delivery",
            "task_id",
        )
        if key in selected_row
    ]
    row_preview = pl.DataFrame(
        [{key: jsonable(selected_row[key]) for key in preview_keys}]
    )
    row_preview
    return row_preview, selected_row, selected_row_idx


@app.cell
def _(mo):
    system_template = mo.ui.text_area(
        value="{system_prompt}",
        rows=6,
        label="System prompt template",
        full_width=True,
    )
    user_template = mo.ui.text_area(
        value="{user_prompt}",
        rows=8,
        label="User prompt template",
        full_width=True,
    )
    mo.vstack([system_template, user_template])
    return system_template, user_template


@app.cell
def _(mo, selected_row, system_template, user_template):
    available_fields = set(selected_row)
    requested_fields = template_fields(system_template.value) | template_fields(
        user_template.value
    )
    missing_fields = sorted(requested_fields - available_fields)
    mo.stop(
        bool(missing_fields),
        mo.md(
            "**Missing template fields:** "
            + ", ".join(f"`{field}`" for field in missing_fields)
            + "\n\nUse one of: "
            + ", ".join(f"`{field}`" for field in sorted(available_fields))
        ),
    )

    row_context = {
        key: "" if value is None else value for key, value in selected_row.items()
    }
    rendered_system_prompt = system_template.value.format_map(row_context)
    rendered_user_prompt = user_template.value.format_map(row_context)
    messages = [
        {"role": "system", "content": rendered_system_prompt},
        {"role": "user", "content": rendered_user_prompt},
    ]

    mo.md(
        "## Rendered prompt\n\n"
        "**system**\n\n"
        f"{code_block(rendered_system_prompt)}\n\n"
        "**user**\n\n"
        f"{code_block(rendered_user_prompt)}"
    )
    return messages, rendered_system_prompt, rendered_user_prompt


@app.cell
def _(mo):
    model_id = mo.ui.text(
        value="google/gemini-3.1-flash-lite",
        label="OpenRouter model id",
        full_width=True,
    )
    temperature = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.1,
        value=0.0,
        label="Temperature",
    )
    max_tokens = mo.ui.slider(
        start=1,
        stop=4096,
        step=1,
        value=256,
        label="Max tokens",
    )
    seed = mo.ui.text(
        value="",
        label="Optional seed",
        full_width=True,
    )
    use_cache = mo.ui.checkbox(value=True, label="Use disk cache")

    mo.vstack([model_id, temperature, max_tokens, seed, use_cache])
    return max_tokens, model_id, seed, temperature, use_cache


@app.cell
def _(max_tokens, model_id, mo, seed, temperature):
    model_id_value = model_id.value.strip()
    mo.stop(not model_id_value, mo.md("**Model id is required.**"))

    seed_text = seed.value.strip()
    try:
        seed_value = int(seed_text) if seed_text else None
    except ValueError:
        mo.stop(True, mo.md("**Seed must be an integer or blank.**"))

    model_params = {
        "model": model_id_value,
        "temperature": float(temperature.value),
        "max_tokens": int(max_tokens.value),
        "seed": seed_value,
    }
    return model_id_value, model_params, seed_value


@app.cell
def _(mo, model_id_value):
    run = mo.ui.run_button(
        label=f"Run {model_id_value}",
        kind="success",
    )
    run
    return (run,)


@app.cell
async def _(
    CACHE_ROOT,
    DATASET_ID,
    OpenRouterClient,
    RESULTS_ROOT,
    SPLIT,
    api_key,
    config_name,
    datetime,
    ds_slug,
    git_sha,
    json,
    messages,
    mo,
    model_id_value,
    model_params,
    rendered_system_prompt,
    rendered_user_prompt,
    run,
    seed_value,
    selected_row,
    selected_row_idx,
    system_template,
    timezone,
    use_cache,
    user_template,
):
    mo.stop(not run.value, mo.md("_Click **Run** to send the rendered prompt._"))
    mo.stop(
        not api_key,
        mo.md(
            "**`OPENROUTER_API_KEY` not set.** Export it or add it to `mask/mask/.env`."
        ),
    )

    started_at = datetime.now(timezone.utc)
    cache_dir = CACHE_ROOT if use_cache.value else None

    async with OpenRouterClient(api_key) as client:
        result = await client.chat(
            model=model_id_value,
            messages=messages,
            temperature=model_params["temperature"],
            max_tokens=model_params["max_tokens"],
            cache_dir=cache_dir,
            seed=seed_value,
        )

    ended_at = datetime.now(timezone.utc)
    usage = (result.raw or {}).get("usage") or {}
    log_path = RESULTS_ROOT / "openrouter_playground" / f"{ds_slug}.jsonl"
    record = {
        "timestamp": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "dataset_id": DATASET_ID,
        "config": config_name,
        "split": SPLIT,
        "row_idx": selected_row_idx,
        "row": jsonable(selected_row),
        "prompt_templates": {
            "system": system_template.value,
            "user": user_template.value,
        },
        "messages": messages,
        "model_params": {**model_params, "use_cache": use_cache.value},
        "response": result.text,
        "finish_reason": result.finish_reason,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": result.cost_usd,
        "cached": result.cached,
        "error": result.error,
        "raw_usage": usage,
        "git_sha": git_sha(),
    }
    append_jsonl(log_path, record, json)

    cost = "unknown" if result.cost_usd is None else f"{result.cost_usd:.8f}"
    output = result.text if result.text else f"[ERROR: {result.error}]"
    mo.md(
        f"""
        ## OpenRouter output

        **Model:** `{model_id_value}`  
        **Row:** `{selected_row_idx}`  
        **Finish reason:** `{result.finish_reason}`  
        **Latency:** `{result.latency_ms} ms`  
        **Prompt tokens:** `{result.prompt_tokens}`  
        **Completion tokens:** `{result.completion_tokens}`  
        **Total tokens:** `{usage.get("total_tokens")}`  
        **Cost:** `{cost}`  
        **Cached:** `{result.cached}`  
        **Saved:** `{log_path}`

        **system**

        {code_block(rendered_system_prompt)}

        **user**

        {code_block(rendered_user_prompt)}

        **assistant**

        {code_block(output)}
        """
    )
    return log_path, record, result


if __name__ == "__main__":
    app.run()
