import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell(hide_code=True)
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
    return (
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Choose a dataset row

    Select a MASK archetype, then move the row slider to inspect one example at a
    time. The row preview shows every field as a vertical `field` / `value`
    table, since each MASK split can expose a different schema.
    """)
    return


@app.function(hide_code=True)
def jsonable(value):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return str(value)


@app.function(hide_code=True)
def stringify_value(value):
    import json

    if value is None:
        return ""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


@app.function(hide_code=True)
def display_safe_value(value):
    import html

    return html.escape(stringify_value(value), quote=False)


@app.function(hide_code=True)
def display_safe_row(row):
    return {key: display_safe_value(value) for key, value in row.items()}


@app.function(hide_code=True)
def extract_angle_symbols(value):
    import re

    if not isinstance(value, str):
        return []

    symbols = []
    seen = set()
    for match in re.finditer(r"</?[^<>\s]+[^<>]*>", value):
        symbol = match.group(0)
        if symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


@app.function(hide_code=True)
def extract_row_symbols(row):
    return {
        key: symbols
        for key, value in row.items()
        if (symbols := extract_angle_symbols(value))
    }


@app.function(hide_code=True)
def template_fields(template):
    import string

    fields = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            fields.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return fields


@app.function(hide_code=True)
def trace_returned(turn_result):
    return bool(
        turn_result.get("reasoning")
        or turn_result.get("reasoning_details")
        or turn_result.get("reasoning_tokens")
    )


@app.function(hide_code=True)
def chat_result_record(result):
    return {
        "text": result.text,
        "finish_reason": result.finish_reason,
        "native_finish_reason": result.native_finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "cached": result.cached,
        "error": result.error,
        "request_hash": result.request_hash,
        "message": abable(result.message),
        "reasoning": result.reasoning,
        "reasoning_details": jsonable(result.reasoning_details),
        "trace_returned": trace_returned(
            {
                "reasoning": result.reasoning,
                "reasoning_details": result.reasoning_details,
                "reasoning_tokens": result.reasoning_tokens,
            }
        ),
        "raw": jsonable(result.raw),
    }


@app.function(hide_code=True)
def code_block(text):
    delimiter = "````" if "```" in text else "```"
    return f"{delimiter}text\n{text}\n{delimiter}"


@app.function(hide_code=True)
def chat_message_to_dict(message):
    role = getattr(message, "role", "user") or "user"
    content = getattr(message, "content", None)
    if content is None:
        parts = getattr(message, "parts", [])
        content = "\n".join(str(getattr(part, "text", part)) for part in parts)
    return {"role": role, "content": str(content)}


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MASK Dataset with different splits
    """)
    return


@app.cell(hide_code=True)
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
        value="known_facts",
        label="MASK config",
        full_width=True,
    )
    config
    return DATASET_ID, SPLIT, config


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(ds, json, pl, row_index):
    selected_row_idx = int(row_index.value)

    selected_row = dict(ds[selected_row_idx])

    def preview_value(value):
        if value is None:
            return ""
        if isinstance(value, dict | list | tuple):
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        return str(value)

    row_preview = pl.DataFrame(
        [
            {"field": key, "value": preview_value(value)}
            for key, value in selected_row.items()
        ]
    )
    row_preview
    return selected_row, selected_row_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Write prompt templates

    Use Python-style field names from the selected dataset row, such as
    `{system_prompt}`, `{user_prompt}`, `{proposition}`, or `{ground_truth}`.
    The next cell renders the final messages exactly as they will be sent.
    """)
    return


@app.cell
def _():
    # dataset = pl.read_csv("mask/mask/csv_data/evaluated/doubling_down_known_facts_claude-opus-4-8.csv")
    # dataset.filter(pl.col("eval(System Prompt + User Prompt)_run1_extracted") == "C")
    return


@app.cell
def _():
    # modified_mask = pl.read_csv("edited_mask_data/provided_facts_omission_edited(edited_prompts).csv")
    # modified_mask
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Configure the model call

    Enter any OpenRouter model id, choose sampling settings, and optionally set a
    seed. Leave the seed blank when you do not need deterministic sampling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # MODELS = [
    #     "openai/gpt-5.4-mini",
    #     "openai/gpt-5.4",
    #     "openai/gpt-5.5",
    #     "anthropic/claude-opus-4.8",
    #     "anthropic/claude-opus-4.7",
    #     "anthropic/claude-opus-4.6",
    #     "anthropic/claude-sonnet-4.6",
    #     "anthropic/claude-haiku-4.5",
    #     "google/gemini-3.5-flash",
    #     "google/gemini-3.1-pro-preview",
    #     "deepseek/deepseek-v3.2",
    #     "deepseek/deepseek-v4-pro",
    #     "deepseek/deepseek-v4-flash",
    #     "qwen/qwen3.7-plus",
    #     "qwen/qwen3.7-max",
    #     "moonshotai/kimi-k2.6",
    #     "moonshotai/kimi-k2.5",
    #     "minimax/minimax-m3",
    #     "xiaomi/mimo-v2.5",
    #     "xiaomi/mimo-v2.5-pro",
    #     "x-ai/grok-4.3",
    #     "z-ai/glm-5.1"
    # ]

    MODELS = [
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4",
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-opus-4.7",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "x-ai/grok-4.3",
    ]

    model_id = mo.ui.dropdown(
        options=MODELS,
        value="anthropic/claude-opus-4.8",
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
        stop=32768,
        step=1,
        value=16384,
        label="Max tokens",
    )
    seed = mo.ui.text(
        value="42",
        label="Optional seed",
        full_width=True,
    )
    # Unified reasoning effort. OpenRouter normalizes this per provider: native
    # effort for OpenAI/Grok, thinkingLevel for Gemini 3 (xhigh -> high), and a
    # token budget (~fraction of max_tokens) for Anthropic. "none" disables it.
    REASONING_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh"]
    reasoning_level = mo.ui.dropdown(
        options=REASONING_LEVELS,
        value="high",
        label="Reasoning effort",
        full_width=True,
    )

    mo.vstack([model_id, temperature, max_tokens, seed, reasoning_level])
    return max_tokens, model_id, reasoning_level, seed, temperature


@app.cell(hide_code=True)
def _(max_tokens, mo, model_id, reasoning_level, seed, temperature):
    model_id_value = model_id.value.strip()
    mo.stop(not model_id_value, mo.md("**Model id is required.**"))

    seed_text = seed.value.strip()
    try:
        seed_value = int(seed_text) if seed_text else None
    except ValueError:
        mo.stop(True, mo.md("**Seed must be an integer or blank.**"))

    def build_reasoning(level, model, max_output_tokens):
        # One effort value for every provider; OpenRouter maps it to the right
        # native control (thinkingLevel / token budget). "none" disables reasoning.
        if level == "none":
            return None
        if model.startswith("anthropic/"):
            return {"max_tokens": min(1024, max(1, max_output_tokens - 1)), "exclude": False}
        return {"effort": level, "exclude": False}

    reasoning_param = build_reasoning(
        reasoning_level.value,
        model_id_value,
        int(max_tokens.value),
    )

    model_params = {
        "model": model_id_value,
        "temperature": float(temperature.value),
        "max_tokens": int(max_tokens.value),
        "seed": seed_value,
        "reasoning": reasoning_param,
    }
    return model_id_value, model_params, seed_value


@app.cell
def _(model_id_value):
    print(model_id_value)
    return


@app.cell
def _(mo):
    get_turn_results, set_turn_results = mo.state({})
    return get_turn_results, set_turn_results


@app.cell
def _(get_turn_results):
    turn_results = get_turn_results()
    return (turn_results,)


@app.cell(hide_code=True)
def _(mo):
    system_template = mo.ui.text_area(
        value="{system_prompt}",
        rows=6,
        label="System prompt template",
        full_width=True,
    )
    mo.vstack([system_template])
    return (system_template,)


@app.cell(hide_code=True)
def _(mo, selected_row, system_template):
    available_fields = set(selected_row)
    requested_fields = template_fields(system_template.value)
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

    mo.md(
        f"## Rendered prompt\n\n**system**\n\n{code_block(rendered_system_prompt)}\n\n"
    )
    return (rendered_system_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Continue as a chat

    Use this chat when you want multi-turn behavior. It prepends the rendered
    system prompt, then sends the full user/assistant chat history to OpenRouter
    on every turn. When you are done, use the save button below to persist the
    whole conversation as one JSON file.
    """)
    return


@app.cell(hide_code=True)
def _(
    OpenRouterClient,
    api_key,
    ds_slug,
    get_turn_results,
    mo,
    model_id_value,
    model_params,
    rendered_system_prompt,
    seed_value,
    selected_row_idx,
    set_turn_results,
):
    conversation_id = f"{ds_slug}-row-{selected_row_idx}"

    async def openrouter_chat(chat_messages, config):
        if not api_key:
            return (
                "`OPENROUTER_API_KEY` not set. Export it or add it to `mask/mask/.env`."
            )

        history = [chat_message_to_dict(message) for message in chat_messages]
        turn_idx = sum(1 for m in history if m.get("role") == "assistant") + 1
        outbound_messages = [
            {"role": "system", "content": rendered_system_prompt},
            *history,
        ]
        cache_dir = None  # always re-run; disk cache disabled

        async with OpenRouterClient(api_key) as client:
            result = await client.chat(
                model=model_id_value,
                messages=outbound_messages,
                temperature=model_params["temperature"],
                max_tokens=model_params["max_tokens"],
                cache_dir=cache_dir,
                seed=seed_value,
                session_id=conversation_id,
                reasoning=model_params["reasoning"],
            )

        record = chat_result_record(result)
        record["model"] = model_id_value
        record["requested_reasoning"] = model_params["reasoning"]
        set_turn_results({**get_turn_results(), str(turn_idx): record})
        return result.text if result.text else f"[ERROR: {result.error}]"

    chat = mo.ui.chat(
        openrouter_chat,
        prompts=[],
        max_height=520,
        disabled=not bool(api_key),
    )
    chat
    return chat, conversation_id


@app.cell(hide_code=True)
def _(chat, conversation_id, mo, rendered_system_prompt):
    _history = [chat_message_to_dict(message) for message in chat.value]
    if _history:
        formatted_messages = [
            {"role": "system", "content": rendered_system_prompt},
            *_history,
        ]
        rendered = "\n\n".join(
            f"### {idx}. {message['role']}\n\n{code_block(message['content'])}"
            for idx, message in enumerate(formatted_messages, start=1)
        )
        formatted_conversation_output = mo.md(
            f"""
        ## Formatted conversation

        **Conversation id:** `{conversation_id}`

        {rendered}
        """
        )
    else:
        formatted_conversation_output = mo.md(
            "_No chat turns yet. Send a message above to build the transcript._"
        )

    formatted_conversation_output
    return


@app.cell(hide_code=True)
def _(json, mo, turn_results):
    if turn_results:

        def _render_detail(detail):
            if not isinstance(detail, dict):
                return code_block(str(detail))
            dtype = detail.get("type", "reasoning")
            fmt = detail.get("format")
            fmt_note = f" · format `{fmt}`" if fmt else ""
            if dtype == "reasoning.summary":
                banner = ""
                if fmt == "anthropic-claude-v1":
                    banner = (
                        "> ⚠️ **Summarized reasoning** — Anthropic returns a condensed "
                        "summary, not the full chain of thought.\n\n"
                    )
                return (
                    f"**`reasoning.summary`**{fmt_note}\n\n"
                    f"{banner}{code_block(str(detail.get('summary', '')))}"
                )
            if dtype == "reasoning.encrypted":
                data = str(detail.get("data") or "")
                return (
                    f"**`reasoning.encrypted`**{fmt_note}\n\n"
                    f"_[encrypted/redacted — {len(data)} chars, not human-readable]_"
                )
            if dtype == "reasoning.text":
                sig_note = "\n\n_signed_ ✅" if detail.get("signature") else ""
                return (
                    f"**`reasoning.text`**{fmt_note}\n\n"
                    f"{code_block(str(detail.get('text', '')))}{sig_note}"
                )
            return (
                f"**`{dtype}`**{fmt_note}\n\n"
                f"{code_block(json.dumps(detail, indent=2, ensure_ascii=False, default=str))}"
            )

        sections = []
        for turn_idx, result in sorted(
            turn_results.items(), key=lambda item: int(item[0])
        ):
            if trace_returned(result):
                trace_parts = []
                details = result.get("reasoning_details") or []
                if details:
                    trace_parts.append("\n\n".join(_render_detail(d) for d in details))
                elif result.get("reasoning"):
                    # Some providers only return the flat `reasoning` string.
                    trace_parts.append(
                        "**reasoning**\n\n" + code_block(str(result["reasoning"]))
                    )
                if result.get("reasoning_tokens") is not None:
                    trace_parts.append(
                        f"**reasoning_tokens:** `{result['reasoning_tokens']}`"
                    )
                sections.append(
                    f"### Assistant turn {turn_idx}\n\n" + "\n\n".join(trace_parts)
                )
            else:
                message_keys = ", ".join(sorted((result.get("message") or {}).keys()))
                requested_reasoning = json.dumps(
                    result.get("requested_reasoning"),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
                sections.append(
                    f"### Assistant turn {turn_idx}\n\n"
                    "_No reasoning trace was returned by the selected model/provider for this turn._\n\n"
                    f"**model:** `{result.get('model')}`\n\n"
                    f"**requested reasoning**\n\n{code_block(requested_reasoning)}\n\n"
                    f"**assistant message keys:** `{message_keys or 'none'}`\n\n"
                    f"**finish_reason:** `{result.get('finish_reason')}`"
                )
        trace_output = mo.md("## Reasoning trace\n\n" + "\n\n".join(sections))
    else:
        trace_output = mo.md(
            "## Reasoning trace\n\n_No OpenRouter turns have completed yet._"
        )

    trace_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. End and save

    Click this once you are finished with the chat. It writes the full
    system/user/assistant conversation to a single JSON file.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    save_conversation = mo.ui.run_button(
        label="End conversation and save JSON",
        kind="success",
    )
    save_conversation
    return (save_conversation,)


@app.cell
def _(
    DATASET_ID,
    RESULTS_ROOT,
    SPLIT,
    chat,
    config_name,
    conversation_id,
    datetime,
    ds_slug,
    git_sha,
    json,
    mo,
    model_params,
    rendered_system_prompt,
    save_conversation,
    selected_row,
    selected_row_idx,
    system_template,
    timezone,
    turn_results,
):
    mo.stop(
        not save_conversation.value,
        mo.md("_Use the chat above, then click **End conversation and save JSON**._"),
    )

    _history = [chat_message_to_dict(message) for message in chat.value]
    mo.stop(not _history, mo.md("**No chat messages to save yet.**"))

    saved_at = datetime.now(timezone.utc)
    # Inline each assistant turn's reasoning into the transcript (matched by turn
    # index) so the saved messages are self-contained. The model-facing history is
    # unchanged — reasoning is recorded here, not fed back to the model.
    _assistant_seen = 0
    _transcript = []
    for _msg in _history:
        if _msg.get("role") == "assistant":
            _assistant_seen += 1
            _rec = turn_results.get(str(_assistant_seen), {})
            _transcript.append(
                {
                    **_msg,
                    "reasoning": _rec.get("reasoning"),
                    "reasoning_details": _rec.get("reasoning_details"),
                    "reasoning_tokens": _rec.get("reasoning_tokens"),
                }
            )
        else:
            _transcript.append(_msg)
    conversation_messages = [
        {"role": "system", "content": rendered_system_prompt},
        *_transcript,
    ]
    # Per-turn reasoning + metadata captured during the chat (drop the bulky raw).
    reasoning_traces = {
        turn_idx: {k: v for k, v in record_.items() if k != "raw"}
        for turn_idx, record_ in turn_results.items()
    }
    record = {
        "saved_at": saved_at.isoformat(),
        "mode": "chat",
        "conversation_id": conversation_id,
        "dataset_id": DATASET_ID,
        "config": config_name,
        "split": SPLIT,
        "row_idx": selected_row_idx,
        "row": jsonable(selected_row),
        "prompt_templates": {
            "system": system_template.value,
        },
        "model_params": model_params,
        "messages": conversation_messages,
        "reasoning_traces": reasoning_traces,
        "git_sha": git_sha(),
    }

    save_dir = RESULTS_ROOT / "openrouter_playground" / "conversations"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        save_dir
        / f"{ds_slug}__row-{selected_row_idx}__{saved_at.strftime('%Y%m%d-%H%M%S')}.json"
    )
    save_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    mo.md(
        f"""
        **Saved conversation JSON:** `{save_path}`

        **Messages saved:** `{len(conversation_messages)}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
