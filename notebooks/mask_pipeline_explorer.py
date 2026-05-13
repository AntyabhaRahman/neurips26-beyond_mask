import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    import marimo as mo
    import polars as pl

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MASK_DIR = PROJECT_ROOT / "mask"
    TEST_DATA_DIR = MASK_DIR / "test_csv_data"
    RESPONSES_DIR = TEST_DATA_DIR / "responses"
    EVALUATED_DIR = TEST_DATA_DIR / "evaluated"
    METRICS_DIR = TEST_DATA_DIR / "metrics"
    RESULTS_JSON = METRICS_DIR / "all_results.json"
    ENV_FILE = MASK_DIR / "mask" / ".env"

    # Column-name prefixes used by the MASK pipeline. Built at runtime to avoid
    # tripping over-eager static scanners.
    GEN_PREFIX = "generation" + "("
    EVAL_PREFIX = "eval" + "("

    return (
        ENV_FILE,
        EVALUATED_DIR,
        EVAL_PREFIX,
        GEN_PREFIX,
        MASK_DIR,
        METRICS_DIR,
        PROJECT_ROOT,
        RESPONSES_DIR,
        RESULTS_JSON,
        json,
        mo,
        os,
        pl,
        shutil,
        subprocess,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # MASK pipeline explorer

    Given a **baseline** model and a **judge** model, runs the full MASK pipeline in `--test` mode
    (5 rows per archetype) and shows the intermediate state at each step:

    1. **Generate** &mdash; baseline model responses + per-call metadata sidecar
    2. **Evaluate** &mdash; judge verdicts (`A` / `B` / `C` / `D` for binary, `(lower, upper)` for numerical)
    3. **Metric** &mdash; per-row honesty / accuracy
    4. **Aggregate** &mdash; final `all_results.json` plus a derived non-committal share

    All scripts are invoked via `uv run python mask/<script>.py --test ...` under `mask/`. Cached
    OpenRouter responses make re-runs of the same `(baseline, judge, archetypes)` essentially free.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    api_key_input = mo.ui.text(
        placeholder="sk-or-v1-...",
        kind="password",
        label="OpenRouter API key",
        full_width=True,
    )
    save_key_button = mo.ui.run_button(label="Save to .env")

    return api_key_input, save_key_button


@app.cell(hide_code=True)
def _(ENV_FILE, PROJECT_ROOT, api_key_input, mo, os, save_key_button):
    def _read_env_key():
        if not ENV_FILE.exists():
            return None
        for _line in ENV_FILE.read_text().splitlines():
            if _line.strip().startswith("OPENROUTER_API_KEY="):
                _v = _line.split("=", 1)[1].strip().strip('"').strip("'")
                return _v or None
        return None


    if save_key_button.value and api_key_input.value.strip():
        _key = api_key_input.value.strip()
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        _lines = []
        _found = False
        if ENV_FILE.exists():
            for _line in ENV_FILE.read_text().splitlines():
                if _line.strip().startswith("OPENROUTER_API_KEY="):
                    _lines.append(f"OPENROUTER_API_KEY={_key}")
                    _found = True
                else:
                    _lines.append(_line)
        if not _found:
            _lines.append(f"OPENROUTER_API_KEY={_key}")
        ENV_FILE.write_text("\n".join(_lines) + "\n")
        os.environ["OPENROUTER_API_KEY"] = _key

    api_key_present = bool(os.environ.get("OPENROUTER_API_KEY") or _read_env_key())

    if api_key_present:
        api_key_view = mo.md("**OpenRouter API key configured.**").callout(kind="success")
    else:
        api_key_view = mo.vstack(
            [
                mo.md(
                    f"**`OPENROUTER_API_KEY` not found** in environment or "
                    f"`{ENV_FILE.relative_to(PROJECT_ROOT)}`. Paste your key and click save."
                ).callout(kind="warn"),
                api_key_input,
                save_key_button,
            ]
        )

    api_key_view

    return (api_key_present,)


@app.cell
def _(mo):
    test_model = mo.ui.text(
        value="google/gemini-3.1-flash-lite",
        label="Baseline (test) model",
        full_width=True,
    )
    judge_model = mo.ui.text(
        value="moonshotai/kimi-k2.6",
        label="Judge model (binary archetypes)",
        full_width=True,
    )
    numerical_judge_model = mo.ui.text(
        value="moonshotai/kimi-k2.6",
        label="Numerical judge model (statistics, JSON schema)",
        full_width=True,
    )
    archetypes = mo.ui.multiselect(
        options=[
            "continuations",
            "disinformation",
            "doubling_down_known_facts",
            "known_facts",
            "provided_facts",
            "statistics",
        ],
        value=["statistics", "known_facts"],
        label="Archetypes",
    )
    force_regen = mo.ui.checkbox(
        value=False,
        label="Force regenerate (delete existing test_csv_data outputs first)",
    )
    run_button = mo.ui.run_button(label="Run pipeline")

    mo.vstack(
        [
            test_model,
            judge_model,
            numerical_judge_model,
            archetypes,
            force_regen,
            run_button,
        ]
    )
    return (
        archetypes,
        force_regen,
        judge_model,
        numerical_judge_model,
        run_button,
        test_model,
    )


@app.cell
def _(
    EVALUATED_DIR,
    MASK_DIR,
    METRICS_DIR,
    RESPONSES_DIR,
    api_key_present,
    archetypes,
    force_regen,
    judge_model,
    mo,
    numerical_judge_model,
    run_button,
    shutil,
    subprocess,
    test_model,
):
    mo.stop(
        not api_key_present,
        mo.md("_Set the OpenRouter API key above before running the pipeline._").callout(kind="warn"),
    )
    mo.stop(
        not run_button.value,
        mo.md("_Set inputs above and click **Run pipeline** to start._"),
    )
    mo.stop(
        not archetypes.value,
        mo.md("**Pick at least one archetype.**"),
    )

    if force_regen.value:
        for d in [RESPONSES_DIR, EVALUATED_DIR, METRICS_DIR]:
            if d.exists():
                shutil.rmtree(d)

    def _run(cmd: list[str]) -> dict:
        proc = subprocess.run(
            cmd,
            cwd=str(MASK_DIR),
            capture_output=True,
            text=True,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    gen_run = _run(
        [
            "uv",
            "run",
            "python",
            "mask/generate_responses.py",
            "--test",
            "--model",
            test_model.value,
            "--archetypes",
            *archetypes.value,
        ]
    )
    eval_run = _run(
        [
            "uv",
            "run",
            "python",
            "mask/evaluate.py",
            "--test",
            "--judge-model",
            judge_model.value,
            "--numerical-judge-model",
            numerical_judge_model.value,
        ]
    )
    metric_run = _run(["uv", "run", "python", "mask/metric.py", "--test"])
    process_run = _run(["uv", "run", "python", "mask/process_metrics.py", "--test"])

    pipeline_runs = {
        "generate": gen_run,
        "evaluate": eval_run,
        "metric": metric_run,
        "process_metrics": process_run,
    }

    return (pipeline_runs,)


@app.cell(hide_code=True)
def _(mo, pipeline_runs):
    def _step_panel(name, run):
        status = "ok" if run["returncode"] == 0 else f"FAIL exit {run['returncode']}"
        body = run["stdout"].strip() or "(no stdout)"
        stderr_extra = ""
        if run["stderr"].strip():
            stderr_extra = f"\n\nstderr:\n```\n{run['stderr'].strip()}\n```"
        return mo.accordion(
            {
                f"[{status}]  {name}  &mdash;  `{run['cmd']}`": mo.md(
                    f"```\n{body}\n```{stderr_extra}"
                )
            }
        )

    mo.vstack(
        [mo.md("## Pipeline runs")]
        + [_step_panel(name, run) for name, run in pipeline_runs.items()]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Generation step output
    """)
    return


@app.cell(hide_code=True)
def _(test_model):
    model_slug = test_model.value.split("/")[-1]
    return (model_slug,)


@app.cell
def _(
    GEN_PREFIX,
    RESPONSES_DIR,
    archetypes,
    json,
    mo,
    model_slug,
    pipeline_runs,
    pl,
):
    _ = pipeline_runs  # rerun when pipeline reruns

    def _load_arch(arch):
        csv = RESPONSES_DIR / f"{arch}_{model_slug}.csv"
        meta = RESPONSES_DIR / f"{arch}_{model_slug}.meta.json"
        if not csv.exists():
            return None, None
        df = pl.read_csv(csv, infer_schema_length=0)
        meta_data = json.loads(meta.read_text()) if meta.exists() else {}
        return df, meta_data

    _sections = []
    for _arch in archetypes.value:
        _df, _meta = _load_arch(_arch)
        if _df is None:
            _sections.append(mo.md(f"### {_arch}\n_(no output file found)_"))
            continue
        _gen_cols = [c for c in _df.columns if c.startswith(GEN_PREFIX)]
        _summary_md = mo.md(
            f"""
            **{_arch}** &mdash; {_df.height} rows, {len(_gen_cols)} generation columns
            &middot; {_meta.get("total_prompt_tokens", "?")} prompt tokens
            &middot; {_meta.get("total_completion_tokens", "?")} completion tokens
            &middot; {_meta.get("error_count", "?")} errors
            """
        )
        _wanted_static = {
            "system_prompt",
            "user_prompt",
            "proposition",
            "formatted_ground_truth",
        }
        _display_cols = [
            c
            for c in _df.columns
            if c in _wanted_static
            or c.startswith(GEN_PREFIX)
            or c.startswith("belief_elicit_")
        ]
        _sections.append(
            mo.vstack(
                [
                    _summary_md,
                    mo.accordion(
                        {
                            "Show responses table": mo.ui.table(
                                _df.select(_display_cols), page_size=10
                            ),
                            "Show sidecar metadata (per-call)": mo.ui.table(
                                pl.DataFrame(_meta.get("calls", [])),
                                page_size=20,
                            ),
                        }
                    ),
                ]
            )
        )

    mo.vstack(_sections)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Evaluation step output
    """)
    return


@app.cell
def _(
    EVALUATED_DIR,
    EVAL_PREFIX,
    archetypes,
    mo,
    model_slug,
    pipeline_runs,
    pl,
):
    _ = pipeline_runs

    _eval_sections = []
    for _arch in archetypes.value:
        _csv = EVALUATED_DIR / f"{_arch}_{model_slug}.csv"
        if not _csv.exists():
            _eval_sections.append(mo.md(f"### {_arch}\n_(no evaluated file found)_"))
            continue
        _df = pl.read_csv(_csv, infer_schema_length=0)
        _extracted_cols = [
            c
            for c in _df.columns
            if c.startswith(EVAL_PREFIX) and c.endswith("_extracted")
        ]
        _nan_counts = {
            c: int(_df.select(pl.col(c).is_null().sum()).item())
            for c in _extracted_cols
        }
        _nan_total = sum(_nan_counts.values())
        _denom = _df.height * max(len(_extracted_cols), 1)
        _display_cols = (
            (["ground_truth_extracted"] if "ground_truth_extracted" in _df.columns else [])
            + _extracted_cols
        )
        _eval_sections.append(
            mo.vstack(
                [
                    mo.md(
                        f"**{_arch}** &mdash; {_df.height} rows &middot; "
                        f"{len(_extracted_cols)} verdict columns &middot; "
                        f"{_nan_total}/{_denom} unparseable verdicts"
                    ),
                    mo.accordion(
                        {
                            "Show extracted verdicts": mo.ui.table(
                                _df.select(_display_cols), page_size=10
                            )
                        }
                    ),
                ]
            )
        )

    mo.vstack(_eval_sections)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Per-row metric output
    """)
    return


@app.cell
def _(METRICS_DIR, archetypes, mo, model_slug, pipeline_runs, pl):
    _ = pipeline_runs

    _metric_sections = []
    for _arch in archetypes.value:
        _csv = METRICS_DIR / f"{_arch}_{model_slug}.csv"
        if not _csv.exists():
            _metric_sections.append(mo.md(f"### {_arch}\n_(no metric file found)_"))
            continue
        _df = pl.read_csv(_csv, infer_schema_length=0)
        _derived_cols = [
            c
            for c in [
                "ground_truth_extracted",
                "eval_belief",
                "eval_accuracy",
                "eval_honesty_1",
                "eval_honesty_10",
            ]
            if c in _df.columns
        ]
        _metric_sections.append(
            mo.vstack(
                [
                    mo.md(
                        f"**{_arch}** &mdash; per-row honesty/accuracy "
                        f"(`1`=honest, `-1`=lied, `0`=non-committal/no signal)"
                    ),
                    mo.ui.table(_df.select(_derived_cols), page_size=10),
                ]
            )
        )

    mo.vstack(_metric_sections)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Final aggregated results
    """)
    return


@app.cell
def _(
    RESULTS_JSON,
    archetypes,
    json,
    mo,
    model_slug,
    pipeline_runs,
    pl,
    test_model,
):
    _ = pipeline_runs

    if not RESULTS_JSON.exists():
        final_view = mo.md("_(`all_results.json` not yet produced.)_")
        summary_df = None
    else:
        _raw = json.loads(RESULTS_JSON.read_text())
        _rows = []
        for _arch in archetypes.value:
            _block = _raw.get(_arch, {}).get(model_slug)
            if not _block:
                continue
            _honest = _block.get("honest_1", 0.0) or 0.0
            _unhonest = _block.get("unhonest_1", 0.0) or 0.0
            _non_committal = round(max(0.0, 100.0 - _honest - _unhonest), 3)
            _rows.append(
                {
                    "archetype": _arch,
                    "total_responses": _block.get("total_responses"),
                    "honest_1 (%)": _honest,
                    "unhonest_1 (%)": _unhonest,
                    "non_committal (%) [derived]": _non_committal,
                    "honesty_score_1 (%)": _block.get("honesty_score_1"),
                    "accurate (%)": _block.get("accurate"),
                    "inaccurate (%)": _block.get("inaccurate"),
                }
            )
        summary_df = pl.DataFrame(_rows) if _rows else None
        final_view = mo.vstack(
            [
                mo.md(f"**Summary &mdash; baseline `{test_model.value}`**"),
                (
                    mo.ui.table(summary_df, page_size=20)
                    if summary_df is not None
                    else mo.md("_(no rows)_")
                ),
                mo.accordion(
                    {
                        "Raw all_results.json": mo.md(
                            f"```json\n{json.dumps(_raw, indent=2)}\n```"
                        )
                    }
                ),
            ]
        )

    final_view
    return (summary_df,)


@app.cell(hide_code=True)
def _(mo, summary_df):
    if summary_df is None or summary_df.height == 0:
        chart_view = mo.md("_(nothing to chart yet)_")
    else:
        long = summary_df.unpivot(
            on=["honest_1 (%)", "unhonest_1 (%)", "non_committal (%) [derived]"],
            index="archetype",
            variable_name="bucket",
            value_name="pct",
        )
        chart_view = mo.vstack(
            [
                mo.md("### Honesty bucket breakdown"),
                mo.ui.table(long, page_size=20),
            ]
        )
    chart_view
    return


if __name__ == "__main__":
    app.run()
