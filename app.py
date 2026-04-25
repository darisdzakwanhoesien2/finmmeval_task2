from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.model_selector_ui import render_model_selector
from shared.model_provider import call_chat_completion

from evaluate import DEFAULT_SYSTEM_PROMPT, EvaluationConfig, run_evaluation
from src.data_utils import DATA_DIR, NOTES_PATH, dataset_stats
from src.research_utils import list_experiment_runs, load_run_config, load_run_results


st.set_page_config(
    page_title="FinMMEval Task 2 Explorer",
    page_icon=":bar_chart:",
    layout="wide",
)

# ── Checkpoint helpers ────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _checkpoint_path(run_name: str) -> Path:
    return RESULTS_DIR / f"{run_name}_checkpoint.jsonl"


def _load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    """Return (records, done_task_ids) from an existing checkpoint file."""
    if not path.exists():
        return [], set()
    records: list[dict] = []
    done_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
                # task_id is the natural dedup key for Task 2
                if "task_id" in rec:
                    done_ids.add(str(rec["task_id"]))
            except json.JSONDecodeError:
                pass  # skip corrupt lines silently
    return records, done_ids


def _append_checkpoint(path: Path, record: dict) -> None:
    """Append a single completed record to the JSONL checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────

st.title("FinMMEval Task 2 Explorer")
st.caption("Dataset exploration, workflow notes, streamed Hugging Face inference, and end-to-end batch evaluation from one app.")

overview_tab, run_tab, runs_tab = st.tabs(["Overview", "Run Full Evaluation", "Recent Runs"])

with overview_tab:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Project Overview")
        st.markdown(
            """
            This workspace now supports the full research workflow in one place:

            - Inspect both `polyfiqa-easy` and `polyfiqa-expert` test sets.
            - Read the thesis-style workflow and implementation notes already in the repo.
            - Open individual prompts as structured financial statements plus multilingual news evidence.
            - Run a remote Hugging Face chat model in streaming mode without downloading weights locally.
            - Launch a reproducible batch evaluation that saves metrics, predictions, and error-analysis artifacts.
            """
        )
        st.dataframe(dataset_stats(), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Available Assets")
        st.markdown(
            f"""
            - Dataset directory: `{DATA_DIR}`
            - Notes file: `{NOTES_PATH.name}`
            - Main pages:
              - `Dataset Explorer`
              - `Workflow Notes`
              - `Model Playground`
              - `Research Analysis`
              - `Experiment Review`
            """
        )
        st.info(
            "Use the `Run Full Evaluation` tab here when you want the homepage to orchestrate the full experiment instead of running `evaluate.py` manually."
        )

    st.subheader("Suggested Flow")
    st.markdown(
        """
        1. Inspect the dataset and prompt structure with `Dataset Explorer`.
        2. Review methodology in `Workflow Notes` and `Research Analysis`.
        3. Test one row interactively in `Model Playground`.
        4. Run the full evaluation from the tab on this page.
        5. Review saved outputs in `Experiment Review`.
        """
    )

with run_tab:
    st.subheader("Run Batch Evaluation")
    st.caption("Supports both OpenRouter (free/paid/vision) and HuggingFace (free/vision) hosted inference.")

    settings_col, help_col = st.columns([1.15, 0.85])

    with settings_col:

        # ── Unified model & key selector ──────────────────────────────────────
        with st.expander("⚙️ Model & API Key Settings", expanded=True):
            model_id, provider, api_key = render_model_selector(
                key_prefix="task2_run",
                show_vision_filter=False,
            )

        hf_provider_hint = st.selectbox(
            "HF provider hint (only used when provider = huggingface)",
            options=["auto", "hf-inference", "nebius", "together"],
            key="task2_hf_hint",
        )

        # ── Eval settings ─────────────────────────────────────────────────────
        prompt_mode   = st.selectbox("Prompt mode", ["curated", "raw"], index=0)
        split         = st.selectbox("Dataset split", ["all", "polyfiqa-easy", "polyfiqa-expert"], index=0)
        limit         = st.number_input("Row limit (0 = all rows)", min_value=0, max_value=500, value=0, step=1)
        run_name      = st.text_input("Run name", value="task2-run")
        sample_size   = st.slider("Error-analysis sample size", min_value=6, max_value=30, value=12, step=2)
        max_tokens    = st.slider("Max new tokens", min_value=64, max_value=1024, value=256, step=32)
        temperature   = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.1)
        top_p         = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=110)

        # ── Checkpoint status ─────────────────────────────────────────────────
        effective_run_name = run_name.strip() or "task2-run"
        ckpt_path          = _checkpoint_path(effective_run_name)
        existing_records, done_ids = _load_checkpoint(ckpt_path)
        resume_count = len(existing_records)

        if resume_count > 0:
            st.info(
                f"Checkpoint found: **{resume_count}** records already completed "
                f"for run `{effective_run_name}`. Clicking **Run** will resume from where it left off."
            )
            if st.button("Clear checkpoint and restart", key="task2_clear_checkpoint"):
                ckpt_path.unlink(missing_ok=True)
                st.success("Checkpoint cleared. Re-click **Run** to start fresh.")
                st.rerun()

        run_button = st.button("Run Full Evaluation", type="primary", use_container_width=True)

    with help_col:
        st.subheader("What Gets Saved")
        st.markdown(
            """
            Each run writes a reproducible artifact bundle:

            - `config.json`
            - `dataset_summary.csv`
            - split-level `*_predictions.jsonl`
            - `all_predictions.csv`
            - `metrics.json`
            - `error_analysis_template.csv`
            - `report.md`

            **Checkpoint file** (`results/<run_name>_checkpoint.jsonl`):
            - Written row-by-row during the run.
            - Survives browser refresh or network interruption.
            - Deleted automatically once the run is fully saved.
            """
        )
        st.subheader("Provider Guide")
        st.markdown(
            """
            | Provider | Free models | Paid models | Vision |
            |---|---|---|---|
            | 🌐 OpenRouter | ✅ (`:free` suffix) | ✅ | ✅ |
            | 🤗 HuggingFace | ✅ (warm endpoint) | ➖ | ✅ some |

            Use **Fetch models** in the selector above to pull a live list.
            """
        )
        st.caption("Re-click **Run** at any time to resume from the last saved checkpoint.")

    if run_button:
        if not api_key:
            st.error("Provide an API key first (OpenRouter or HuggingFace).")
        else:
            # Reload checkpoint right before running (state may differ from page load)
            existing_records, done_ids = _load_checkpoint(ckpt_path)
            skipped = len(existing_records)

            progress_bar = st.progress(
                0.0,
                text=f"Starting — {skipped} rows already in checkpoint",
            )
            status = st.empty()

            # ── Row-level progress callback that also writes the checkpoint ───
            completed_this_session: list[dict] = []

            def update_progress(
                current: int,
                total: int,
                message: str,
                *,
                completed_record: dict | None = None,
            ) -> None:
                """
                Called by run_evaluation after each row.

                `completed_record` should be the dict for the just-finished row.
                If run_evaluation doesn't pass it, checkpointing still works at
                the end via the returned predictions DataFrame.
                """
                fraction = 0.0 if total == 0 else min(max(current / total, 0.0), 1.0)
                progress_bar.progress(fraction, text=f"{message} ({current}/{total})")
                status.info(message)

                if completed_record is not None:
                    task_id = str(completed_record.get("task_id", ""))
                    if task_id and task_id not in done_ids:
                        _append_checkpoint(ckpt_path, completed_record)
                        done_ids.add(task_id)
                        completed_this_session.append(completed_record)

            try:
                outcome = run_evaluation(
                    EvaluationConfig(
                        token=api_key.strip(),
                        model=model_id.strip(),
                        provider=provider,
                        hf_provider_hint=hf_provider_hint,
                        prompt_mode=prompt_mode,
                        split=split,
                        limit=None if limit == 0 else int(limit),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system_prompt=system_prompt,
                        run_name=effective_run_name,
                        sample_size=sample_size,
                        skip_task_ids=done_ids,
                    ),
                    progress_callback=update_progress,
                )

                # ── Fallback: if run_evaluation returns predictions, flush any
                #    records not yet written by the callback ─────────────────
                if "predictions" in outcome:
                    predictions_df: pd.DataFrame = outcome["predictions"]
                    for _, row in predictions_df.iterrows():
                        rec = row.to_dict()
                        task_id = str(rec.get("task_id", ""))
                        if task_id and task_id not in done_ids:
                            _append_checkpoint(ckpt_path, rec)
                            done_ids.add(task_id)

                # ── Merge checkpoint records with new ones for the summary view
                all_records = existing_records + completed_this_session

                # ── Clean up checkpoint only after a confirmed full save ───────
                if outcome.get("run_dir"):
                    ckpt_path.unlink(missing_ok=True)

                progress_bar.progress(1.0, text="Evaluation complete")
                status.success(f"Saved run to `{outcome['run_dir']}` — checkpoint cleared.")

                st.dataframe(
                    pd.DataFrame(outcome["metrics"].get("by_split", [])),
                    use_container_width=True,
                    hide_index=True,
                )
                st.code("\n".join(outcome["summary_lines"]), language="text")

                if all_records:
                    st.markdown(f"**Total rows processed (including resumed):** {len(all_records)}")
                    st.dataframe(
                        pd.DataFrame(all_records).head(25),
                        use_container_width=True,
                        hide_index=True,
                    )

            except ValueError as exc:
                status.empty()
                progress_bar.empty()
                msg = str(exc)
                if "400" in msg or "Bad Request" in msg:
                    st.error(
                        "**400 Bad Request** from OpenRouter.\n\n"
                        "**Common causes:**\n"
                        "- Model ID is incorrect or no longer available — try fetching the model list again.\n"
                        "- Temperature or top-p is out of the accepted range.\n"
                        "- The selected model does not support `chat/completions`.\n\n"
                        f"**Detail:** `{msg}`"
                    )
                elif "401" in msg or "Unauthorized" in msg:
                    st.error("**401 Unauthorized** — your API key is invalid or expired. Please re-enter it.")
                else:
                    st.exception(exc)

            except RuntimeError as exc:
                status.empty()
                progress_bar.empty()
                msg = str(exc)
                if "429" in msg or "Too Many Requests" in msg or "after" in msg.lower():
                    st.warning(
                        "**429 Too Many Requests** — OpenRouter rate limit hit.\n\n"
                        "The evaluation was automatically retried with exponential back-off. "
                        "If it still fails, wait 60 seconds and click **Run Full Evaluation** again — "
                        f"your checkpoint has **{len(existing_records) + len(completed_this_session)} rows** saved."
                    )
                else:
                    st.exception(exc)
                st.warning(
                    f"**{len(existing_records) + len(completed_this_session)} records** "
                    f"are saved in the checkpoint at `{ckpt_path}`. "
                    f"Re-click **Run Full Evaluation** to resume."
                )

            except Exception as exc:
                status.empty()
                progress_bar.empty()
                st.exception(exc)
                st.warning(
                    f"Evaluation interrupted. "
                    f"**{len(existing_records) + len(completed_this_session)} records** "
                    f"are saved in the checkpoint at `{ckpt_path}`. "
                    f"Re-click **Run Full Evaluation** to resume from where it stopped."
                )

with runs_tab:
    st.subheader("Recent Runs")
    runs = list_experiment_runs()
    if not runs:
        st.info("No saved runs yet.")
    else:
        latest = runs[0]
        config = load_run_config(latest)
        results = load_run_results(latest)
        latest_cols = st.columns(4)
        latest_cols[0].metric("Saved runs", len(runs))
        latest_cols[1].metric("Latest model", config.get("model_id", "—"))
        latest_cols[2].metric("Latest prompt mode", config.get("prompt_mode", "—"))
        latest_cols[3].metric("Latest rows", len(results))
        st.markdown(f"**Latest run:** `{latest.name}`")
        if not results.empty:
            preview_columns = [
                "split",
                "task_id",
                "company",
                "task_category",
                "rouge1_f1_length_penalized",
                "length_compliant",
            ]
            available_columns = [column for column in preview_columns if column in results.columns]
            st.dataframe(results[available_columns], use_container_width=True, hide_index=True)

        # ── Show any active checkpoints alongside saved runs ──────────────────
        active_checkpoints = sorted(RESULTS_DIR.glob("*_checkpoint.jsonl")) if RESULTS_DIR.exists() else []
        if active_checkpoints:
            st.divider()
            st.subheader("Active Checkpoints")
            st.caption("These runs were interrupted and can be resumed from the Run tab.")
            for ckpt in active_checkpoints:
                recs, _ = _load_checkpoint(ckpt)
                col_a, col_b, col_c = st.columns([2, 1, 1])
                col_a.write(f"`{ckpt.stem}`")
                col_b.write(f"{len(recs)} rows saved")
                if col_c.button("Delete", key=f"del_{ckpt.stem}"):
                    ckpt.unlink(missing_ok=True)
                    st.rerun()

if not Path(DATA_DIR).exists() or not Path(NOTES_PATH).exists():
    st.error("Expected dataset files or notes are missing from the workspace.")
