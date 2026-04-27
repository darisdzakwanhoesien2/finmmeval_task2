from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.research_utils import (
    ERROR_LABELS,
    build_results_discussion_draft,
    error_label_summary,
    list_experiment_runs,
    load_error_analysis_table,
    load_run_config,
    load_run_results,
    save_error_analysis_table,
)


st.set_page_config(page_title="Experiment Review", page_icon=":microscope:", layout="wide")
st.title("Experiment Review")
st.caption("Inspect saved batch runs, summarize quantitative results, and maintain a reusable error-analysis sheet.")

runs = list_experiment_runs()
if not runs:
    st.info(
        "No experiment artifacts found yet. Run `python3 evaluate.py --token <HF_TOKEN> --model <MODEL_ID>` to create the first batch run."
    )
    st.stop()

selected_run = st.selectbox(
    "Saved run",
    options=runs,
    format_func=lambda path: path.name,
)
run_dir = Path(selected_run)
config = load_run_config(run_dir)
results = load_run_results(run_dir)
metrics_path = run_dir / "metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
report_path = run_dir / "report.md"
error_table = load_error_analysis_table(run_dir)


def redacted_config(payload: dict[str, object]) -> dict[str, object]:
    redacted = dict(payload)
    for key in ("token", "api_key", "openrouter_api_key", "hf_token"):
        if key in redacted and redacted[key]:
            redacted[key] = "[redacted]"
    return redacted


def filter_options(frame: pd.DataFrame, column: str, fallback: str = "Unspecified") -> list[str]:
    if column not in frame.columns:
        return []
    values = frame[column].fillna(fallback).astype(str).str.strip()
    values = values.replace("", fallback)
    return sorted(values.unique().tolist())


def normalized_filter_series(frame: pd.DataFrame, column: str, fallback: str = "Unspecified") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([fallback] * len(frame), index=frame.index)
    values = frame[column].fillna(fallback).astype(str).str.strip()
    return values.replace("", fallback)

meta_cols = st.columns(4)
meta_cols[0].metric("Rows", len(results))
meta_cols[1].metric("Model", config.get("model_id", "—"))
meta_cols[2].metric("Prompt mode", config.get("prompt_mode", "—"))
meta_cols[3].metric("Provider", config.get("provider", "—"))

tab_summary, tab_breakdowns, tab_predictions, tab_errors, tab_paper, tab_artifacts = st.tabs(
    ["Summary", "Breakdowns", "Predictions", "Error Analysis", "Paper Draft", "Artifacts"]
)

with tab_summary:
    st.subheader("Run Configuration")
    st.json(redacted_config(config))

    overall = metrics.get("overall", {})
    if overall:
        summary = pd.DataFrame(
            [
                {"Metric": "Mean ROUGE-1 F1", "Value": overall.get("mean_rouge1_f1", 0.0)},
                {"Metric": "Mean penalized ROUGE-1 F1", "Value": overall.get("mean_penalized_rouge1_f1", 0.0)},
                {"Metric": "Length compliance rate", "Value": overall.get("length_compliance_rate", 0.0)},
                {"Metric": "Citation marker rate", "Value": overall.get("citation_marker_rate", 0.0)},
                {"Metric": "Quote grounding coverage", "Value": overall.get("quote_grounding_coverage_mean", 0.0)},
                {
                    "Metric": "Penalized ROUGE-1 bootstrap CI",
                    "Value": str(overall.get("bootstrap_ci_penalized_rouge1_f1", [])),
                },
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if report_path.exists():
        st.subheader("Run Report")
        st.markdown(report_path.read_text(encoding="utf-8"))

with tab_breakdowns:
    st.subheader("By Split")
    st.dataframe(pd.DataFrame(metrics.get("by_split", [])), use_container_width=True, hide_index=True)

    st.subheader("By Task Category")
    st.dataframe(pd.DataFrame(metrics.get("by_task_category", [])), use_container_width=True, hide_index=True)

    st.subheader("By Company")
    st.dataframe(pd.DataFrame(metrics.get("by_company", [])), use_container_width=True, hide_index=True)

    comparisons = pd.DataFrame(metrics.get("paired_prompt_comparisons", []))
    if not comparisons.empty:
        st.subheader("Paired Prompt Comparisons")
        st.dataframe(comparisons, use_container_width=True, hide_index=True)

with tab_predictions:
    st.subheader("Prediction Table")
    if results.empty:
        st.warning("This run has no prediction records.")
    else:
        display_results = results.copy()
        display_results["_filter_split"] = normalized_filter_series(display_results, "split")
        display_results["_filter_company"] = normalized_filter_series(display_results, "company")
        display_results["_filter_task_category"] = normalized_filter_series(display_results, "task_category")
        split_options = filter_options(display_results, "_filter_split")
        company_options = filter_options(display_results, "_filter_company")
        category_options = filter_options(display_results, "_filter_task_category")

        filter_split = st.multiselect(
            "Filter split",
            options=split_options,
            default=split_options,
        )
        filter_company = st.multiselect(
            "Filter company",
            options=company_options,
            default=company_options,
        )
        filter_category = st.multiselect(
            "Filter task category",
            options=category_options,
            default=category_options,
        )
        filtered = display_results[
            display_results["_filter_split"].isin(filter_split)
            & display_results["_filter_company"].isin(filter_company)
            & display_results["_filter_task_category"].isin(filter_category)
        ].copy()
        filtered = filtered.drop(
            columns=["_filter_split", "_filter_company", "_filter_task_category"],
            errors="ignore",
        )
        st.dataframe(filtered, use_container_width=True, hide_index=True)

with tab_errors:
    st.subheader("Error Label Template")
    st.caption("Label a stratified sample of strong and weak outputs to support the qualitative analysis section of the paper.")

    if error_table.empty:
        st.info("No error analysis template found for this run.")
    else:
        editable = error_table.copy()
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            column_config={
                "primary_error_label": st.column_config.SelectboxColumn(
                    "Primary error label",
                    options=[""] + ERROR_LABELS,
                ),
                "secondary_error_label": st.column_config.SelectboxColumn(
                    "Secondary error label",
                    options=[""] + ERROR_LABELS,
                ),
                "analyst_notes": st.column_config.TextColumn("Analyst notes"),
            },
            disabled=[
                "split",
                "row_id",
                "task_id",
                "company",
                "task_category",
                "rouge1_f1_length_penalized",
                "question",
                "reference_answer",
                "prediction",
                "allowed_error_labels",
            ],
        )
        if st.button("Save Error Labels", use_container_width=True):
            save_error_analysis_table(run_dir, pd.DataFrame(edited))
            st.success(f"Saved updated labels to `{run_dir / 'error_analysis_template.csv'}`")

        st.subheader("Labeled Error Summary")
        st.dataframe(error_label_summary(pd.DataFrame(edited)), use_container_width=True, hide_index=True)

with tab_paper:
    st.subheader("Results and Discussion Generator")
    st.caption(
        "Create a paper-ready Markdown draft from the selected run's metrics, breakdowns, predictions, and any saved error labels."
    )

    draft_path = run_dir / "results_discussion_draft.md"
    if st.button("Generate Results & Discussion Draft", type="primary", use_container_width=True):
        draft = build_results_discussion_draft(
            run_name=run_dir.name,
            config=redacted_config(config),
            results=results,
            metrics=metrics,
            error_table=error_table,
        )
        draft_path.write_text(draft, encoding="utf-8")
        st.session_state["results_discussion_draft"] = draft
        st.success(f"Saved draft to `{draft_path}`")

    draft_text = st.session_state.get("results_discussion_draft")
    if draft_text is None and draft_path.exists():
        draft_text = draft_path.read_text(encoding="utf-8")

    if draft_text:
        st.download_button(
            "Download Markdown Draft",
            data=draft_text,
            file_name="results_discussion_draft.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown(draft_text)
    else:
        st.info("Click the button above to generate a Results and Discussion draft for this run.")

with tab_artifacts:
    st.subheader("Files")
    files = [{"name": path.name, "path": str(path)} for path in sorted(run_dir.iterdir())]
    st.dataframe(pd.DataFrame(files), use_container_width=True, hide_index=True)
    st.code(str(run_dir), language="text")
