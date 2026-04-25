from __future__ import annotations

import pandas as pd
import streamlit as st

from src.research_utils import (
    build_dataset_analysis_frame,
    language_presence_table,
    summarize_all_splits,
    summarize_dataset_split,
)


st.set_page_config(page_title="Research Analysis", page_icon=":bar_chart:", layout="wide")
st.title("Research Analysis")
st.caption("Paper-oriented dataset characterization, taxonomy analysis, and evidence-composition summaries.")

split_options = ["All splits", "polyfiqa-easy", "polyfiqa-expert"]
selected = st.sidebar.selectbox("Dataset scope", split_options)

overall_summary = summarize_all_splits()

if selected == "All splits":
    metric_cols = st.columns(4)
    metric_cols[0].metric("Splits", len(overall_summary))
    metric_cols[1].metric("Rows", int(overall_summary["rows"].sum()))
    metric_cols[2].metric("Unique task IDs", int(overall_summary["unique_task_ids"].sum()))
    metric_cols[3].metric("Mean prompt chars", f"{overall_summary['avg_prompt_chars'].mean():.0f}")
else:
    summary = summarize_dataset_split(selected)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", summary["rows"])
    metric_cols[1].metric("Unique task IDs", summary["unique_task_ids"])
    metric_cols[2].metric("Companies", summary["unique_companies"])
    metric_cols[3].metric("Avg prompt chars", f"{summary['avg_prompt_chars']:.0f}")

tab_overview, tab_taxonomy, tab_evidence, tab_cases = st.tabs(
    ["Overview", "Task Taxonomy", "Evidence Composition", "Case Review"]
)

with tab_overview:
    st.subheader("Dataset Statistics")
    st.dataframe(overall_summary, use_container_width=True, hide_index=True)

    if selected != "All splits":
        summary = summarize_dataset_split(selected)
        detailed_summary = pd.DataFrame(
            [
                {"Metric": "Average financial statement chars", "Value": f"{summary['avg_financial_chars']:.0f}"},
                {"Metric": "Average news chars", "Value": f"{summary['avg_news_chars']:.0f}"},
                {"Metric": "Average question words", "Value": f"{summary['avg_question_words']:.1f}"},
                {"Metric": "Average reference answer words", "Value": f"{summary['avg_answer_words']:.1f}"},
                {
                    "Metric": "Language block count distribution",
                    "Value": str(summary["language_block_count_distribution"]),
                },
            ]
        )
        st.dataframe(detailed_summary, use_container_width=True, hide_index=True)

with tab_taxonomy:
    if selected == "All splits":
        combined = pd.concat(
            [
                build_dataset_analysis_frame("polyfiqa-easy"),
                build_dataset_analysis_frame("polyfiqa-expert"),
            ],
            ignore_index=True,
        )
    else:
        combined = build_dataset_analysis_frame(selected)

    taxonomy = (
        combined.groupby(["split", "task_category"], as_index=False)
        .agg(
            rows=("row_id", "count"),
            avg_question_words=("question_word_count", "mean"),
            avg_answer_words=("answer_word_count", "mean"),
        )
        .sort_values(["split", "rows", "task_category"], ascending=[True, False, True])
    )
    st.subheader("Question Categories")
    st.dataframe(taxonomy, use_container_width=True, hide_index=True)

    grounding = (
        combined.groupby(["split", "expected_grounding"], as_index=False)
        .agg(rows=("row_id", "count"))
        .sort_values(["split", "rows"], ascending=[True, False])
    )
    st.subheader("Grounding Expectations")
    st.dataframe(grounding, use_container_width=True, hide_index=True)

with tab_evidence:
    if selected == "All splits":
        presence = pd.concat(
            [
                language_presence_table("polyfiqa-easy").assign(split="polyfiqa-easy"),
                language_presence_table("polyfiqa-expert").assign(split="polyfiqa-expert"),
            ],
            ignore_index=True,
        )
        st.subheader("Language Coverage")
        st.dataframe(presence, use_container_width=True, hide_index=True)
    else:
        frame = build_dataset_analysis_frame(selected)
        evidence = frame[
            [
                "task_id",
                "company",
                "task_category",
                "financial_statement_chars",
                "news_chars",
                "news_language_count",
                "financial_share",
                "news_share",
            ]
        ].copy()
        st.subheader("Evidence Composition Per Row")
        st.dataframe(evidence, use_container_width=True, hide_index=True)
        st.subheader("Language Coverage")
        st.dataframe(language_presence_table(selected), use_container_width=True, hide_index=True)

with tab_cases:
    review_split = "polyfiqa-easy" if selected == "All splits" else selected
    frame = build_dataset_analysis_frame(review_split)
    reviewer_companies = st.multiselect(
        "Filter companies",
        options=sorted(frame["company"].unique()),
        default=sorted(frame["company"].unique()),
    )
    review_frame = frame[frame["company"].isin(reviewer_companies)].copy()
    review_index = st.selectbox(
        "Select a row",
        options=review_frame.index.tolist(),
        format_func=lambda idx: f"{review_frame.loc[idx, 'task_id']} | {review_frame.loc[idx, 'question'][:90]}",
    )
    row = review_frame.loc[review_index]
    parsed = row["parsed_prompt"]

    st.markdown(f"**Task category:** `{row['task_category']}`")
    st.markdown(f"**Expected grounding:** `{row['expected_grounding']}`")
    st.markdown(f"**Question:** {row['question']}")
    st.markdown(f"**Reference answer:** {row['answer']}")
    st.text_area("Financial statements context", parsed.financial_statements, height=220)
    for language, content in parsed.news_sections.items():
        with st.expander(f"{language.title()} news", expanded=language == "english"):
            st.text_area(f"{language.title()} block", content, height=160, key=f"{review_split}_{language}_{row['row_id']}")
