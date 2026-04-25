from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_utils import build_company_summary, load_dataset, parse_prompt, row_label


st.set_page_config(page_title="Dataset Explorer", page_icon=":mag:")
st.title("Dataset Explorer")

split = st.sidebar.selectbox("Dataset split", ["polyfiqa-easy", "polyfiqa-expert"])
frame = load_dataset(split)
parsed_frame = frame.assign(
    company=frame["task_id"].str.split("_").str[0],
    question_chars=frame["question"].astype(str).str.len(),
    answer_chars=frame["answer"].astype(str).str.len(),
    prompt_chars=frame["query"].astype(str).str.len(),
)

top_left, top_mid, top_right = st.columns(3)
top_left.metric("Rows", len(parsed_frame))
top_mid.metric("Unique task IDs", parsed_frame["task_id"].nunique())
top_right.metric("Companies", parsed_frame["company"].nunique())

stats_col, summary_col = st.columns([1.25, 1])

with stats_col:
    st.subheader("Length Profile")
    st.dataframe(
        pd.DataFrame(
            {
                "Field": ["Prompt", "Question", "Reference answer"],
                "Mean chars": [
                    int(parsed_frame["prompt_chars"].mean()),
                    int(parsed_frame["question_chars"].mean()),
                    int(parsed_frame["answer_chars"].mean()),
                ],
                "Max chars": [
                    int(parsed_frame["prompt_chars"].max()),
                    int(parsed_frame["question_chars"].max()),
                    int(parsed_frame["answer_chars"].max()),
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with summary_col:
    st.subheader("Company Coverage")
    st.dataframe(build_company_summary(parsed_frame), use_container_width=True, hide_index=True)

st.subheader("Browse Rows")
company_filter = st.multiselect(
    "Filter by company",
    options=sorted(parsed_frame["company"].unique()),
    default=sorted(parsed_frame["company"].unique()),
)

filtered = parsed_frame[parsed_frame["company"].isin(company_filter)].reset_index(drop=True)
selected_index = st.selectbox(
    "Choose a dataset row",
    options=filtered.index.tolist(),
    format_func=lambda idx: row_label(filtered.loc[idx]),
)
row = filtered.loc[selected_index]
parsed = parse_prompt(row["query"], row["question"])

meta_1, meta_2, meta_3 = st.columns(3)
meta_1.markdown(f"**Task ID**  \n`{row['task_id']}`")
meta_2.markdown(f"**Company**  \n`{row['company']}`")
meta_3.markdown(f"**Row ID**  \n`{int(row['row_id'])}`")

question_col, answer_col = st.columns(2)
with question_col:
    st.subheader("Question")
    st.write(parsed.question)

with answer_col:
    st.subheader("Reference Answer")
    st.write(row["answer"])

tabs = st.tabs(["Instructions", "Financial Statements", "Multilingual News", "Raw Prompt"])

with tabs[0]:
    st.text_area("Prompt instructions", parsed.instructions, height=220)

with tabs[1]:
    st.text_area("Financial statements context", parsed.financial_statements, height=380)

with tabs[2]:
    if not parsed.news_sections:
        st.warning("No language-separated news blocks were detected in this prompt.")
    else:
        for language, content in parsed.news_sections.items():
            with st.expander(language.title(), expanded=language == "english"):
                st.text_area(f"{language.title()} news", content, height=220, key=f"{language}_{row['row_id']}")

with tabs[3]:
    st.text_area("Original query payload", row["query"], height=450)
