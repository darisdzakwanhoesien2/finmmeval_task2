from __future__ import annotations

import streamlit as st

from src.data_utils import load_notes


st.set_page_config(page_title="Workflow Notes", page_icon=":memo:")
st.title("Workflow Notes")
st.caption("A readable view of the existing project notes with a condensed implementation checklist.")

notes = load_notes()

summary_tab, raw_tab = st.tabs(["Structured Summary", "Raw Notes"])

with summary_tab:
    st.subheader("Workflow")
    st.markdown(
        """
        - Ingest and normalize 344 PolyFiQA instances across Easy and Expert settings.
        - Retrieve multilingual evidence with provenance-aware passage tracking.
        - Build prompts that enforce short, grounded, citation-rich answers.
        - Run multilingual model inference and score with ROUGE-1 plus grounding checks.
        - Produce diagnostics, confidence intervals, and reproducibility artifacts.
        """
    )

    st.subheader("Implementation Focus")
    st.markdown(
        """
        - Keep the evaluation harness consistent across splits and languages.
        - Track prompts, seeds, retrieval settings, and model versions.
        - Preserve provenance metadata for financial statements and news evidence.
        - Support replication with documented configs, scripts, and artifact logging.
        """
    )

    st.subheader("Why This App Helps")
    st.markdown(
        """
        This interface makes the notes operational: we can inspect source prompts, compare reference answers,
        and test a hosted model interactively without changing the local dataset or downloading weights.
        """
    )

with raw_tab:
    st.markdown(notes)
