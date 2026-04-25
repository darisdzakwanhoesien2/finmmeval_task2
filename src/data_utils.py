from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
NOTES_PATH = ROOT_DIR / "notes.md"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

LANGUAGE_MARKERS = {
    "english": "English News:",
    "chinese": "Chinese News:",
    "japanese": "Japanese News:",
    "spanish": "Spanish News:",
    "greek": "Greek News:",
}


@dataclass(frozen=True)
class ParsedPrompt:
    instructions: str
    financial_statements: str
    news_sections: Dict[str, str]
    question: str


@st.cache_data(show_spinner=False)
def load_dataset(split: str) -> pd.DataFrame:
    split_path = DATA_DIR / split / "test.csv"
    frame = pd.read_csv(split_path)
    frame["row_id"] = frame.index
    return frame


@st.cache_data(show_spinner=False)
def load_notes() -> str:
    return NOTES_PATH.read_text(encoding="utf-8")


def dataset_stats() -> pd.DataFrame:
    rows = []
    for split in ("polyfiqa-easy", "polyfiqa-expert"):
        frame = load_dataset(split)
        rows.append(
            {
                "Split": split,
                "Rows": len(frame),
                "Unique task IDs": frame["task_id"].nunique(),
                "Avg prompt chars": int(frame["query"].astype(str).str.len().mean()),
                "Avg question chars": int(frame["question"].astype(str).str.len().mean()),
                "Avg answer chars": int(frame["answer"].astype(str).str.len().mean()),
            }
        )
    return pd.DataFrame(rows)


def available_splits() -> list[str]:
    return ["polyfiqa-easy", "polyfiqa-expert"]


def company_from_task_id(task_id: str) -> str:
    return str(task_id).split("_")[0]


def parse_prompt(query_text: str, question_text: str) -> ParsedPrompt:
    context_marker = "Context:\n"
    financial_marker = "Financial Statements:\n"
    question_marker = "\nQuestion:"

    context_start = query_text.find(context_marker)
    question_start = query_text.rfind(question_marker)

    instructions = query_text[:context_start].strip() if context_start != -1 else query_text.strip()
    context_body = (
        query_text[context_start + len(context_marker) : question_start].strip()
        if context_start != -1 and question_start != -1
        else ""
    )
    extracted_question = (
        query_text[question_start + len(question_marker) :].strip() if question_start != -1 else question_text.strip()
    )

    financial_start = context_body.find(financial_marker)
    if financial_start == -1:
        return ParsedPrompt(
            instructions=instructions,
            financial_statements=context_body,
            news_sections={},
            question=extracted_question,
        )

    content_after_financial = context_body[financial_start + len(financial_marker) :]
    marker_positions = {
        name: content_after_financial.find(marker)
        for name, marker in LANGUAGE_MARKERS.items()
        if content_after_financial.find(marker) != -1
    }

    if not marker_positions:
        return ParsedPrompt(
            instructions=instructions,
            financial_statements=content_after_financial.strip(),
            news_sections={},
            question=extracted_question,
        )

    ordered_markers = sorted(marker_positions.items(), key=lambda item: item[1])
    first_marker_index = ordered_markers[0][1]
    financial_statements = content_after_financial[:first_marker_index].strip()

    news_sections: Dict[str, str] = {}
    for index, (language, start_position) in enumerate(ordered_markers):
        marker = LANGUAGE_MARKERS[language]
        content_start = start_position + len(marker)
        content_end = (
            ordered_markers[index + 1][1]
            if index + 1 < len(ordered_markers)
            else len(content_after_financial)
        )
        news_sections[language] = content_after_financial[content_start:content_end].strip()

    return ParsedPrompt(
        instructions=instructions,
        financial_statements=financial_statements,
        news_sections=news_sections,
        question=extracted_question,
    )


def build_prompt_text(record: dict, *args, **kwargs) -> str:
    # Replace undefined parser with a robust, defensive local parsing step.
    # This accepts several possible shapes for the "news" payload and
    # normalizes it to a dict-like sequence of (language, text) pairs.
    raw_news = record.get("news_sections") or record.get("news") or {}
    if isinstance(raw_news, dict):
        news_items = raw_news.items()
    elif isinstance(raw_news, str):
        news_items = [("news", raw_news)]
    elif isinstance(raw_news, (list, tuple)):
        # list of chunks => enumerate as unnamed sections
        news_items = [(f"news_{i}", item) for i, item in enumerate(raw_news)]
    else:
        news_items = []

    news_parts: list[str] = []
    for language, content in news_items:
        if content is None:
            continue
        if isinstance(content, (list, tuple)):
            content_text = "\n".join(str(x) for x in content)
        else:
            content_text = str(content)
        news_parts.append(f"{language.title()} News:\n{content_text}")

    news_text = "\n\n".join(news_parts)

    # Make normalized "parsed" available to the rest of the function if needed.
    parsed = {"news_sections": dict(news_items), "news_text": news_text}
    prompt_parts = [
        # ...other prompt parts...
        news_text,
    ]
    return "\n\n".join(p for p in prompt_parts if p)


def parsed_prompt_to_dict(parsed: ParsedPrompt) -> Dict[str, object]:
    return asdict(parsed)


def build_company_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.assign(company=frame["task_id"].str.split("_").str[0])
        .groupby("company", as_index=False)
        .agg(
            tasks=("task_id", "nunique"),
            questions=("row_id", "count"),
            avg_reference_answer_chars=("answer", lambda s: int(s.astype(str).str.len().mean())),
        )
        .sort_values(["questions", "company"], ascending=[False, True])
    )
    return summary


def enrich_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["company"] = enriched["task_id"].map(company_from_task_id)
    enriched["prompt_chars"] = enriched["query"].astype(str).str.len()
    enriched["question_chars"] = enriched["question"].astype(str).str.len()
    enriched["answer_chars"] = enriched["answer"].astype(str).str.len()
    enriched["parsed_prompt"] = [
        parse_prompt(query_text, question_text)
        for query_text, question_text in zip(enriched["query"], enriched["question"])
    ]
    enriched["financial_statement_chars"] = enriched["parsed_prompt"].map(
        lambda parsed: len(parsed.financial_statements)
    )
    enriched["news_chars"] = enriched["parsed_prompt"].map(
        lambda parsed: sum(len(text) for text in parsed.news_sections.values())
    )
    enriched["news_language_count"] = enriched["parsed_prompt"].map(lambda parsed: len(parsed.news_sections))
    for language in LANGUAGE_MARKERS:
        enriched[f"has_{language}_news"] = enriched["parsed_prompt"].map(
            lambda parsed, lang=language: lang in parsed.news_sections and bool(parsed.news_sections[lang].strip())
        )
    return enriched


def row_label(row: pd.Series) -> str:
    preview = str(row["question"]).replace("\n", " ")
    if len(preview) > 90:
        preview = f"{preview[:87]}..."
    return f'{row["task_id"]} | {preview}'
