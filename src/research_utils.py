from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable
import warnings

import pandas as pd

from src.data_utils import ARTIFACTS_DIR, available_splits, build_prompt_text, enrich_dataset, load_dataset


ERROR_LABELS = [
    "unsupported_claim",
    "wrong_financial_comparison",
    "missing_multilingual_evidence",
    "over_reliance_on_single_language",
    "hallucinated_quote",
    "incomplete_answer",
    "formatting_violation",
    "other",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))


def normalize_text(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)


def rouge_1_scores(reference: str, prediction: str) -> dict[str, float]:
    ref_tokens = normalize_text(reference)
    pred_tokens = normalize_text(prediction)
    if not ref_tokens or not pred_tokens:
        return {"rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0}

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(ref_counts[token], pred_counts[token]) for token in ref_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "rouge1_precision": precision,
        "rouge1_recall": recall,
        "rouge1_f1": f1,
    }


def detect_task_category(question: str) -> str:
    question_lower = str(question).lower()
    rules = [
        ("capital_allocation", ["capital", "share repurchase", "dividend", "investments"]),
        ("balance_sheet_health", ["balance sheet", "current assets", "liability", "equity", "ratio"]),
        ("revenue_trend", ["revenue", "sales", "trend"]),
        ("profitability", ["profit", "margin", "earnings", "income"]),
        ("risk_outlook", ["risk", "outlook", "guidance", "uncertainty"]),
        ("segment_performance", ["segment", "business unit", "cloud", "azure", "division"]),
        ("evidence_retrieval", ["quote", "evidence", "news", "financial statements evidence"]),
    ]
    for label, keywords in rules:
        if any(keyword in question_lower for keyword in keywords):
            return label
    return "other"


def detect_expected_grounding(question: str, answer: str) -> str:
    combined = f"{question}\n{answer}".lower()
    if "news evidence" in combined and "financial statement" in combined:
        return "both"
    if "news" in combined:
        return "news_first"
    if "financial statement" in combined:
        return "statements_first"
    return "unspecified"


def extract_quotes(text: str) -> list[str]:
    quote_patterns = [
        r'"([^"\n]{5,300})"',
        r"\u201c([^\u201d\n]{5,300})\u201d",
        r"'([^'\n]{5,300})'",
    ]
    quotes: list[str] = []
    for pattern in quote_patterns:
        quotes.extend(match.strip() for match in re.findall(pattern, str(text)))
    return quotes


def grounding_quote_coverage(prediction: str, source_text: str) -> float:
    quotes = extract_quotes(prediction)
    if not quotes:
        return 0.0
    grounded = sum(1 for quote in quotes if quote in source_text)
    return grounded / len(quotes)


def has_citation_markers(text: str) -> bool:
    lowered = str(text).lower()
    markers = [
        "answer:",
        "news evidence:",
        "financial statements evidence:",
        "evidence:",
        "quote",
    ]
    return any(marker in lowered for marker in markers)


def compute_length_penalty(prediction: str, max_words: int = 100) -> float:
    count = word_count(prediction)
    if count <= max_words:
        return 1.0
    return max(max_words / count, 0.0)


def bootstrap_confidence_interval(
    values: Iterable[float],
    *,
    num_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float]:
    clean_values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean_values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(num_samples):
        sample = [clean_values[rng.randrange(len(clean_values))] for _ in range(len(clean_values))]
        means.append(mean(sample))
    means.sort()
    alpha = (1 - confidence) / 2
    low_index = max(0, int(alpha * len(means)) - 1)
    high_index = min(len(means) - 1, int((1 - alpha) * len(means)) - 1)
    return means[low_index], means[high_index]


def load_enriched_split(split: str) -> pd.DataFrame:
    return enrich_dataset(load_dataset(split))


def build_dataset_analysis_frame(split: str) -> pd.DataFrame:
    frame = load_enriched_split(split).copy()
    frame["task_category"] = frame["question"].map(detect_task_category)
    frame["expected_grounding"] = [
        detect_expected_grounding(question, answer)
        for question, answer in zip(frame["question"], frame["answer"])
    ]
    frame["question_word_count"] = frame["question"].map(word_count)
    frame["answer_word_count"] = frame["answer"].map(word_count)
    frame["financial_share"] = frame.apply(
        lambda row: row["financial_statement_chars"] / row["prompt_chars"] if row["prompt_chars"] else 0.0,
        axis=1,
    )
    frame["news_share"] = frame.apply(
        lambda row: row["news_chars"] / row["prompt_chars"] if row["prompt_chars"] else 0.0,
        axis=1,
    )
    return frame


def summarize_dataset_split(split: str) -> dict[str, object]:
    frame = build_dataset_analysis_frame(split)
    language_distribution = frame["news_language_count"].value_counts().sort_index().to_dict()
    return {
        "split": split,
        "rows": int(len(frame)),
        "unique_task_ids": int(frame["task_id"].nunique()),
        "unique_companies": int(frame["company"].nunique()),
        "avg_prompt_chars": float(frame["prompt_chars"].mean()),
        "avg_financial_chars": float(frame["financial_statement_chars"].mean()),
        "avg_news_chars": float(frame["news_chars"].mean()),
        "avg_question_words": float(frame["question_word_count"].mean()),
        "avg_answer_words": float(frame["answer_word_count"].mean()),
        "language_block_count_distribution": language_distribution,
        "task_category_distribution": frame["task_category"].value_counts().to_dict(),
    }


def summarize_all_splits() -> pd.DataFrame:
    return pd.DataFrame([summarize_dataset_split(split) for split in available_splits()])


def language_presence_table(split: str) -> pd.DataFrame:
    frame = build_dataset_analysis_frame(split)
    rows = []
    for language in ["english", "chinese", "japanese", "spanish", "greek"]:
        rows.append(
            {
                "language": language,
                "rows_with_language": int(frame[f"has_{language}_news"].sum()),
                "share_of_rows": float(frame[f"has_{language}_news"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction(
    *,
    split: str,
    row: dict | pd.Series,
    prediction: str,
    prompt_mode: str,
    model_id: str,
    provider: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
) -> dict[str, object]:
    """
    Evaluate a single prediction against its reference.

    `row` can be either a dict (as returned by _evaluate_single_row in evaluate.py)
    or a pd.Series (as used in batch analysis). All field access is done via .get()
    so both types work safely.
    """
    # Normalise row to a plain dict so .get() always works
    if isinstance(row, pd.Series):
        row = row.to_dict()

    prediction = str(prediction).strip()

    # Build prompt text defensively — fall back to empty string if keys missing
    parsed_prompt = row.get("parsed_prompt") or row.get("query") or ""
    try:
        prompt_text = (
            row.get("query", "")
            if prompt_mode == "raw"
            else build_prompt_text(parsed_prompt, "curated")
        )
    except Exception:
        prompt_text = str(parsed_prompt)

    reference_answer = str(row.get("answer", row.get("reference_answer", "")))
    source_text = f"{row.get('query', '')}\n{reference_answer}"

    rouge = rouge_1_scores(reference_answer, prediction)
    word_total = word_count(prediction)
    length_penalty = compute_length_penalty(prediction)
    quote_coverage = grounding_quote_coverage(prediction, source_text)

    # row_id: prefer explicit field, fall back to index or empty string
    row_id_raw = row.get("row_id", row.get("idx", ""))
    try:
        row_id = int(row_id_raw)
    except (TypeError, ValueError):
        row_id = str(row_id_raw)

    return {
        "split": split,
        "row_id": row_id,
        "task_id": row.get("task_id", ""),
        "company": row.get("company", ""),
        "question": row.get("question", ""),
        "reference_answer": reference_answer,
        "prediction": prediction,
        "prompt_mode": prompt_mode,
        "task_category": detect_task_category(str(row.get("question", ""))),
        "expected_grounding": detect_expected_grounding(
            str(row.get("question", "")), reference_answer
        ),
        "model_id": model_id,
        "provider": provider,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "prompt_chars": len(str(prompt_text)),
        "prediction_words": word_total,
        "length_compliant": word_total <= 100,
        "length_penalty": length_penalty,
        "has_citation_markers": has_citation_markers(prediction),
        "quote_grounding_coverage": quote_coverage,
        "news_language_count": row.get("news_language_count", 0),
        **rouge,
        "rouge1_f1_length_penalized": rouge["rouge1_f1"] * length_penalty,
    }


def _safe_col(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Return df[col] if it exists, else a Series filled with *default*."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def aggregate_experiment_metrics(results: pd.DataFrame) -> dict:
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        return {
            "overall": {},
            "by_split": [],
            "by_company": [],
            "by_task_category": [],
        }

    results = results.copy()

    # ── Ensure ROUGE columns exist ────────────────────────────────────────────
    if "rouge1_f1" not in results.columns:
        if {"prediction", "reference_answer"} <= set(results.columns):
            rouge_cols = results.apply(
                lambda r: pd.Series(rouge_1_scores(r["reference_answer"], r["prediction"])),
                axis=1,
            )
            results = pd.concat(
                [results.reset_index(drop=True), rouge_cols.reset_index(drop=True)],
                axis=1,
            )
        else:
            warnings.warn(
                "Column 'rouge1_f1' missing and no prediction/reference present — filling zeros.",
                UserWarning,
            )
            results["rouge1_precision"] = 0.0
            results["rouge1_recall"] = 0.0
            results["rouge1_f1"] = 0.0

    # Guarantee all three ROUGE sub-columns exist
    for col in ("rouge1_precision", "rouge1_recall", "rouge1_f1"):
        if col not in results.columns:
            results[col] = 0.0

    # ── Ensure length penalty column exists ───────────────────────────────────
    if "length_penalty" not in results.columns:
        if "prediction" in results.columns:
            results["length_penalty"] = results["prediction"].astype(str).apply(compute_length_penalty)
        else:
            results["length_penalty"] = 1.0

    # ── Ensure penalised ROUGE column exists ──────────────────────────────────
    if "rouge1_f1_length_penalized" not in results.columns:
        results["rouge1_f1_length_penalized"] = (
            results["rouge1_f1"].astype(float).fillna(0.0)
            * results["length_penalty"].astype(float).fillna(1.0)
        )

    # ── Ensure grouping columns exist ─────────────────────────────────────────
    for col in ("split", "company", "task_category"):
        if col not in results.columns:
            results[col] = "unknown"

    overall_scores = results["rouge1_f1_length_penalized"].tolist()
    ci_low, ci_high = bootstrap_confidence_interval(overall_scores)

    overall = {
        "rows": int(len(results)),
        "mean_rouge1_f1": float(results["rouge1_f1"].astype(float).mean()),
        "mean_penalized_rouge1_f1": float(results["rouge1_f1_length_penalized"].astype(float).mean()),
        "mean_rouge1_precision": float(results["rouge1_precision"].astype(float).mean()),
        "mean_rouge1_recall": float(results["rouge1_recall"].astype(float).mean()),
        "length_compliance_rate": float(
            _safe_col(results, "length_compliant", False).astype(bool).mean()
        ),
        "citation_marker_rate": float(
            _safe_col(results, "has_citation_markers", False).astype(bool).mean()
        ),
        "quote_grounding_coverage_mean": float(
            _safe_col(results, "quote_grounding_coverage", 0.0).astype(float).mean()
        ),
        "bootstrap_ci_penalized_rouge1_f1": [float(ci_low), float(ci_high)],
    }

    def grouped_records(column: str) -> list[dict[str, object]]:
        rows_out = []
        for key, frame in results.groupby(column):
            low, high = bootstrap_confidence_interval(
                frame["rouge1_f1_length_penalized"].tolist()
            )
            rows_out.append(
                {
                    column: key,
                    "rows": int(len(frame)),
                    "mean_penalized_rouge1_f1": float(
                        frame["rouge1_f1_length_penalized"].mean()
                    ),
                    "length_compliance_rate": float(
                        _safe_col(frame, "length_compliant", False).astype(bool).mean()
                    ),
                    "citation_marker_rate": float(
                        _safe_col(frame, "has_citation_markers", False).astype(bool).mean()
                    ),
                    "bootstrap_ci_low": float(low),
                    "bootstrap_ci_high": float(high),
                }
            )
        return sorted(
            rows_out,
            key=lambda item: (-item["mean_penalized_rouge1_f1"], str(item[column])),
        )

    return {
        "overall": overall,
        "by_split": grouped_records("split"),
        "by_company": grouped_records("company"),
        "by_task_category": grouped_records("task_category"),
    }


def build_error_analysis_template(
    results: pd.DataFrame, sample_size: int = 50
) -> pd.DataFrame:
    """
    Build a CSV template for manual error analysis.

    Samples up to *sample_size* rows from the worst-performing predictions
    (lowest penalised ROUGE-1 F1) and adds empty annotation columns.
    Returns an empty DataFrame if *results* is empty.
    """
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        return pd.DataFrame()

    results = results.copy()

    # ── Ensure a stable row_id column ────────────────────────────────────────
    if "row_id" not in results.columns:
        if (
            "row_index" in results.columns
            and {"dataset_name", "split_name"} <= set(results.columns)
        ):
            results["row_id"] = (
                results["dataset_name"].astype(str)
                + "|"
                + results["split_name"].astype(str)
                + "|"
                + results["row_index"].astype(str)
            )
        elif "example_id" in results.columns:
            results["row_id"] = results["example_id"].astype(str)
        else:
            results["row_id"] = results.index.astype(str)

    # ── Ensure all needed columns exist with safe defaults ───────────────────
    for col, default in [
        ("rouge1_f1_length_penalized", 0.0),
        ("rouge1_f1", 0.0),
        ("length_compliant", False),
        ("prediction", ""),
        ("raw_response", ""),
        ("reference_answer", ""),
        ("question", ""),
        ("company", ""),
        ("split", ""),
        ("task_category", ""),
        ("has_citation_markers", False),
    ]:
        if col not in results.columns:
            results[col] = default

    # ── Sample the worst predictions ─────────────────────────────────────────
    sorted_results = results.sort_values(
        "rouge1_f1_length_penalized", ascending=True
    ).reset_index(drop=True)

    n = min(sample_size, len(sorted_results))
    sampled = sorted_results.head(n).copy()

    # ── Add annotation columns for human review ───────────────────────────────
    sampled["primary_error_label"] = ""
    sampled["secondary_error_label"] = ""
    sampled["annotator_notes"] = ""
    sampled["is_hallucination"] = ""
    sampled["is_length_violation"] = sampled["length_compliant"].apply(
        lambda v: "yes" if not bool(v) else ""
    )

    output_cols = [
        "row_id",
        "split",
        "company",
        "task_category",
        "question",
        "reference_answer",
        "prediction",
        "raw_response",
        "rouge1_f1",
        "rouge1_f1_length_penalized",
        "length_compliant",
        "has_citation_markers",
        "primary_error_label",
        "secondary_error_label",
        "is_hallucination",
        "is_length_violation",
        "annotator_notes",
    ]
    # Only keep columns that actually exist after the above defaults
    output_cols = [c for c in output_cols if c in sampled.columns]
    return sampled[output_cols].reset_index(drop=True)


def paired_prompt_comparison(results: pd.DataFrame) -> list[dict[str, object]]:
    if results is None or results.empty or "prompt_mode" not in results.columns:
        return []
    if results["prompt_mode"].nunique() < 2:
        return []
    pivot = results.pivot_table(
        index=["split", "row_id", "task_id"],
        columns="prompt_mode",
        values="rouge1_f1_length_penalized",
        aggfunc="first",
    ).dropna()
    if {"curated", "raw"} - set(pivot.columns):
        return []
    deltas = (pivot["curated"] - pivot["raw"]).tolist()
    ci_low, ci_high = bootstrap_confidence_interval(deltas)
    return [
        {
            "comparison": "curated_vs_raw",
            "rows": int(len(pivot)),
            "mean_delta_penalized_rouge1_f1": float(mean(deltas) if deltas else 0.0),
            "bootstrap_ci_low": float(ci_low),
            "bootstrap_ci_high": float(ci_high),
        }
    ]


def experiment_root() -> Path:
    root = ARTIFACTS_DIR / "predictions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_run_dir(model_id: str, run_name: str | None = None) -> Path:
    suffix = slugify(run_name) if run_name else slugify(model_id)
    run_dir = experiment_root() / f"{utc_timestamp()}__{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: pd.DataFrame | list[dict] | None) -> None:
    """Write rows (DataFrame or list-of-dicts) to a CSV file safely."""
    if rows is None:
        return
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return
        rows.to_csv(path, index=False)
    else:
        # rows is a list (or other non-DataFrame iterable)
        if not rows:
            return
        pd.DataFrame(rows).to_csv(path, index=False)


def list_experiment_runs() -> list[Path]:
    root = experiment_root()
    return sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)


def load_run_results(run_dir: Path) -> pd.DataFrame:
    # Try the consolidated CSV first (written by evaluate.py)
    csv_path = run_dir / "all_predictions.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    # Fall back to per-split JSONL files
    result_files = sorted(run_dir.glob("*_predictions.jsonl"))
    records: list[dict[str, object]] = []
    for result_file in result_files:
        with result_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(records)


def load_run_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def environment_metadata() -> dict[str, object]:
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "timestamp_utc": utc_timestamp(),
    }


def save_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_run_report(
    config_payload: dict,
    metrics: dict,
    dataset_summary: dict | pd.DataFrame,
) -> str:
    """Render a Markdown experiment report."""
    overall = metrics.get("overall", {})

    # ── Config section ────────────────────────────────────────────────────────
    config_lines = ["## Configuration\n"]
    for key, value in config_payload.items():
        config_lines.append(f"- **{key}**: `{value}`")
    config_md = "\n".join(config_lines)

    # ── Metrics section ───────────────────────────────────────────────────────
    metrics_lines = ["## Overall Metrics\n"]
    for key, value in overall.items():
        metrics_lines.append(f"- **{key}**: `{value}`")

    by_split = metrics.get("by_split", [])
    if by_split:
        try:
            split_md = pd.DataFrame(by_split).to_markdown(index=False)
            metrics_lines.append("\n### By Split\n")
            metrics_lines.append(split_md or "")
        except Exception:
            pass

    metrics_md = "\n".join(metrics_lines)

    # ── Dataset summary section ───────────────────────────────────────────────
    try:
        if isinstance(dataset_summary, pd.DataFrame):
            ds_df = dataset_summary
        elif isinstance(dataset_summary, dict):
            ds_df = pd.DataFrame(
                [{"split": k, "count": v} for k, v in dataset_summary.items()]
            )
        else:
            ds_df = pd.DataFrame(list(dataset_summary))
    except Exception:
        ds_df = pd.DataFrame()

    summary_md = "## Dataset Summary\n\n"
    if not ds_df.empty:
        try:
            summary_md += ds_df.to_markdown(index=False) or ""
        except Exception:
            summary_md += str(dataset_summary)

    # ── Assemble ──────────────────────────────────────────────────────────────
    return "\n\n".join(["# Experiment Summary", config_md, metrics_md, summary_md])


def load_error_analysis_table(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "error_analysis_template.csv"
    if not path.exists():
        return pd.DataFrame()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(path)


def save_error_analysis_table(run_dir: Path, frame: pd.DataFrame) -> Path:
    path = run_dir / "error_analysis_template.csv"
    frame.to_csv(path, index=False)
    return path


def error_label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "primary_error_label" not in frame.columns:
        return pd.DataFrame(columns=["primary_error_label", "rows"])
    cleaned = frame.copy()
    cleaned["primary_error_label"] = (
        cleaned["primary_error_label"].fillna("").astype(str).str.strip()
    )
    cleaned = cleaned[cleaned["primary_error_label"] != ""]
    if cleaned.empty:
        return pd.DataFrame(columns=["primary_error_label", "rows"])
    return (
        cleaned["primary_error_label"]
        .value_counts()
        .rename_axis("primary_error_label")
        .reset_index(name="rows")
    )
