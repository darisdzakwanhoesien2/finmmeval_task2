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


def _format_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_score(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No records available."

    display = frame.fillna("").copy()
    headers = [str(column) for column in display.columns]

    def render_cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    rows = [
        [render_cell(value) for value in row]
        for row in display.astype(object).itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _records_table(records: list[dict[str, object]], max_rows: int = 8) -> str:
    if not records:
        return "No records available."
    frame = pd.DataFrame(records).head(max_rows)
    return _markdown_table(frame)


def _safe_config_value(config: dict[str, object], key: str, default: str = "not recorded") -> str:
    value = config.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


def build_results_discussion_draft(
    *,
    run_name: str,
    config: dict[str, object],
    results: pd.DataFrame,
    metrics: dict,
    error_table: pd.DataFrame | None = None,
) -> str:
    """Create a paper-oriented Results and Discussion draft from one run."""
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    by_split = metrics.get("by_split", []) if isinstance(metrics, dict) else []
    by_category = metrics.get("by_task_category", []) if isinstance(metrics, dict) else []
    by_company = metrics.get("by_company", []) if isinstance(metrics, dict) else []

    rows = int(overall.get("rows", len(results) if results is not None else 0) or 0)
    ci = overall.get("bootstrap_ci_penalized_rouge1_f1", [])
    ci_text = "n/a"
    if isinstance(ci, list) and len(ci) == 2:
        ci_text = f"[{_format_score(ci[0])}, {_format_score(ci[1])}]"

    split_values: list[str] = []
    if results is not None and not results.empty and "split" in results.columns:
        split_values = [
            str(value)
            for value in results["split"].dropna().unique().tolist()
            if str(value).strip()
        ]
    split_note = ""
    if not split_values:
        split_note = (
            "\n\n> Note: this run does not contain saved split labels in `all_predictions.csv`, "
            "so Easy-vs-Expert claims should be avoided until the evaluation is rerun with split metadata."
        )

    best_category = by_category[0] if by_category else {}
    weakest_category = by_category[-1] if by_category else {}
    best_company = by_company[0] if by_company else {}
    weakest_company = by_company[-1] if by_company else {}

    error_summary_md = "No manually labeled error-analysis rows are available yet."
    if error_table is not None and not error_table.empty:
        summary = error_label_summary(error_table)
        if not summary.empty:
            error_summary_md = _markdown_table(summary)

    model = _safe_config_value(config, "model", _safe_config_value(config, "model_id"))
    provider = _safe_config_value(config, "provider")
    prompt_mode = _safe_config_value(config, "prompt_mode")
    temperature = _safe_config_value(config, "temperature")
    max_tokens = _safe_config_value(config, "max_tokens")

    return f"""# Results and Discussion Draft

Generated from run `{run_name}`.

## Results

The experiment evaluated `{rows}` PolyFiQA predictions using `{model}` through `{provider}` with `{prompt_mode}` prompts, `temperature={temperature}`, and `max_tokens={max_tokens}`. The primary evaluation metric was ROUGE-1 F1, with an additional length-penalized ROUGE-1 score to reflect the task requirement that answers remain within 100 words.

Overall, the model achieved a mean ROUGE-1 F1 of `{_format_score(overall.get("mean_rouge1_f1"))}` and a mean length-penalized ROUGE-1 F1 of `{_format_score(overall.get("mean_penalized_rouge1_f1"))}`. The bootstrap confidence interval for the penalized score was `{ci_text}`. The answer-length compliance rate was `{_format_percent(overall.get("length_compliance_rate"))}`, while the citation-marker rate was `{_format_percent(overall.get("citation_marker_rate"))}` and the mean quote-grounding coverage was `{_format_percent(overall.get("quote_grounding_coverage_mean"))}`.{split_note}

### Breakdown by Split

{_records_table(by_split)}

### Breakdown by Task Category

{_records_table(by_category)}

### Breakdown by Company

{_records_table(by_company)}

## Discussion

The results suggest that answer-format compliance and lexical overlap capture different parts of the system behavior. A high length-compliance rate indicates that the generation setup can respect the short-answer constraint, but the ROUGE-1 scores show how closely the model output matches the expert reference wording. Because PolyFiQA requires cross-lingual evidence grounding across SEC filings and multilingual news, low lexical overlap should be interpreted cautiously: it may reflect genuinely different phrasing, missing evidence, or unsupported reasoning.

The strongest task category in this run was `{best_category.get("task_category", "not available")}` with a mean penalized ROUGE-1 F1 of `{_format_score(best_category.get("mean_penalized_rouge1_f1"))}`. The weakest category was `{weakest_category.get("task_category", "not available")}` with a mean penalized ROUGE-1 F1 of `{_format_score(weakest_category.get("mean_penalized_rouge1_f1"))}`. This gap is useful for the Results section because it identifies where the system is most and least aligned with the reference answers.

At the company level, `{best_company.get("company", "not available")}` had the highest mean penalized score (`{_format_score(best_company.get("mean_penalized_rouge1_f1"))}`), while `{weakest_company.get("company", "not available")}` had the lowest (`{_format_score(weakest_company.get("mean_penalized_rouge1_f1"))}`). These differences may reflect variation in source-document clarity, topic complexity, or how much relevant information appears in multilingual news versus the English filing.

The citation-marker and quote-grounding measures are especially important for PolyFiQA because the benchmark is not only asking for short answers, but also for grounded financial reasoning. If these values are low, the Discussion should state that the model may be producing fluent answers without making evidence use explicit. A stronger final study should pair ROUGE-1 with manual error analysis, especially for unsupported claims, missing multilingual evidence, and incorrect financial comparisons.

### Error Analysis Summary

{error_summary_md}

## Suggested Paper Text

In our PolyFiQA Task 2 experiment, the evaluated model produced concise answers with a length-compliance rate of `{_format_percent(overall.get("length_compliance_rate"))}`. However, the mean length-penalized ROUGE-1 F1 was `{_format_score(overall.get("mean_penalized_rouge1_f1"))}`, indicating limited unigram overlap with expert references. Performance varied across task categories, with `{best_category.get("task_category", "the strongest category")}` outperforming `{weakest_category.get("task_category", "the weakest category")}`. This pattern suggests that the model handles some financial reasoning questions more reliably than others, but still struggles to consistently reproduce reference-grounded analytical answers.

These findings support the central difficulty of PolyFiQA: cross-lingual financial question answering requires more than extracting isolated facts from a single English filing. Strong answers must combine SEC report evidence with multilingual news signals and present the conclusion within a strict word budget. Future improvements should therefore focus on retrieval quality, explicit evidence attribution, and targeted error reduction for multi-document analytical questions.
"""


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
