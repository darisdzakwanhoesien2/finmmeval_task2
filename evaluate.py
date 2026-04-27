from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data_utils import build_prompt_text, DATA_DIR, NOTES_PATH   # ← build_prompt_text already imported in the file
from src.hf_streaming import DEFAULT_MODEL, stream_chat_completion
from src.research_utils import (
    aggregate_experiment_metrics,
    build_error_analysis_template,
    build_run_report,
    create_run_dir,
    environment_metadata,
    evaluate_prediction,
    load_enriched_split,
    paired_prompt_comparison,
    save_markdown,
    summarize_all_splits,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a financial QA assistant. Follow the user's instructions exactly, stay grounded in the provided "
    "financial statements and multilingual news, and keep the response concise."
)


@dataclass
class EvaluationConfig:
    token: str
    model: str
    provider: str = "huggingface"
    hf_provider_hint: str = "auto"                 # ← NEW
    prompt_mode: str = "curated"
    split: str = "all"
    limit: int | None = None
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    run_name: str = "task2-run"
    sample_size: int = 12
    skip_task_ids: set[str] = field(default_factory=set)   # ← NEW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible PolyFiQA evaluations against a hosted Hugging Face model.")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    parser.add_argument("--model", default=os.getenv("HF_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--provider", default=os.getenv("HF_PROVIDER", "hf-inference"))
    parser.add_argument("--prompt-mode", choices=["curated", "raw"], default="curated")
    parser.add_argument("--split", choices=["polyfiqa-easy", "polyfiqa-expert", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--sample-size", type=int, default=12)
    return parser.parse_args()


def selected_splits(split_arg: str) -> list[str]:
    if split_arg == "all":
        return ["polyfiqa-easy", "polyfiqa-expert"]
    return [split_arg]


# ── NEW: internal split iterator ──────────────────────────────────────────────

def _iter_splits(config: "EvaluationConfig"):
    """Yield (split_name, split_df) for every split requested in *config*."""
    for split_name in selected_splits(config.split):
        split_df = load_enriched_split(split_name)
        if config.limit is not None:
            split_df = split_df.head(config.limit).reset_index(drop=True)
        yield split_name, split_df


# ── NEW: single-row evaluator using the existing generate_prediction helper ───

def _evaluate_single_row(row: "pd.Series", config: "EvaluationConfig") -> dict:
    import time, sys
    from pathlib import Path
    from datetime import datetime, timezone
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.model_provider import call_chat_completion
    import inspect as _inspect

    record = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    # build prompt (previous logic, unchanged)
    # prompt = build_prompt_text(...) or dynamic inspector earlier
    prompt = build_prompt_text(record)  # keep as-is

    started = time.perf_counter()
    try:
        raw_response = call_chat_completion(
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user",   "content": prompt},
            ],
            model_id=config.model,
            provider=config.provider,
            api_key=config.token,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            hf_provider_hint=getattr(config, "hf_provider_hint", "auto"),
        )
        error_flag = False
    except Exception as exc:
        raw_response = f"[request_error] {exc}"
        error_flag = True
    latency_s = time.perf_counter() - started

    # --- Robust call to evaluate_prediction: inspect signature and pass available kwargs ---
    metrics = {}
    try:
        sig = _inspect.signature(evaluate_prediction)
        # build candidate kwargs from local context / config
        candidate_kwargs = {
            "split":         str(record.get("split", "")),
            "row":           record,
            "prediction":    raw_response,
            "prompt_mode":   getattr(config, "prompt_mode", None),
            "model_id":      getattr(config, "model", None),
            "provider":      getattr(config, "provider", None),
            "max_tokens":    getattr(config, "max_tokens", None),
            "temperature":   getattr(config, "temperature", None),
            "system_prompt": getattr(config, "system_prompt", None),
        }
        # filter to params that actually exist in the function signature
        filtered = {
            name: val for name, val in candidate_kwargs.items() if name in sig.parameters
        }
        if len(sig.parameters) == 0:
            # older versions may expect no args
            metrics = evaluate_prediction()
        else:
            metrics = evaluate_prediction(**filtered)
    except TypeError:
        # fallback: try calling with (record, raw_response) if older variant
        try:
            metrics = evaluate_prediction(record, raw_response)
        except Exception:
            metrics = {}
    except Exception:
        metrics = {}

    return {
        "task_id":          str(record.get("task_id", "")),
        "split":            str(record.get("split", "")),
        "model":            config.model,
        "provider":         config.provider,
        "prompt_mode":      config.prompt_mode,
        "raw_response":     raw_response,
        "error":            error_flag,
        "latency_seconds":  latency_s,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        **{k: record.get(k, "") for k in ("company", "task_category", "language", "question")},
        **(metrics or {}),
    }


def split_name_from_row(record: dict) -> str:
    """Best-effort: infer split name from a record dict."""
    return str(record.get("split", record.get("source", "")))


# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    config: "EvaluationConfig",
    progress_callback=None,
) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if config.provider not in {"lmstudio", "lm_studio", "local"} and not config.token:
        raise ValueError("Missing API token.")

    run_dir = create_run_dir(config.model, config.run_name)

    # Serialize config safely — convert set/frozenset → sorted list
    config_payload = {
        k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
        for k, v in vars(config).items()
    }
    write_json(run_dir / "config.json", config_payload)

    all_records: list[dict] = []
    dataset_summary: dict = {}

    for split_name, split_df in _iter_splits(config):
        dataset_summary[split_name] = len(split_df)

        rows_to_run = [
            (idx, row) for idx, row in split_df.iterrows()
            if str(row.get("task_id", idx)) not in config.skip_task_ids
        ]
        total = len(rows_to_run)

        for processed, (idx, row) in enumerate(rows_to_run, start=1):
            row = row.copy()
            row["split"] = split_name
            record = _evaluate_single_row(row=row, config=config)
            all_records.append(record)

            if progress_callback is not None:
                progress_callback(
                    processed,
                    total,
                    message=f"{split_name} row {processed}/{total}",
                    completed_record=record,
                )

    results_frame = pd.DataFrame(all_records)
    metrics = aggregate_experiment_metrics(results_frame)
    write_json(run_dir / "metrics.json", metrics)

    error_template = build_error_analysis_template(results_frame, sample_size=config.sample_size)
    write_csv(run_dir / "error_analysis_template.csv", error_template)
    results_frame.to_csv(run_dir / "all_predictions.csv", index=False)

    report = build_run_report(config_payload, metrics, dataset_summary)
    save_markdown(run_dir / "report.md", report)

    summary_lines = [
        f"Saved experiment artifacts to: {run_dir}",
        f"Rows evaluated: {len(results_frame)}",
        f"Mean penalized ROUGE-1 F1: {metrics.get('overall', {}).get('mean_penalized_rouge1_f1', 0.0):.4f}",
        f"Length compliance rate: {metrics.get('overall', {}).get('length_compliance_rate', 0.0):.4f}",
    ]
    (run_dir / "README.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "run_dir":          run_dir,
        "results":          results_frame,
        "predictions":      results_frame,   # alias used by app.py fallback flush
        "metrics":          metrics,
        "dataset_summary":  dataset_summary,
        "summary_lines":    summary_lines,
    }


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        token=args.token,
        model=args.model,
        provider=args.provider,
        prompt_mode=args.prompt_mode,
        split=args.split,
        limit=args.limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        run_name=args.run_name,
        sample_size=args.sample_size,
    )
    outcome = run_evaluation(config)
    print("\n".join(outcome["summary_lines"]))


if __name__ == "__main__":
    main()
