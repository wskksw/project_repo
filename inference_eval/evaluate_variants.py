#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_eval.common import (
    DEFAULT_DATASET,
    EMOTIONS,
    INFERENCE_PATTERNS,
    DEFAULT_RUNS_DIR,
    build_ground_truth,
    ensure_dir,
    is_missing,
    load_dataset,
    load_predictions_file,
    normalize_prediction,
    majority_value,
    markdown_table,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extractor variants against merged ground truth.")
    parser.add_argument("--run-dir", default=DEFAULT_RUNS_DIR, help="Stable output directory created by run_variants.py.")
    parser.add_argument("--ground-truth", default=DEFAULT_DATASET, help="Workbook or CSV used as merged ground truth.")
    parser.add_argument("--include-variants", nargs="+", help="Optional subset of variant names to evaluate.")
    parser.add_argument("--exclude-variants", nargs="+", help="Variant names to skip while evaluating.")
    parser.add_argument(
        "--batch-summary-dirs",
        nargs="+",
        help="Optional OpenAI batch directories to summarize at the bottom of evaluation/metrics_summary.md.",
    )
    parser.add_argument(
        "--batch-sample-limit",
        type=int,
        default=4,
        help="Number of parsed batch samples to include per batch summary section.",
    )
    return parser.parse_args()


def discover_prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(path for path in predictions_dir.glob("*.jsonl") if path.is_file())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def parse_custom_id(custom_id: str) -> dict[str, str] | None:
    parts = str(custom_id).split("::")
    if len(parts) != 4:
        return None
    return {
        "variant_name": parts[0],
        "row_index": parts[1],
        "complaint_id": parts[2],
        "construct": parts[3],
    }


def extract_openai_output(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return response["output_text"]
    for block in response.get("output", []):
        if block.get("type") != "message":
            continue
        for item in block.get("content", []):
            text = item.get("text")
            if text:
                return text
    raise RuntimeError("Could not find output text in OpenAI response.")


def parse_json_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise RuntimeError("Model returned empty text.")

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(candidate.strip() for candidate in fenced if candidate.strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"[{[]", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise RuntimeError(f"Could not parse model output as JSON object: {raw_text[:500]}")


def compact_json(value: Any, max_len: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def construct_sort_key(construct: str) -> tuple[int, str]:
    order = {"c1": 0, "c2": 1, "c3": 2, "c4": 3}
    return (order.get(str(construct), 999), str(construct))


def select_batch_samples(samples: list[dict[str, Any]], sample_limit: int) -> list[dict[str, Any]]:
    if sample_limit <= 0:
        return []
    chosen: list[dict[str, Any]] = []
    chosen_keys: set[tuple[str, str]] = set()
    seen_constructs: set[str] = set()

    for sample in samples:
        sample_key = (str(sample["complaint_id"]), str(sample["construct"]))
        if sample["construct"] in seen_constructs or sample_key in chosen_keys:
            continue
        chosen.append(sample)
        chosen_keys.add(sample_key)
        seen_constructs.add(str(sample["construct"]))
        if len(chosen) >= sample_limit:
            return chosen

    for sample in samples:
        sample_key = (str(sample["complaint_id"]), str(sample["construct"]))
        if sample_key in chosen_keys:
            continue
        chosen.append(sample)
        chosen_keys.add(sample_key)
        if len(chosen) >= sample_limit:
            return chosen
    return chosen


def build_batch_summary_section(batch_dir: Path, sample_limit: int) -> str:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Batch summary requested, but manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    request_index_lookup = {
        str(item["custom_id"]): item
        for item in load_jsonl(batch_dir / "request_index.jsonl")
        if item.get("custom_id")
    }
    output_records = load_jsonl(batch_dir / "output.jsonl")
    error_records = load_jsonl(batch_dir / "errors.jsonl")
    all_records = output_records + error_records

    counts = Counter()
    incomplete_reasons: Counter[str] = Counter()
    per_construct: dict[str, Counter[str]] = defaultdict(Counter)
    sample_candidates: list[dict[str, Any]] = []

    for record in all_records:
        counts["records_seen"] += 1
        metadata = parse_custom_id(record.get("custom_id", "")) or {}
        construct = str(metadata.get("construct") or "unknown")
        complaint_id = str(metadata.get("complaint_id") or "")
        per_construct[construct]["records_seen"] += 1

        response = record.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code") == 200:
            counts["http_200"] += 1
            per_construct[construct]["http_200"] += 1
        if record.get("error") or body.get("error"):
            counts["error_records"] += 1
            per_construct[construct]["error_records"] += 1

        status = body.get("status")
        if status:
            counts[f"status_{status}"] += 1
            per_construct[construct][f"status_{status}"] += 1
        if status == "incomplete":
            reason = str((body.get("incomplete_details") or {}).get("reason") or "unknown")
            incomplete_reasons[reason] += 1

        try:
            raw_text = extract_openai_output(body)
        except Exception:  # noqa: BLE001
            raw_text = None

        if raw_text is None:
            continue

        counts["with_output_text"] += 1
        per_construct[construct]["with_output_text"] += 1
        try:
            parsed = parse_json_response(raw_text)
            normalized = normalize_prediction(construct, parsed)
        except Exception:  # noqa: BLE001
            continue

        counts["parseable_json"] += 1
        per_construct[construct]["parseable_json"] += 1
        request_item = request_index_lookup.get(record.get("custom_id", ""), {})
        sample_candidates.append(
            {
                "complaint_id": complaint_id,
                "construct": construct,
                "retrieved_complaint_ids": request_item.get("retrieved_complaint_ids") or [],
                "normalized_prediction": normalized,
            }
        )

    construct_rows: list[list[Any]] = []
    for construct in sorted(per_construct, key=construct_sort_key):
        stats = per_construct[construct]
        construct_rows.append(
            [
                construct,
                stats.get("records_seen", 0),
                stats.get("status_completed", 0),
                stats.get("status_incomplete", 0),
                stats.get("with_output_text", 0),
                stats.get("parseable_json", 0),
            ]
        )

    sample_candidates = sorted(
        sample_candidates,
        key=lambda item: (construct_sort_key(str(item["construct"])), str(item["complaint_id"])),
    )
    selected_samples = select_batch_samples(sample_candidates, sample_limit)
    sample_rows = [
        [
            sample["complaint_id"],
            sample["construct"],
            ", ".join(str(item) for item in sample["retrieved_complaint_ids"]) or "NA",
            markdown_text(compact_json(sample["normalized_prediction"], max_len=140)),
        ]
        for sample in selected_samples
    ]

    lines = [
        f"## Batch Summary: `{batch_dir.name}`",
        "",
        f"- Variant: `{manifest.get('variant_name', batch_dir.name)}`",
        f"- Model: `{manifest.get('model', 'unknown')}`",
        f"- Prompt strategy: `{manifest.get('prompt_strategy', 'unknown')}`",
        f"- Requests in manifest: `{manifest.get('request_count', 'unknown')}`",
        f"- Output records seen: `{counts.get('records_seen', 0)}`",
        f"- Completed responses: `{counts.get('status_completed', 0)}`",
        f"- Incomplete responses: `{counts.get('status_incomplete', 0)}`",
        f"- Responses with output text: `{counts.get('with_output_text', 0)}`",
        f"- Parseable JSON outputs: `{counts.get('parseable_json', 0)}`",
    ]
    if incomplete_reasons:
        reason_summary = ", ".join(
            f"`{reason}`: {count}" for reason, count in sorted(incomplete_reasons.items())
        )
        lines.append(f"- Incomplete reasons: {reason_summary}")

    if construct_rows:
        lines.extend(
            [
                "",
                "### By Construct",
                "",
                markdown_table(
                    ["Construct", "Records", "Completed", "Incomplete", "With Output Text", "Parseable JSON"],
                    construct_rows,
                ),
            ]
        )

    if sample_rows:
        lines.extend(
            [
                "",
                "### Parsed Samples",
                "",
                markdown_table(
                    ["Complaint ID", "Construct", "Retrieved Complaint IDs", "Normalized Prediction"],
                    sample_rows,
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def metric_row(variant: str, provider: str, model: str, strategy: str, construct: str, metric: str, value: float, n: int) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "provider": provider,
        "model": model,
        "prompt_strategy": strategy,
        "construct": construct,
        "metric": metric,
        "value": round(float(value), 4),
        "n": int(n),
    }


def evaluate_scalar(rows: list[dict[str, Any]], gt_key: str, pred_key: str, variant_meta: dict[str, str], construct: str, metric_name: str) -> list[dict[str, Any]]:
    paired = [
        (row[gt_key], row[pred_key])
        for row in rows
        if not is_missing(row.get(gt_key)) and not is_missing(row.get(pred_key))
    ]
    if not paired:
        return []
    correct = sum(1 for gt, pred in paired if gt == pred)
    return [
        metric_row(
            variant=variant_meta["variant_name"],
            provider=variant_meta["provider"],
            model=variant_meta["model"],
            strategy=variant_meta["prompt_strategy"],
            construct=construct,
            metric=metric_name,
            value=correct / len(paired),
            n=len(paired),
        )
    ]


def evaluate_amount(rows: list[dict[str, Any]], variant_meta: dict[str, str]) -> list[dict[str, Any]]:
    paired = [
        (row["gt_c2_abs_amount"], row["pred_c2_abs_amount"])
        for row in rows
        if not is_missing(row.get("gt_c2_abs_amount")) and not is_missing(row.get("pred_c2_abs_amount"))
    ]
    if not paired:
        return []
    correct = sum(1 for gt, pred in paired if gt == pred)
    return [
        metric_row(
            variant=variant_meta["variant_name"],
            provider=variant_meta["provider"],
            model=variant_meta["model"],
            strategy=variant_meta["prompt_strategy"],
            construct="c2",
            metric="abs_amount_exact_match",
            value=correct / len(paired),
            n=len(paired),
        )
    ]


def display_metric(metric_lookup: dict[tuple[str, str], float], construct: str, metric: str) -> str | float:
    value = metric_lookup.get((construct, metric))
    return "NA" if value is None else value


def evaluate_multilabel(
    rows: list[dict[str, Any]],
    gt_key: str,
    pred_key: str,
    labels: tuple[str, ...],
    variant_meta: dict[str, str],
    construct: str,
    prefix: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    tp = fp = fn = 0
    exact = 0
    for row in rows:
        gt_set = set(as_label_list(row.get(gt_key)))
        pred_set = set(as_label_list(row.get(pred_key)))
        if gt_set == pred_set:
            exact += 1
        for label in labels:
            gt_present = label in gt_set
            pred_present = label in pred_set
            if gt_present and pred_present:
                tp += 1
            elif pred_present and not gt_present:
                fp += 1
            elif gt_present and not pred_present:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return [
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], construct, f"{prefix}_precision", precision, len(rows)),
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], construct, f"{prefix}_recall", recall, len(rows)),
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], construct, f"{prefix}_f1", f1, len(rows)),
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], construct, f"{prefix}_exact_match", exact / len(rows), len(rows)),
    ]


def evaluate_c3(rows: list[dict[str, Any]], variant_meta: dict[str, str]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    metrics.extend(evaluate_scalar(rows, "gt_c3_code", "pred_c3_code", variant_meta, "c3", "code_accuracy"))
    metrics.extend(evaluate_multilabel(rows, "gt_c3_emotions", "pred_c3_emotions", EMOTIONS, variant_meta, "c3", "emotion_label"))
    positive_rows = [row for row in rows if not is_missing(row.get("gt_c3_code")) and row.get("gt_c3_code") != 0]
    metrics.extend(evaluate_scalar(positive_rows, "gt_c3_tier", "pred_c3_tier", variant_meta, "c3", "tier_accuracy_on_positive"))

    tier2_rows = [row for row in rows if row.get("gt_c3_tier") == 2]
    paired = [
        (tuple(as_label_list(row.get("gt_c3_inference_patterns"))), tuple(as_label_list(row.get("pred_c3_inference_patterns"))))
        for row in tier2_rows
        if not is_missing(row.get("pred_c3_inference_patterns")) or isinstance(row.get("pred_c3_inference_patterns"), list)
    ]
    if paired:
        correct = sum(1 for gt, pred in paired if gt == pred)
        metrics.append(
            metric_row(
                variant_meta["variant_name"],
                variant_meta["provider"],
                variant_meta["model"],
                variant_meta["prompt_strategy"],
                "c3",
                "pattern_exact_match_on_tier2",
                correct / len(paired),
                len(paired),
            )
        )
        metrics.extend(
            evaluate_multilabel(
                tier2_rows,
                "gt_c3_inference_patterns",
                "pred_c3_inference_patterns",
                INFERENCE_PATTERNS,
                variant_meta,
                "c3",
                "pattern_label_tier2",
            )
        )
    return metrics


def as_label_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return []


def make_majority_baseline(gt_df: pd.DataFrame) -> dict[str, Any]:
    c3_signature = majority_value(
        [
            (
                0 if is_missing(row["gt_c3_code"]) else row["gt_c3_code"],
                tuple(row["gt_c3_emotions"] or []),
                0 if is_missing(row["gt_c3_tier"]) else row["gt_c3_tier"],
                tuple(row["gt_c3_inference_patterns"] or []),
            )
            for _, row in gt_df.iterrows()
        ],
        (0, (), 0, ()),
    )
    return {
        "variant_name": "majority_baseline",
        "provider": "baseline",
        "model": "deterministic",
        "prompt_strategy": "majority_class",
        "pred_c0_label": majority_value(gt_df["gt_c0_label"].tolist(), 0),
        "pred_c1_score": majority_value(gt_df["gt_c1_score"].tolist(), 0),
        "pred_c2_abs_code": majority_value(gt_df["gt_c2_abs_code"].tolist(), 0),
        "pred_c2_abs_amount": majority_value(gt_df["gt_c2_abs_amount"].dropna().tolist(), "Unspecified"),
        "pred_c2_rel_score": majority_value(gt_df["gt_c2_rel_score"].tolist(), 0),
        "pred_c3_code": c3_signature[0],
        "pred_c3_emotions": list(c3_signature[1]),
        "pred_c3_tier": c3_signature[2],
        "pred_c3_inference_patterns": list(c3_signature[3]),
        "pred_c4_score": majority_value(gt_df["gt_c4_score"].tolist(), 0),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions_dir = run_dir / "predictions"
    evaluation_dir = ensure_dir(run_dir / "evaluation")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    include_variants = set(args.include_variants or [])
    exclude_variants = set(args.exclude_variants or [])

    gt_df = build_ground_truth(load_dataset(args.ground_truth))
    gt_lookup = gt_df.set_index("Complaint ID").to_dict(orient="index")

    all_rows: list[dict[str, Any]] = []
    for prediction_path in discover_prediction_files(predictions_dir):
        records = load_predictions_file(prediction_path)
        if not records:
            continue
        variant_name = str(records[0].get("variant_name") or prediction_path.stem)
        if include_variants and variant_name not in include_variants:
            continue
        if variant_name in exclude_variants:
            continue
        for record in records:
            complaint_id = record["complaint_id"]
            try:
                gt_key_int = int(float(complaint_id))
            except (TypeError, ValueError):
                gt_key_int = None
            gt = gt_lookup.get(gt_key_int) or gt_lookup.get(str(complaint_id))
            if not gt or record.get("error") or not record.get("normalized_prediction"):
                continue

            normalized = record["normalized_prediction"]
            row: dict[str, Any] = {
                "variant_name": record["variant_name"],
                "provider": record["provider"],
                "model": record["model"],
                "prompt_strategy": record["prompt_strategy"],
                "construct": record["construct"],
                "complaint_id": complaint_id,
                **gt,
            }
            if record["construct"] == "c1":
                row["pred_c0_label"] = normalized.get("c0_label")
                row["pred_c1_score"] = normalized.get("c1_score")
            elif record["construct"] == "c2":
                row["pred_c2_abs_code"] = normalized.get("c2_abs_code")
                row["pred_c2_abs_amount"] = normalized.get("c2_abs_amount")
                row["pred_c2_rel_score"] = normalized.get("c2_rel_score")
            elif record["construct"] == "c3":
                row["pred_c3_code"] = normalized.get("c3_code")
                row["pred_c3_emotions"] = normalized.get("c3_emotions", [])
                row["pred_c3_tier"] = normalized.get("c3_tier")
                row["pred_c3_inference_patterns"] = normalized.get("c3_inference_patterns", [])
            elif record["construct"] == "c4":
                row["pred_c4_score"] = normalized.get("c4_score")
            all_rows.append(row)

    metrics: list[dict[str, Any]] = []
    rows_df = pd.DataFrame(all_rows)
    if not rows_df.empty:
        for _, variant_rows in rows_df.groupby("variant_name"):
            first = variant_rows.iloc[0]
            variant_meta = {
                "variant_name": first["variant_name"],
                "provider": first["provider"],
                "model": first["model"],
                "prompt_strategy": first["prompt_strategy"],
            }
            c1_rows = variant_rows[variant_rows["construct"] == "c1"].to_dict(orient="records")
            c2_rows = variant_rows[variant_rows["construct"] == "c2"].to_dict(orient="records")
            c3_rows = variant_rows[variant_rows["construct"] == "c3"].to_dict(orient="records")
            c4_rows = variant_rows[variant_rows["construct"] == "c4"].to_dict(orient="records")

            metrics.extend(evaluate_scalar(c1_rows, "gt_c0_label", "pred_c0_label", variant_meta, "c1", "c0_label_accuracy"))
            metrics.extend(evaluate_scalar(c1_rows, "gt_c1_score", "pred_c1_score", variant_meta, "c1", "c1_score_accuracy"))
            metrics.extend(evaluate_scalar(c2_rows, "gt_c2_abs_code", "pred_c2_abs_code", variant_meta, "c2", "abs_code_accuracy"))
            metrics.extend(evaluate_scalar(c2_rows, "gt_c2_rel_score", "pred_c2_rel_score", variant_meta, "c2", "rel_score_accuracy"))
            metrics.extend(evaluate_amount(c2_rows, variant_meta))
            metrics.extend(evaluate_c3(c3_rows, variant_meta))
            metrics.extend(evaluate_scalar(c4_rows, "gt_c4_score", "pred_c4_score", variant_meta, "c4", "score_accuracy"))

    if not rows_df.empty:
        evaluated_ids = set(rows_df["complaint_id"].astype(str).tolist())
        gt_subset = gt_df[gt_df["Complaint ID"].astype(str).isin(evaluated_ids)].copy()
    else:
        gt_subset = gt_df.copy()

    baseline = make_majority_baseline(gt_subset)
    baseline_rows = []
    for _, row in gt_subset.iterrows():
        baseline_rows.append({**baseline, **row.to_dict(), "construct": "c1"})
        baseline_rows.append({**baseline, **row.to_dict(), "construct": "c2"})
        baseline_rows.append({**baseline, **row.to_dict(), "construct": "c3"})
        baseline_rows.append({**baseline, **row.to_dict(), "construct": "c4"})
    baseline_df = pd.DataFrame(baseline_rows)
    variant_meta = {
        "variant_name": baseline["variant_name"],
        "provider": baseline["provider"],
        "model": baseline["model"],
        "prompt_strategy": baseline["prompt_strategy"],
    }
    metrics.extend(evaluate_scalar(baseline_df[baseline_df["construct"] == "c1"].to_dict(orient="records"), "gt_c0_label", "pred_c0_label", variant_meta, "c1", "c0_label_accuracy"))
    metrics.extend(evaluate_scalar(baseline_df[baseline_df["construct"] == "c1"].to_dict(orient="records"), "gt_c1_score", "pred_c1_score", variant_meta, "c1", "c1_score_accuracy"))
    c2_rows = baseline_df[baseline_df["construct"] == "c2"].to_dict(orient="records")
    metrics.extend(evaluate_scalar(c2_rows, "gt_c2_abs_code", "pred_c2_abs_code", variant_meta, "c2", "abs_code_accuracy"))
    metrics.extend(evaluate_scalar(c2_rows, "gt_c2_rel_score", "pred_c2_rel_score", variant_meta, "c2", "rel_score_accuracy"))
    metrics.extend(evaluate_amount(c2_rows, variant_meta))
    metrics.extend(evaluate_c3(baseline_df[baseline_df["construct"] == "c3"].to_dict(orient="records"), variant_meta))
    metrics.extend(evaluate_scalar(baseline_df[baseline_df["construct"] == "c4"].to_dict(orient="records"), "gt_c4_score", "pred_c4_score", variant_meta, "c4", "score_accuracy"))

    metrics_df = pd.DataFrame(metrics).sort_values(["variant_name", "construct", "metric"])
    metrics_csv = evaluation_dir / "metrics_by_variant.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary_rows = []
    for variant_name, group in metrics_df.groupby("variant_name"):
        metric_lookup = {(row["construct"], row["metric"]): row["value"] for _, row in group.iterrows()}
        summary_rows.append(
            [
                variant_name,
                group.iloc[0]["provider"],
                group.iloc[0]["model"],
                group.iloc[0]["prompt_strategy"],
                display_metric(metric_lookup, "c1", "c0_label_accuracy"),
                display_metric(metric_lookup, "c1", "c1_score_accuracy"),
                display_metric(metric_lookup, "c2", "abs_code_accuracy"),
                display_metric(metric_lookup, "c2", "rel_score_accuracy"),
                display_metric(metric_lookup, "c3", "code_accuracy"),
                display_metric(metric_lookup, "c3", "emotion_label_f1"),
                display_metric(metric_lookup, "c4", "score_accuracy"),
            ]
        )
    summary_md = markdown_table(
        [
            "Variant",
            "Provider",
            "Model",
            "Prompt Strategy",
            "C0 Acc",
            "C1 Acc",
            "C2 Abs Acc",
            "C2 Rel Acc",
            "C3 Code Acc",
            "C3 Emotion F1",
            "C4 Acc",
        ],
        summary_rows,
    )
    summary_path = evaluation_dir / "metrics_summary.md"
    summary_text = (
        "# Variant Evaluation Summary\n\n"
        "The first call bundles Construct 0 and Construct 1 so the run still uses four model calls per narrative.\n\n"
        "`NA` means the evaluator found no scorable ground-truth/prediction pairs for that metric.\n\n"
        f"{summary_md}\n"
    )
    if args.batch_summary_dirs:
        for batch_summary_dir in args.batch_summary_dirs:
            summary_text += "\n" + build_batch_summary_section(Path(batch_summary_dir), args.batch_sample_limit)

    summary_path.write_text(summary_text)

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
