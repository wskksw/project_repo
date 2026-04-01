#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_eval.common import (
    EMOTIONS,
    DEFAULT_DATASET,
    build_ground_truth,
    ensure_dir,
    load_predictions_file,
    majority_value,
    markdown_table,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extractor variants against provisional ground truth.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by run_variants.py.")
    parser.add_argument("--ground-truth", default=DEFAULT_DATASET, help="CSV used as provisional ground truth.")
    return parser.parse_args()


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
    paired = [(row[gt_key], row[pred_key]) for row in rows if row.get(gt_key) is not None and row.get(pred_key) is not None]
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
        if row.get("gt_c2_abs_amount") is not None and row.get("pred_c2_abs_amount") is not None
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


def evaluate_c3(rows: list[dict[str, Any]], variant_meta: dict[str, str]) -> list[dict[str, Any]]:
    tp = fp = fn = 0
    exact = 0
    tier_total = tier_correct = 0
    pattern_total = pattern_correct = 0

    for row in rows:
        gt_emotions = row.get("gt_c3_emotions", {})
        pred_emotions = row.get("pred_c3_emotions", {})
        gt_tiers = row.get("gt_c3_tiers", {})
        pred_tiers = row.get("pred_c3_tiers", {})
        gt_patterns = row.get("gt_c3_patterns", {})
        pred_patterns = row.get("pred_c3_patterns", {})

        row_exact = True
        for emotion in EMOTIONS:
            gt_present = int(gt_emotions.get(emotion, 0) or 0)
            pred_present = int(pred_emotions.get(emotion, 0) or 0)
            if gt_present == 1 and pred_present == 1:
                tp += 1
            elif gt_present == 0 and pred_present == 1:
                fp += 1
                row_exact = False
            elif gt_present == 1 and pred_present == 0:
                fn += 1
                row_exact = False
            if gt_present != pred_present:
                row_exact = False

            gt_tier = int(gt_tiers.get(emotion, 0) or 0)
            pred_tier = int(pred_tiers.get(emotion, 0) or 0)
            if gt_present == 1:
                tier_total += 1
                if gt_tier == pred_tier:
                    tier_correct += 1

            gt_pattern = str(gt_patterns.get(emotion, "none") or "none")
            pred_pattern = str(pred_patterns.get(emotion, "none") or "none")
            if gt_tier == 2:
                pattern_total += 1
                if gt_pattern == pred_pattern:
                    pattern_correct += 1

        if row_exact:
            exact += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    metrics = [
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], "c3", "emotion_presence_precision", precision, len(rows)),
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], "c3", "emotion_presence_recall", recall, len(rows)),
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], "c3", "emotion_presence_f1", f1, len(rows)),
        metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], "c3", "emotion_exact_match", exact / len(rows) if rows else 0.0, len(rows)),
    ]
    if tier_total:
        metrics.append(
            metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], "c3", "tier_accuracy_on_positive", tier_correct / tier_total, tier_total)
        )
    if pattern_total:
        metrics.append(
            metric_row(variant_meta["variant_name"], variant_meta["provider"], variant_meta["model"], variant_meta["prompt_strategy"], "c3", "pattern_accuracy_on_tier2", pattern_correct / pattern_total, pattern_total)
        )
    return metrics


def make_majority_baseline(gt_df: pd.DataFrame) -> dict[str, Any]:
    emotions_mode = {emotion: majority_value(gt_df["gt_c3_emotions"].apply(lambda x: int((x or {}).get(emotion, 0) or 0)).tolist(), 0) for emotion in EMOTIONS}
    tiers_mode = {emotion: majority_value(gt_df["gt_c3_tiers"].apply(lambda x: int((x or {}).get(emotion, 0) or 0)).tolist(), 0) for emotion in EMOTIONS}
    patterns_mode = {
        emotion: majority_value(gt_df["gt_c3_patterns"].apply(lambda x: str((x or {}).get(emotion, "none") or "none")).tolist(), "none")
        for emotion in EMOTIONS
    }
    return {
        "variant_name": "majority_baseline",
        "provider": "baseline",
        "model": "deterministic",
        "prompt_strategy": "majority_class",
        "pred_c1_score": majority_value(gt_df["gt_c1_score"].tolist(), 0),
        "pred_c2_abs_code": majority_value(gt_df["gt_c2_abs_code"].tolist(), 0),
        "pred_c2_abs_amount": majority_value(gt_df["gt_c2_abs_amount"].dropna().tolist(), "Unspecified"),
        "pred_c2_rel_score": majority_value(gt_df["gt_c2_rel_score"].tolist(), 0),
        "pred_c2_psych_harm": majority_value(gt_df["gt_c2_psych_harm"].tolist(), 0),
        "pred_c4_score": majority_value(gt_df["gt_c4_score"].tolist(), 0),
        "pred_c3_emotions": emotions_mode,
        "pred_c3_tiers": tiers_mode,
        "pred_c3_patterns": patterns_mode,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions_dir = run_dir / "predictions"
    evaluation_dir = ensure_dir(run_dir / "evaluation")

    gt_df = build_ground_truth(pd.read_csv(args.ground_truth))
    gt_lookup = gt_df.set_index("Complaint ID").to_dict(orient="index")

    all_rows: list[dict[str, Any]] = []
    for variant in manifest["variants"]:
        prediction_path = predictions_dir / f"{slugify(variant['name'])}.jsonl"
        if not prediction_path.exists():
            continue
        for record in load_predictions_file(prediction_path):
            complaint_id = record["complaint_id"]
            try:
                gt_key_int = int(complaint_id)
            except (TypeError, ValueError):
                gt_key_int = None
            gt = gt_lookup.get(gt_key_int) or gt_lookup.get(str(complaint_id))
            if not gt or record.get("error") or not record.get("normalized_prediction"):
                continue
            row: dict[str, Any] = {
                "variant_name": record["variant_name"],
                "provider": record["provider"],
                "model": record["model"],
                "prompt_strategy": record["prompt_strategy"],
                "construct": record["construct"],
                "complaint_id": complaint_id,
                **gt,
            }
            normalized = record["normalized_prediction"]
            if record["construct"] == "c1":
                row["pred_c1_score"] = normalized.get("c1_score")
            elif record["construct"] == "c2":
                row["pred_c2_abs_code"] = normalized.get("c2_abs_code")
                row["pred_c2_abs_amount"] = normalized.get("c2_abs_amount")
                row["pred_c2_rel_score"] = normalized.get("c2_rel_score")
                row["pred_c2_psych_harm"] = normalized.get("c2_psych_harm")
            elif record["construct"] == "c3":
                row["pred_c3_emotions"] = normalized.get("c3_emotions_json", {})
                row["pred_c3_tiers"] = normalized.get("c3_tier_json", {})
                row["pred_c3_patterns"] = normalized.get("c3_inference_pattern_json", {})
            elif record["construct"] == "c4":
                row["pred_c4_score"] = normalized.get("c4_score")
            all_rows.append(row)

    metrics: list[dict[str, Any]] = []
    rows_df = pd.DataFrame(all_rows)
    if not rows_df.empty:
        for variant_name, variant_rows in rows_df.groupby("variant_name"):
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

            metrics.extend(evaluate_scalar(c1_rows, "gt_c1_score", "pred_c1_score", variant_meta, "c1", "accuracy"))
            metrics.extend(evaluate_scalar(c2_rows, "gt_c2_abs_code", "pred_c2_abs_code", variant_meta, "c2", "abs_code_accuracy"))
            metrics.extend(evaluate_scalar(c2_rows, "gt_c2_rel_score", "pred_c2_rel_score", variant_meta, "c2", "rel_score_accuracy"))
            metrics.extend(evaluate_scalar(c2_rows, "gt_c2_psych_harm", "pred_c2_psych_harm", variant_meta, "c2", "psych_harm_accuracy"))
            metrics.extend(evaluate_amount(c2_rows, variant_meta))
            if c3_rows:
                metrics.extend(evaluate_c3(c3_rows, variant_meta))
            metrics.extend(evaluate_scalar(c4_rows, "gt_c4_score", "pred_c4_score", variant_meta, "c4", "accuracy"))

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
    metrics.extend(evaluate_scalar(baseline_df[baseline_df["construct"] == "c1"].to_dict(orient="records"), "gt_c1_score", "pred_c1_score", variant_meta, "c1", "accuracy"))
    c2_rows = baseline_df[baseline_df["construct"] == "c2"].to_dict(orient="records")
    metrics.extend(evaluate_scalar(c2_rows, "gt_c2_abs_code", "pred_c2_abs_code", variant_meta, "c2", "abs_code_accuracy"))
    metrics.extend(evaluate_scalar(c2_rows, "gt_c2_rel_score", "pred_c2_rel_score", variant_meta, "c2", "rel_score_accuracy"))
    metrics.extend(evaluate_scalar(c2_rows, "gt_c2_psych_harm", "pred_c2_psych_harm", variant_meta, "c2", "psych_harm_accuracy"))
    metrics.extend(evaluate_amount(c2_rows, variant_meta))
    metrics.extend(evaluate_c3(baseline_df[baseline_df["construct"] == "c3"].to_dict(orient="records"), variant_meta))
    metrics.extend(evaluate_scalar(baseline_df[baseline_df["construct"] == "c4"].to_dict(orient="records"), "gt_c4_score", "pred_c4_score", variant_meta, "c4", "accuracy"))

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
                metric_lookup.get(("c1", "accuracy"), ""),
                metric_lookup.get(("c2", "abs_code_accuracy"), ""),
                metric_lookup.get(("c2", "rel_score_accuracy"), ""),
                metric_lookup.get(("c2", "psych_harm_accuracy"), ""),
                metric_lookup.get(("c3", "emotion_presence_f1"), ""),
                metric_lookup.get(("c4", "accuracy"), ""),
            ]
        )
    summary_md = markdown_table(
        [
            "Variant",
            "Provider",
            "Model",
            "Prompt Strategy",
            "C1 Acc",
            "C2 Abs Acc",
            "C2 Rel Acc",
            "C2 Psych Acc",
            "C3 F1",
            "C4 Acc",
        ],
        summary_rows,
    )
    summary_path = evaluation_dir / "metrics_summary.md"
    summary_path.write_text(
        "# Variant Evaluation Summary\n\n"
        f"Ground truth source: `{Path(args.ground_truth).resolve()}`\n\n"
        f"{summary_md}\n"
    )

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
