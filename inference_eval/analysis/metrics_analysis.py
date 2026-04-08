from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "inference_eval" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "inference_eval" / "analysis" / "outputs"

PRIMARY_METRICS: list[tuple[str, str, str]] = [
    ("c1", "c0_label_accuracy", "C0 Acc"),
    ("c1", "c1_score_accuracy", "C1 Acc"),
    ("c2", "abs_code_accuracy", "C2 Abs Acc"),
    ("c2", "rel_score_accuracy", "C2 Rel Acc"),
    ("c3", "code_accuracy", "C3 Code Acc"),
    ("c3", "emotion_label_f1", "C3 Emotion F1"),
    ("c4", "score_accuracy", "C4 Acc"),
]
PRIMARY_METRIC_LABELS = [label for _, _, label in PRIMARY_METRICS]
PRIMARY_LOOKUP = {(construct, metric): label for construct, metric, label in PRIMARY_METRICS}
PROMPT_ORDER = {
    "majority_class": 0,
    "zero_shot": 1,
    "few_shot": 2,
    "retrieval_few_shot": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis tables for inference evaluation metrics.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory containing evaluation/metrics_by_variant.csv.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where analysis tables will be written.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def metric_sort_key(metric_label: str) -> tuple[int, str]:
    try:
        return (PRIMARY_METRIC_LABELS.index(metric_label), metric_label)
    except ValueError:
        return (len(PRIMARY_METRIC_LABELS), metric_label)


def prompt_sort_key(prompt_strategy: str) -> tuple[int, str]:
    return (PROMPT_ORDER.get(str(prompt_strategy), 999), str(prompt_strategy))


def model_family(provider: str, model: str) -> str:
    provider_text = str(provider or "")
    model_text = str(model or "")
    if provider_text == "baseline":
        return "baseline"
    if ":" in model_text:
        return model_text.split(":", 1)[0]
    dated_openai = re.match(r"^(gpt-[a-z0-9.-]+)-\d{4}-\d{2}-\d{2}$", model_text)
    if dated_openai:
        return dated_openai.group(1)
    return model_text


def markdown_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return str(round(value, 4))
    text = str(value)
    return text if text else "NA"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(markdown_cell(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df) + "\n", encoding="utf-8")


def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "evaluation" / "metrics_by_variant.csv"
    df = pd.read_csv(metrics_path)
    df["family"] = [model_family(provider, model) for provider, model in zip(df["provider"], df["model"], strict=False)]
    df["provider_family"] = [
        "baseline" if provider == "baseline" else f"{provider}:{family}"
        for provider, family in zip(df["provider"], df["family"], strict=False)
    ]
    df["metric_label"] = [
        PRIMARY_LOOKUP.get((construct, metric), f"{construct}::{metric}")
        for construct, metric in zip(df["construct"], df["metric"], strict=False)
    ]
    return df


def build_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    key_df = metrics_df[metrics_df["metric_label"].isin(PRIMARY_METRIC_LABELS)].copy()
    summary = (
        key_df.pivot_table(
            index=["variant_name", "provider", "model", "family", "prompt_strategy"],
            columns="metric_label",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    for column in PRIMARY_METRIC_LABELS:
        if column not in summary.columns:
            summary[column] = pd.NA
    summary = summary[
        ["variant_name", "provider", "model", "family", "prompt_strategy", *PRIMARY_METRIC_LABELS]
    ].copy()
    summary = summary.sort_values(
        by=["prompt_strategy", "provider", "family", "variant_name"],
        key=lambda column: column.map(prompt_sort_key) if column.name == "prompt_strategy" else column,
    )
    return summary.reset_index(drop=True)


def build_prompt_strategy_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby("prompt_strategy", dropna=False)
        .agg(
            variant_count=("variant_name", "nunique"),
            family_count=("family", "nunique"),
            **{label: (label, "mean") for label in PRIMARY_METRIC_LABELS},
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        by="prompt_strategy",
        key=lambda column: column.map(prompt_sort_key),
    ).reset_index(drop=True)
    return grouped


def build_family_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["provider", "family"], dropna=False)
        .agg(
            variant_count=("variant_name", "nunique"),
            prompt_strategies=("prompt_strategy", lambda values: ", ".join(sorted(set(values), key=lambda item: prompt_sort_key(item)))),
            **{label: (label, "mean") for label in PRIMARY_METRIC_LABELS},
        )
        .reset_index()
    )
    grouped = grouped.sort_values(by=["provider", "family"]).reset_index(drop=True)
    return grouped


def build_metric_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_view = summary_df.set_index("variant_name")[PRIMARY_METRIC_LABELS].transpose().reset_index()
    metric_view = metric_view.rename(columns={"index": "metric_label"})
    return metric_view


def build_primary_metric_leaderboard(metrics_df: pd.DataFrame) -> pd.DataFrame:
    leaderboard = metrics_df[metrics_df["metric_label"].isin(PRIMARY_METRIC_LABELS)].copy()
    leaderboard = leaderboard[
        ["metric_label", "variant_name", "provider", "family", "prompt_strategy", "value", "n"]
    ].sort_values(
        by=["metric_label", "value", "n", "variant_name"],
        ascending=[True, False, False, True],
    )
    leaderboard["rank"] = leaderboard.groupby("metric_label").cumcount() + 1
    leaderboard = leaderboard[
        ["metric_label", "rank", "variant_name", "provider", "family", "prompt_strategy", "value", "n"]
    ].reset_index(drop=True)
    return leaderboard


def build_primary_metric_coverage(metrics_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        metrics_df[metrics_df["metric_label"].isin(PRIMARY_METRIC_LABELS)]
        .pivot_table(index="variant_name", columns="metric_label", values="n", aggfunc="first")
        .reset_index()
    )
    for column in PRIMARY_METRIC_LABELS:
        if column not in coverage.columns:
            coverage[column] = pd.NA
    metadata = summary_df[["variant_name", "provider", "family", "prompt_strategy"]].drop_duplicates()
    coverage = metadata.merge(coverage, on="variant_name", how="left")
    coverage["available_primary_metrics"] = coverage[PRIMARY_METRIC_LABELS].notna().sum(axis=1)
    coverage = coverage[
        ["variant_name", "provider", "family", "prompt_strategy", "available_primary_metrics", *PRIMARY_METRIC_LABELS]
    ].sort_values(
        by=["prompt_strategy", "provider", "family", "variant_name"],
        key=lambda column: column.map(prompt_sort_key) if column.name == "prompt_strategy" else column,
    )
    return coverage.reset_index(drop=True)


def build_analysis_outputs(run_dir: Path, output_dir: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    output_dir = ensure_dir(Path(output_dir))
    metrics_df = load_metrics(run_dir)

    summary_df = build_summary_table(metrics_df)
    prompt_strategy_df = build_prompt_strategy_comparison(summary_df)
    family_df = build_family_comparison(summary_df)
    metric_view_df = build_metric_view(summary_df)
    leaderboard_df = build_primary_metric_leaderboard(metrics_df)
    coverage_df = build_primary_metric_coverage(metrics_df, summary_df)

    tables = {
        "summary_table": summary_df,
        "prompt_strategy_comparison": prompt_strategy_df,
        "family_comparison": family_df,
        "metric_view": metric_view_df,
        "primary_metric_leaderboard": leaderboard_df,
        "primary_metric_coverage": coverage_df,
    }

    for stem, df in tables.items():
        write_table(df, output_dir, stem)

    overview_path = output_dir / "analysis_overview.md"
    overview_lines = [
        "# Metrics Analysis Outputs",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "Generated tables:",
    ]
    for stem in tables:
        overview_lines.append(f"- `{stem}.csv`")
        overview_lines.append(f"- `{stem}.md`")
    overview_path.write_text("\n".join(overview_lines) + "\n", encoding="utf-8")

    return {
        "output_dir": output_dir,
        **{stem: output_dir / f"{stem}.csv" for stem in tables},
        "overview": overview_path,
    }


def main() -> int:
    args = parse_args()
    outputs = build_analysis_outputs(Path(args.run_dir), Path(args.output_dir))
    print(f"Wrote analysis outputs to {outputs['output_dir']}")
    for key, path in outputs.items():
        if key == "output_dir":
            continue
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
