from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from inference_eval.analysis.metrics_analysis import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_DIR,
    PRIMARY_METRIC_LABELS,
    build_summary_table,
    load_metrics,
    markdown_table,
    prompt_sort_key,
)

C3_DEEP_DIVE_METRICS = [
    "C3 Code Acc",
    "Emotion Precision",
    "Emotion Recall",
    "Emotion F1",
    "Tier Accuracy",
]
MODEL_DISPLAY_ORDER = {
    "Gemma 3 4B": 1,
    "Qwen 3 8B": 2,
    "GPT-5-nano": 3,
    "GPT-5.4-mini": 4,
}
PROMPT_DISPLAY = {
    "majority_class": "Baseline",
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "retrieval_few_shot": "Retrieval few-shot",
}
DISPLAY_PROMPT_ORDER = {
    "Baseline": 0,
    "Zero-shot": 1,
    "Few-shot": 2,
    "Retrieval few-shot": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slide-ready presentation tables for inference evaluation.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory containing evaluation/metrics_by_variant.csv.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where presentation tables will be written.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_display_name(provider: str, model: str) -> str:
    if provider == "baseline":
        return "Majority baseline"
    if model.startswith("gemma3:"):
        return f"Gemma 3 {model.split(':', 1)[1].upper()}"
    if model.startswith("qwen3:"):
        return f"Qwen 3 {model.split(':', 1)[1].upper()}"
    dated_openai = re.match(r"^(gpt-[a-z0-9.-]+)-\d{4}-\d{2}-\d{2}$", model)
    if dated_openai:
        model = dated_openai.group(1)
    if model == "gpt-5-nano":
        return "GPT-5-nano"
    if model == "gpt-5.4-mini":
        return "GPT-5.4-mini"
    return model


def _retrieval_backend_label(variant_name: str, prompt_strategy: str) -> str | None:
    if prompt_strategy != "retrieval_few_shot":
        return None
    if "sentence-transformer" in variant_name or "minilm" in variant_name:
        return "MiniLM"
    return "TF-IDF"


def _prompt_display_name(variant_name: str, prompt_strategy: str) -> str:
    if prompt_strategy != "retrieval_few_shot":
        return PROMPT_DISPLAY.get(prompt_strategy, prompt_strategy)
    backend = _retrieval_backend_label(variant_name, prompt_strategy)
    return f"Retrieval few-shot ({backend})"


def _strategy_rollup_name(prompt_strategy: str) -> str:
    return PROMPT_DISPLAY.get(prompt_strategy, prompt_strategy)


def _with_display_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()
    display_df["Model"] = [
        _model_display_name(provider, model)
        for provider, model in zip(display_df["provider"], display_df["model"], strict=False)
    ]
    display_df["Prompt Strategy"] = [
        _prompt_display_name(variant_name, prompt_strategy)
        for variant_name, prompt_strategy in zip(
            display_df["variant_name"],
            display_df["prompt_strategy"],
            strict=False,
        )
    ]
    display_df["model_order"] = display_df["Model"].map(MODEL_DISPLAY_ORDER).fillna(999)
    display_df["prompt_order"] = [
        prompt_sort_key(prompt_strategy)[0] for prompt_strategy in display_df["prompt_strategy"]
    ]
    return display_df


def _load_summary(run_dir: str | Path = DEFAULT_RUN_DIR) -> pd.DataFrame:
    metrics_df = load_metrics(Path(run_dir))
    return build_summary_table(metrics_df)


def build_main_results_table(run_dir: str | Path = DEFAULT_RUN_DIR) -> pd.DataFrame:
    summary_df = _with_display_columns(_load_summary(run_dir))
    summary_df = summary_df[summary_df["prompt_strategy"] != "majority_class"].copy()
    table = summary_df.sort_values(["prompt_order", "model_order", "Prompt Strategy", "Model"]).reset_index(drop=True)
    return table[["Model", "Prompt Strategy", *PRIMARY_METRIC_LABELS]].copy()


def _collapse_strategy_variants(summary_df: pd.DataFrame) -> pd.DataFrame:
    collapsed = _with_display_columns(summary_df)
    collapsed = collapsed[collapsed["prompt_strategy"] != "majority_class"].copy()
    collapsed = (
        collapsed.groupby(["Model", "prompt_strategy"], dropna=False)[PRIMARY_METRIC_LABELS]
        .mean()
        .reset_index()
    )
    collapsed["Prompt Strategy"] = collapsed["prompt_strategy"].map(_strategy_rollup_name)
    collapsed["model_order"] = collapsed["Model"].map(MODEL_DISPLAY_ORDER).fillna(999)
    collapsed["prompt_order"] = [
        prompt_sort_key(prompt_strategy)[0] for prompt_strategy in collapsed["prompt_strategy"]
    ]
    return collapsed


def build_prompt_strategy_table(run_dir: str | Path = DEFAULT_RUN_DIR) -> pd.DataFrame:
    collapsed = _collapse_strategy_variants(_load_summary(run_dir))
    table = (
        collapsed.groupby(["prompt_strategy", "Prompt Strategy"], dropna=False)
        .agg(
            models_averaged=("Model", "nunique"),
            **{metric: (metric, "mean") for metric in PRIMARY_METRIC_LABELS},
        )
        .reset_index()
        .sort_values("prompt_strategy", key=lambda column: column.map(lambda value: prompt_sort_key(value)[0]))
        .reset_index(drop=True)
    )
    table = table.rename(columns={"models_averaged": "Models Averaged"})
    return table[["Prompt Strategy", "Models Averaged", *PRIMARY_METRIC_LABELS]].copy()


def build_family_comparison_table(run_dir: str | Path = DEFAULT_RUN_DIR) -> pd.DataFrame:
    collapsed = _collapse_strategy_variants(_load_summary(run_dir))
    table = (
        collapsed.groupby("Model", dropna=False)
        .agg(
            prompt_strategies=(
                "Prompt Strategy",
                lambda values: ", ".join(sorted(set(values), key=lambda value: DISPLAY_PROMPT_ORDER.get(value, 999))),
            ),
            **{metric: (metric, "mean") for metric in PRIMARY_METRIC_LABELS},
        )
        .reset_index()
    )
    table["model_order"] = table["Model"].map(MODEL_DISPLAY_ORDER).fillna(999)
    table = table.sort_values(["model_order", "Model"]).reset_index(drop=True)
    table = table.rename(columns={"prompt_strategies": "Prompt Strategies"})
    return table[["Model", "Prompt Strategies", *PRIMARY_METRIC_LABELS]].copy()


def build_c3_deep_dive_table(run_dir: str | Path = DEFAULT_RUN_DIR, *, top_n: int = 3) -> pd.DataFrame:
    metrics_df = load_metrics(Path(run_dir))
    summary_df = _with_display_columns(build_summary_table(metrics_df))
    summary_df = summary_df[summary_df["prompt_strategy"] != "majority_class"].copy()
    c3_df = metrics_df[metrics_df["construct"] == "c3"].copy()
    c3_pivot = (
        c3_df.pivot_table(index="variant_name", columns="metric", values="value", aggfunc="first").reset_index()
    )
    metadata = summary_df[["variant_name", "Model", "Prompt Strategy", "model_order", "prompt_order"]].drop_duplicates()
    table = metadata.merge(c3_pivot, on="variant_name", how="inner")
    table = table.rename(
        columns={
            "code_accuracy": "C3 Code Acc",
            "emotion_label_precision": "Emotion Precision",
            "emotion_label_recall": "Emotion Recall",
            "emotion_label_f1": "Emotion F1",
            "tier_accuracy_on_positive": "Tier Accuracy",
        }
    )
    table = table.sort_values(
        ["Emotion F1", "Tier Accuracy", "C3 Code Acc", "model_order", "prompt_order"],
        ascending=[False, False, False, True, True],
    ).head(top_n)
    return table[["Model", "Prompt Strategy", *C3_DEEP_DIVE_METRICS]].reset_index(drop=True)


def summarize_metric_winners(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and column not in {"Models Averaged"}
    ]
    rows = []
    for metric in metric_columns:
        best_value = df[metric].max()
        winners = (
            df.loc[df[metric] == best_value, id_columns]
            .astype(str)
            .agg(" | ".join, axis=1)
            .tolist()
        )
        rows.append(
            {
                "Metric": metric,
                "Best Value": best_value,
                "Winner(s)": "; ".join(winners),
            }
        )
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df) + "\n", encoding="utf-8")
    return csv_path


def build_presentation_outputs(run_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)

    main_results_df = build_main_results_table(run_dir)
    main_best_df = summarize_metric_winners(main_results_df, ["Model", "Prompt Strategy"])
    strategy_df = build_prompt_strategy_table(run_dir)
    strategy_best_df = summarize_metric_winners(strategy_df, ["Prompt Strategy"])
    family_df = build_family_comparison_table(run_dir)
    family_best_df = summarize_metric_winners(family_df, ["Model"])
    c3_df = build_c3_deep_dive_table(run_dir, top_n=3)
    c3_best_df = summarize_metric_winners(c3_df, ["Model", "Prompt Strategy"])

    tables = {
        "presentation_main_results": main_results_df,
        "presentation_main_results_winners": main_best_df,
        "presentation_prompt_strategy_comparison": strategy_df,
        "presentation_prompt_strategy_winners": strategy_best_df,
        "presentation_family_comparison": family_df,
        "presentation_family_winners": family_best_df,
        "presentation_c3_deep_dive": c3_df,
        "presentation_c3_winners": c3_best_df,
    }

    paths = {}
    for stem, df in tables.items():
        paths[stem] = write_table(df, output_dir, stem)
    return paths


def main() -> int:
    args = parse_args()
    outputs = build_presentation_outputs(Path(args.run_dir), Path(args.output_dir))
    print(f"Wrote presentation tables to {args.output_dir}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
