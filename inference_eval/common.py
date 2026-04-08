from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATASET = "annotation_folder/annotated_results.csv"
DEFAULT_RUNS_DIR = "inference_eval/runs"

CONSTRUCTS = ("c1", "c2", "c3", "c4")
EMOTIONS = (
    "anger",
    "sadness",
    "anxiety",
    "shame",
    "embarrassment",
    "disgust",
    "regret",
    "vengeance",
)
INFERENCE_PATTERNS = (
    "none",
    "explicit",
    "intensified_language",
    "behavioural_description",
    "self_blame_counterfactual",
    "impact_statement",
    "relational_consequence",
    "justice_seeking",
    "other",
)


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def load_json(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def to_int(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_amount(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return text


def parse_pipe_separated_evidence(value: Any) -> list[str]:
    if is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        loaded = load_json(text, default=[])
        if isinstance(loaded, list):
            return [str(item).strip() for item in loaded if str(item).strip()]
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return [part.strip() for part in text.split("|") if part.strip()]


def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    gt = df.copy()
    gt["gt_c1_score"] = gt["c1_score"].apply(to_int)
    gt["gt_c2_abs_code"] = gt["c2_abs_code"].apply(to_int)
    gt["gt_c2_abs_amount"] = gt["c2_abs_amount"].apply(normalize_amount)
    gt["gt_c2_rel_score"] = gt["c2_rel_score"].apply(to_int)
    gt["gt_c2_psych_harm"] = gt["c2_psych_harm"].apply(to_int)
    gt["gt_c4_score"] = gt["c4_score"].apply(to_int)
    gt["gt_c4_use_type"] = gt["c4_use_type"].fillna("unknown").astype(str).str.strip()
    gt["gt_c3_emotions"] = gt["c3_emotions_json"].apply(lambda x: load_json(x, default={}) or {})
    gt["gt_c3_tiers"] = gt["c3_tier_json"].apply(lambda x: load_json(x, default={}) or {})
    gt["gt_c3_patterns"] = gt["c3_inference_pattern_json"].apply(lambda x: load_json(x, default={}) or {})
    return gt


def c1_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "perceived_info_vulnerability_score": {"type": "integer", "minimum": 0, "maximum": 4},
            "perceived_info_vulnerability_label": {"type": "string"},
            "evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "perceived_info_vulnerability_score",
            "perceived_info_vulnerability_label",
            "evidence_quotes",
            "rationale",
        ],
    }


def c2_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "economic_absolute_code": {"type": "integer", "minimum": 0, "maximum": 2},
            "economic_absolute_amount_usd": {"type": ["string", "null"]},
            "economic_relative_significance_score": {"type": "integer", "minimum": 0, "maximum": 4},
            "economic_relative_label": {"type": "string"},
            "psychological_harm_present": {"type": "integer", "minimum": 0, "maximum": 1},
            "evidence_quotes": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "economic_absolute": {"type": "array", "items": {"type": "string"}},
                    "economic_relative": {"type": "array", "items": {"type": "string"}},
                    "psychological": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["economic_absolute", "economic_relative", "psychological"],
            },
            "rationale": {"type": "string"},
        },
        "required": [
            "economic_absolute_code",
            "economic_absolute_amount_usd",
            "economic_relative_significance_score",
            "economic_relative_label",
            "psychological_harm_present",
            "evidence_quotes",
            "rationale",
        ],
    }


def c3_schema() -> dict[str, Any]:
    emotion_entry = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "present": {"type": "integer", "minimum": 0, "maximum": 1},
            "tier": {"type": "integer", "minimum": 0, "maximum": 2},
            "inference_pattern": {"type": "string"},
            "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["present", "tier", "inference_pattern", "evidence_quotes"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "emotions": {
                "type": "object",
                "additionalProperties": False,
                "properties": {emotion: emotion_entry for emotion in EMOTIONS},
                "required": list(EMOTIONS),
            },
            "rationale": {"type": "string"},
        },
        "required": ["emotions", "rationale"],
    }


def c4_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "investment_engagement_score": {"type": "integer", "minimum": 0, "maximum": 2},
            "investment_engagement_label": {"type": "string"},
            "use_type": {"type": "string"},
            "evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "investment_engagement_score",
            "investment_engagement_label",
            "use_type",
            "evidence_quotes",
            "rationale",
        ],
    }


def schema_for_construct(construct: str) -> dict[str, Any]:
    if construct == "c1":
        return c1_schema()
    if construct == "c2":
        return c2_schema()
    if construct == "c3":
        return c3_schema()
    if construct == "c4":
        return c4_schema()
    raise ValueError(f"Unknown construct: {construct}")


def normalize_prediction(construct: str, parsed: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed or {}
    if construct == "c1":
        return {
            "c1_score": to_int(parsed.get("perceived_info_vulnerability_score")),
            "c1_label": parsed.get("perceived_info_vulnerability_label"),
            "c1_evidence_quotes": parsed.get("evidence_quotes", []),
        }
    if construct == "c2":
        evidence = parsed.get("evidence_quotes", {}) or {}
        return {
            "c2_abs_code": to_int(parsed.get("economic_absolute_code")),
            "c2_abs_amount": normalize_amount(parsed.get("economic_absolute_amount_usd")),
            "c2_rel_score": to_int(parsed.get("economic_relative_significance_score")),
            "c2_rel_label": parsed.get("economic_relative_label"),
            "c2_psych_harm": to_int(parsed.get("psychological_harm_present")),
            "c2_evidence_quotes": evidence,
        }
    if construct == "c3":
        emotions = parsed.get("emotions", {}) or {}
        return {
            "c3_emotions_json": {emotion: to_int((emotions.get(emotion) or {}).get("present")) or 0 for emotion in EMOTIONS},
            "c3_tier_json": {emotion: to_int((emotions.get(emotion) or {}).get("tier")) or 0 for emotion in EMOTIONS},
            "c3_inference_pattern_json": {
                emotion: str((emotions.get(emotion) or {}).get("inference_pattern", "none")).strip() or "none"
                for emotion in EMOTIONS
            },
            "c3_evidence_quotes": {
                emotion: list((emotions.get(emotion) or {}).get("evidence_quotes", []))
                for emotion in EMOTIONS
            },
        }
    if construct == "c4":
        return {
            "c4_score": to_int(parsed.get("investment_engagement_score")),
            "c4_label": parsed.get("investment_engagement_label"),
            "c4_use_type": str(parsed.get("use_type", "unknown")).strip() or "unknown",
            "c4_evidence_quotes": parsed.get("evidence_quotes", []),
        }
    raise ValueError(f"Unknown construct: {construct}")


def load_predictions_file(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def majority_value(values: list[Any], fallback: Any) -> Any:
    clean = [value for value in values if value is not None]
    if not clean:
        return fallback
    return Counter(clean).most_common(1)[0][0]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rendered)
