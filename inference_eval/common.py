from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATASET = "annotation_folder/annotated_results_merged_100.xlsx"
DEFAULT_RUNS_DIR = "inference_eval/results"

CONSTRUCTS = ("c1", "c2", "c3", "c4")

C0_LABELS = {
    0: "none",
    1: "account_credentials",
    2: "personal_identification",
    3: "personal_possessions",
    4: "behavioural_characteristics",
}
C1_LABELS = {
    0: "not_assessable",
    1: "low_perceived_vulnerability",
    2: "moderate_perceived_vulnerability",
    3: "high_perceived_vulnerability",
    4: "severe_perceived_vulnerability",
}
C2_REL_LABELS = {
    0: "not_assessable",
    1: "principled_loss",
    2: "nuisance_loss",
    3: "high_impact_loss",
    4: "life_altering_loss",
}
C4_LABELS = {
    0: "no_digital_asset_engagement",
    1: "online_transfers_and_transactions",
    2: "digital_asset_engagement",
}

EMOTIONS = (
    "anger",
    "sadness",
    "anxiety",
    "shame",
    "embarrassment",
    "disgust",
    "vengeance",
    "regret",
)
INFERENCE_PATTERNS = (
    "explicit_emotion",
    "intensified_language_and_rhetorical_escalation",
    "behavioural_descriptions_of_emotional_states",
    "self_blame_or_counterfactual_language",
    "impact_statements_on_life_quality",
    "relational_consequence_descriptions",
    "justice_seeking_language",
)

_C0_TEXT_TO_CODE = {
    "none": 0,
    "not assessable": 0,
    "account credentials": 1,
    "credentials": 1,
    "personal identification": 2,
    "personal possessions": 3,
    "behavioural characteristics": 4,
    "behavioral characteristics": 4,
}
_C1_TEXT_TO_CODE = {
    "not assessable": 0,
    "not accessible": 0,
    "low perceived vulnerability": 1,
    "moderate perceived vulnerability": 2,
    "high perceived vulnerability": 3,
    "severe perceived vulnerability": 4,
}
_C4_TEXT_TO_CODE = {
    "no digital asset engagement": 0,
    "online transfers and transactions": 1,
    "online transfers and transcations": 1,
    "digital asset engagement": 2,
}
_EMOTION_ALIASES = {
    "anger": "anger",
    "frustration": "anger",
    "sadness": "sadness",
    "anxiety": "anxiety",
    "fear": "anxiety",
    "fear anxiety": "anxiety",
    "shame": "shame",
    "embarrassment": "embarrassment",
    "embarassment": "embarrassment",
    "disgust": "disgust",
    "vengeance": "vengeance",
    "regret": "regret",
}
_PATTERN_ALIASES = {
    "explicit emotion": "explicit_emotion",
    "intensified language and rhetorical escalation": "intensified_language_and_rhetorical_escalation",
    "behavioural descriptions of emotional states": "behavioural_descriptions_of_emotional_states",
    "behavioral descriptions of emotional states": "behavioural_descriptions_of_emotional_states",
    "behaviourial descriptions of emotional states": "behavioural_descriptions_of_emotional_states",
    "self blame or counterfactual language": "self_blame_or_counterfactual_language",
    "self blame counterfactual language": "self_blame_or_counterfactual_language",
    "selfblame or counterfactual language": "self_blame_or_counterfactual_language",
    "self-blame or counterfactual language": "self_blame_or_counterfactual_language",
    "impact statements on life quality": "impact_statements_on_life_quality",
    "impact statement on life quality": "impact_statements_on_life_quality",
    "impact statements": "impact_statements_on_life_quality",
    "relational consequence descriptions": "relational_consequence_descriptions",
    "justice-seeking language": "justice_seeking_language",
    "justice seeking language": "justice_seeking_language",
}


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


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


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
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_amount(value: Any) -> str | None:
    if is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    text = text.replace("$", "").replace(",", "")
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    numeric = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if numeric:
        try:
            number = float(text)
            if number.is_integer():
                return str(int(number))
            normalized = f"{number:.2f}".rstrip("0").rstrip(".")
            return normalized
        except ValueError:
            pass
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


def normalize_text_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def label_for_code(mapping: dict[int, str], value: Any) -> str | None:
    code = to_int(value)
    if code is None:
        return None
    return mapping.get(code)


def normalize_c0_label(value: Any) -> int | None:
    code = to_int(value)
    if code in C0_LABELS:
        return code
    return _C0_TEXT_TO_CODE.get(normalize_text_key(value))


def normalize_c1_score(value: Any) -> int | None:
    code = to_int(value)
    if code in C1_LABELS:
        return code
    return _C1_TEXT_TO_CODE.get(normalize_text_key(value))


def normalize_c4_score(value: Any) -> int | None:
    code = to_int(value)
    if code in C4_LABELS:
        return code
    return _C4_TEXT_TO_CODE.get(normalize_text_key(value))


def normalize_c2_rel_score(value: Any) -> int | None:
    code = to_int(value)
    if code in C2_REL_LABELS:
        return code
    return None


def ordered_unique(values: list[str], order: tuple[str, ...]) -> list[str]:
    seen = set()
    ordered = []
    for candidate in order:
        if candidate in values and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    for candidate in values:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def parse_delimited_or_json_list(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    loaded = load_json(value, default=None)
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    parts = re.split(r"[|,]", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_emotion_list(value: Any) -> list[str]:
    normalized = []
    for item in parse_delimited_or_json_list(value):
        alias = _EMOTION_ALIASES.get(normalize_text_key(item))
        if alias:
            normalized.append(alias)
    return ordered_unique(normalized, EMOTIONS)


def normalize_pattern_list(value: Any) -> list[str]:
    normalized = []
    for item in parse_delimited_or_json_list(value):
        alias = _PATTERN_ALIASES.get(normalize_text_key(item))
        if alias:
            normalized.append(alias)
    return ordered_unique(normalized, INFERENCE_PATTERNS)


def normalize_c3_code(value: Any, emotions: list[str] | None = None) -> int | None:
    code = to_int(value)
    if code in (0, 1, 2):
        return code
    if emotions is None:
        return None
    if not emotions:
        return 0
    return 2 if len(emotions) >= 2 else 1


def normalize_c3_tier(value: Any, code: int | None = None, patterns: list[str] | None = None) -> int | None:
    tier = to_int(value)
    if tier in (0, 1, 2):
        return tier
    if code == 0:
        return 0
    if patterns:
        if "explicit_emotion" in patterns and len(patterns) == 1:
            return 1
        return 2
    return None


def read_column(df: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    gt = df.copy()
    gt["gt_c0_label"] = read_column(gt, "c0_label").apply(normalize_c0_label)
    gt["gt_c1_score"] = read_column(gt, "c1_score").apply(normalize_c1_score)
    gt["gt_c2_abs_code"] = read_column(gt, "c2_abs_code").apply(to_int)
    gt["gt_c2_abs_amount"] = read_column(gt, "c2_abs_amount").apply(normalize_amount)
    gt["gt_c2_rel_score"] = read_column(gt, "c2_rel_score").apply(normalize_c2_rel_score)
    gt["gt_c4_score"] = read_column(gt, "c4_score").apply(normalize_c4_score)

    gt.loc[gt["gt_c2_abs_code"] == 0, "gt_c2_abs_amount"] = None

    gt["gt_c3_emotions"] = read_column(gt, "c3_emotions").apply(normalize_emotion_list)
    gt["gt_c3_code"] = [
        normalize_c3_code(value, emotions)
        for value, emotions in zip(read_column(gt, "c3_code"), gt["gt_c3_emotions"], strict=False)
    ]
    gt["gt_c3_inference_patterns"] = read_column(gt, "c3_inference_pattern").apply(normalize_pattern_list)
    gt["gt_c3_tier"] = [
        normalize_c3_tier(value, code, patterns)
        for value, code, patterns in zip(
            read_column(gt, "c3_tier"),
            gt["gt_c3_code"],
            gt["gt_c3_inference_patterns"],
            strict=False,
        )
    ]
    gt.loc[gt["gt_c3_code"] == 0, "gt_c3_inference_patterns"] = gt.loc[gt["gt_c3_code"] == 0, "gt_c3_inference_patterns"].apply(lambda _: [])
    return gt


def c1_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "user_info_leaked_code": {"type": "integer", "minimum": 0, "maximum": 4},
            "user_info_leaked_label": {"type": "string", "enum": list(C0_LABELS.values())},
            "user_info_evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "perceived_info_vulnerability_score": {"type": "integer", "minimum": 0, "maximum": 4},
            "perceived_info_vulnerability_label": {"type": "string", "enum": list(C1_LABELS.values())},
            "perceived_vulnerability_evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "user_info_leaked_code",
            "user_info_leaked_label",
            "user_info_evidence_quotes",
            "perceived_info_vulnerability_score",
            "perceived_info_vulnerability_label",
            "perceived_vulnerability_evidence_quotes",
            "rationale",
        ],
    }


def c2_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "economic_absolute_code": {"type": "integer", "minimum": 0, "maximum": 2},
            "economic_absolute_amount_usd": {"type": ["string", "number", "null"]},
            "economic_relative_significance_score": {"type": "integer", "minimum": 0, "maximum": 4},
            "economic_relative_label": {"type": "string", "enum": list(C2_REL_LABELS.values())},
            "economic_evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "economic_absolute_code",
            "economic_absolute_amount_usd",
            "economic_relative_significance_score",
            "economic_relative_label",
            "economic_evidence_quotes",
            "rationale",
        ],
    }


def c3_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "emotion_count_code": {"type": "integer", "minimum": 0, "maximum": 2},
            "emotions": {
                "type": "array",
                "items": {"type": "string", "enum": list(EMOTIONS)},
            },
            "evidence_tier": {"type": "integer", "minimum": 0, "maximum": 2},
            "inference_patterns": {
                "type": "array",
                "items": {"type": "string", "enum": list(INFERENCE_PATTERNS)},
            },
            "evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "emotion_count_code",
            "emotions",
            "evidence_tier",
            "inference_patterns",
            "evidence_quotes",
            "rationale",
        ],
    }


def c4_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "investment_engagement_score": {"type": "integer", "minimum": 0, "maximum": 2},
            "investment_engagement_label": {"type": "string", "enum": list(C4_LABELS.values())},
            "evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "investment_engagement_score",
            "investment_engagement_label",
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
        c0_code = normalize_c0_label(parsed.get("user_info_leaked_code"))
        c1_code = normalize_c1_score(parsed.get("perceived_info_vulnerability_score"))
        return {
            "c0_label": c0_code,
            "c0_label_text": label_for_code(C0_LABELS, c0_code),
            "c0_evidence_quotes": list(parsed.get("user_info_evidence_quotes", [])),
            "c1_score": c1_code,
            "c1_label": label_for_code(C1_LABELS, c1_code),
            "c1_evidence_quotes": list(parsed.get("perceived_vulnerability_evidence_quotes", [])),
        }
    if construct == "c2":
        return {
            "c2_abs_code": to_int(parsed.get("economic_absolute_code")),
            "c2_abs_amount": normalize_amount(parsed.get("economic_absolute_amount_usd")),
            "c2_rel_score": normalize_c2_rel_score(parsed.get("economic_relative_significance_score")),
            "c2_rel_label": label_for_code(C2_REL_LABELS, parsed.get("economic_relative_significance_score")),
            "c2_evidence_quotes": list(parsed.get("economic_evidence_quotes", [])),
        }
    if construct == "c3":
        emotions = normalize_emotion_list(parsed.get("emotions"))
        patterns = normalize_pattern_list(parsed.get("inference_patterns"))
        code = normalize_c3_code(parsed.get("emotion_count_code"), emotions)
        inferred_code = normalize_c3_code(None, emotions)
        if inferred_code is not None:
            code = inferred_code
        tier = normalize_c3_tier(parsed.get("evidence_tier"), code, patterns)
        if code == 0:
            emotions = []
            patterns = []
            tier = 0
        elif patterns and any(pattern != "explicit_emotion" for pattern in patterns):
            tier = 2
        elif patterns == ["explicit_emotion"] and tier is None:
            tier = 1
        return {
            "c3_code": code,
            "c3_emotions": emotions,
            "c3_tier": tier,
            "c3_inference_patterns": patterns,
            "c3_evidence_quotes": list(parsed.get("evidence_quotes", [])),
        }
    if construct == "c4":
        c4_code = normalize_c4_score(parsed.get("investment_engagement_score"))
        return {
            "c4_score": c4_code,
            "c4_label": label_for_code(C4_LABELS, c4_code),
            "c4_evidence_quotes": list(parsed.get("evidence_quotes", [])),
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
    clean = []
    for value in values:
        if isinstance(value, (list, dict, tuple, set)):
            clean.append(value)
        elif not is_missing(value):
            clean.append(value)
    if not clean:
        return fallback
    return Counter(clean).most_common(1)[0][0]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rendered)
