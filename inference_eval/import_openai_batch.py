#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_eval.common import DEFAULT_RUNS_DIR, ensure_dir, normalize_prediction, slugify, write_jsonl
from inference_eval.run_variants import extract_openai_output, parse_json_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI batch output files into normalized prediction JSONL records."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch directory created by prepare_openai_batch.py / poll_openai_batch.py.",
    )
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUNS_DIR,
        help="Evaluation run directory containing the predictions/ folder.",
    )
    parser.add_argument(
        "--prediction-name",
        help="Optional prediction filename stem. Defaults to the batch manifest variant_name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination prediction file instead of appending.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge imported records into an existing prediction file by complaint x construct key.",
    )
    return parser.parse_args()


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


def record_key(record: dict[str, Any]) -> tuple[str, int | None, str, str]:
    return (
        str(record.get("variant_name") or ""),
        int(record["row_index"]) if record.get("row_index") not in (None, "") else None,
        str(record.get("complaint_id") or ""),
        str(record.get("construct") or ""),
    )


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


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


def extract_metadata(
    batch_record: dict[str, Any],
    request_lookup: dict[str, dict[str, Any]],
    request_index_lookup: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    response_body = ((batch_record.get("response") or {}).get("body") or {})
    metadata = dict(response_body.get("metadata") or {})
    request = request_lookup.get(batch_record.get("custom_id"), {})
    request_index_item = request_index_lookup.get(batch_record.get("custom_id"), {})
    if not metadata:
        request_body = request.get("body") or {}
        metadata.update(request_body.get("metadata") or {})
    for key in (
        "retrieval_policy",
        "retrieval_k",
        "retrieval_backend",
        "retrieval_embedding_model",
        "retrieved_complaint_ids",
        "retrieval_scores",
        "retrieved_row_indexes",
    ):
        if key in request_index_item:
            metadata[key] = request_index_item[key]
    parsed_custom_id = parse_custom_id(batch_record.get("custom_id", ""))
    if parsed_custom_id:
        for key, value in parsed_custom_id.items():
            metadata.setdefault(key, value)
    metadata.setdefault("variant_name", manifest.get("variant_name"))
    metadata.setdefault("prompt_strategy", manifest.get("prompt_strategy"))
    metadata.setdefault("complaint_id", None)
    metadata.setdefault("construct", None)
    metadata.setdefault("row_index", None)
    return metadata


def latency_seconds(response_body: dict[str, Any]) -> float | None:
    created_at = response_body.get("created_at")
    completed_at = response_body.get("completed_at")
    if created_at is None or completed_at is None:
        return None
    try:
        return round(float(completed_at) - float(created_at), 3)
    except (TypeError, ValueError):
        return None


def make_record(
    *,
    batch_record: dict[str, Any],
    metadata: dict[str, Any],
    provider: str,
    default_model: str | None,
) -> dict[str, Any]:
    response = batch_record.get("response") or {}
    response_body = response.get("body") or {}
    error_body = response_body.get("error")
    record: dict[str, Any] = {
        "variant_name": str(metadata.get("variant_name") or "openai-batch"),
        "provider": provider,
        "model": response_body.get("model") or default_model,
        "prompt_strategy": str(metadata.get("prompt_strategy") or "unknown"),
        "row_index": int(metadata["row_index"]) if metadata.get("row_index") not in (None, "") else None,
        "complaint_id": str(metadata.get("complaint_id") or ""),
        "construct": str(metadata.get("construct") or ""),
        "parsed_response": None,
        "normalized_prediction": None,
        "raw_text": None,
        "latency_seconds": latency_seconds(response_body),
        "error": None,
    }
    for key in (
        "retrieval_policy",
        "retrieval_k",
        "retrieval_backend",
        "retrieval_embedding_model",
        "retrieved_complaint_ids",
        "retrieval_scores",
        "retrieved_row_indexes",
    ):
        if key in metadata:
            record[key] = metadata[key]

    if error_body:
        record["error"] = error_body.get("message") or json.dumps(error_body, ensure_ascii=True)
        return record

    if response.get("status_code") != 200:
        record["error"] = f"OpenAI batch request failed with status {response.get('status_code')}."
        return record

    try:
        raw_text = extract_openai_output(response_body)
    except Exception:  # noqa: BLE001
        raw_text = None

    if raw_text is None:
        incomplete_reason = ((response_body.get("incomplete_details") or {}).get("reason"))
        if incomplete_reason:
            record["error"] = f"OpenAI response incomplete without output_text (reason: {incomplete_reason})"
        else:
            record["error"] = "OpenAI response missing output_text."
        return record

    record["raw_text"] = raw_text
    try:
        parsed = parse_json_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"Failed to parse response JSON: {exc}"
        return record

    record["parsed_response"] = parsed
    record["normalized_prediction"] = normalize_prediction(record["construct"], parsed)
    return record


def main() -> int:
    args = parse_args()
    if args.overwrite and args.merge:
        raise SystemExit("Use either --overwrite or --merge, not both.")
    batch_dir = Path(args.batch_dir)
    manifest = load_json(batch_dir / "manifest.json")
    request_index = {
        str(item["custom_id"]): item
        for item in load_jsonl(batch_dir / "request_index.jsonl")
        if item.get("custom_id")
    }
    requests_lookup = {
        str(item["custom_id"]): item
        for item in load_jsonl(batch_dir / "requests.jsonl")
        if item.get("custom_id")
    }
    prediction_stem = args.prediction_name or str(manifest.get("variant_name") or batch_dir.name)
    prediction_path = ensure_dir(Path(args.run_dir) / "predictions") / f"{slugify(prediction_stem)}.jsonl"
    if prediction_path.exists() and not args.overwrite and not args.merge:
        raise SystemExit(f"Prediction file already exists: {prediction_path}. Use --overwrite to replace it.")
    if prediction_path.exists() and args.overwrite:
        prediction_path.unlink()

    default_model = manifest.get("model")
    imported_records: list[dict[str, Any]] = []
    for source_name in ("output.jsonl", "errors.jsonl"):
        for batch_record in load_jsonl(batch_dir / source_name):
            metadata = extract_metadata(batch_record, requests_lookup, request_index, manifest)
            record = make_record(
                batch_record=batch_record,
                metadata=metadata,
                provider="openai",
                default_model=default_model,
            )
            if not record["construct"] or not record["complaint_id"]:
                raise SystemExit(f"Missing metadata needed to write prediction record for {batch_record.get('custom_id')}.")
            imported_records.append(record)

    if args.merge and prediction_path.exists():
        merged_records = load_jsonl(prediction_path)
        existing_index = {record_key(record): idx for idx, record in enumerate(merged_records)}
        replaced = 0
        appended = 0
        for record in imported_records:
            key = record_key(record)
            if key in existing_index:
                merged_records[existing_index[key]] = record
                replaced += 1
            else:
                merged_records.append(record)
                appended += 1
        write_records(prediction_path, merged_records)
        print(f"Merged {len(imported_records)} records into {prediction_path}")
        print(f"Replaced existing records: {replaced}")
        print(f"Appended new records: {appended}")
    else:
        for record in imported_records:
            write_jsonl(prediction_path, record)
        print(f"Wrote {len(imported_records)} records to {prediction_path}")

    print("Next step:")
    print(f"  python inference_eval/evaluate_variants.py --run-dir {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
