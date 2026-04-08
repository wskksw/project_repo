#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_eval.common import (
    CONSTRUCTS,
    DEFAULT_DATASET,
    DEFAULT_RUNS_DIR,
    ensure_dir,
    json_dumps,
    load_predictions_file,
    normalize_prediction,
    schema_for_construct,
    slugify,
    utc_timestamp_slug,
    write_jsonl,
)
from inference_eval.prompt_templates import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extractor variants against CFPB complaint narratives.")
    parser.add_argument("--config", required=True, help="Path to variant config JSON.")
    parser.add_argument("--data", default=DEFAULT_DATASET, help="CSV with narratives and provisional ground truth.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Base directory for run outputs.")
    parser.add_argument("--run-name", help="Optional explicit run folder name.")
    parser.add_argument("--limit", type=int, help="Only process the first N rows.")
    parser.add_argument("--constructs", nargs="+", choices=CONSTRUCTS, default=list(CONSTRUCTS))
    parser.add_argument("--variants", nargs="+", help="Optional subset of variant names to execute.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run directory.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds.")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def extract_openai_output(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return response["output_text"]
    output_blocks = response.get("output", [])
    for block in output_blocks:
        if block.get("type") != "message":
            continue
        for item in block.get("content", []):
            text = item.get("text")
            if text:
                return text
    raise RuntimeError(f"Could not find output text in OpenAI response: {response}")


def run_openai_variant(variant: dict[str, Any], system_prompt: str, user_prompt: str, schema: dict[str, Any], timeout: int) -> tuple[dict[str, Any], str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    payload: dict[str, Any] = {
        "model": variant["model"],
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": f"{variant['name']}_schema",
                "schema": schema,
                "strict": True,
            }
        },
    }
    if variant.get("max_output_tokens") is not None:
        payload["max_output_tokens"] = variant["max_output_tokens"]
    if variant.get("reasoning_effort") is not None:
        payload["reasoning"] = {"effort": variant["reasoning_effort"]}

    response = http_post_json(
        url=variant.get("base_url", "https://api.openai.com/v1/responses"),
        payload=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    raw_text = extract_openai_output(response)
    return json.loads(raw_text), raw_text


def run_ollama_variant(variant: dict[str, Any], system_prompt: str, user_prompt: str, schema: dict[str, Any], timeout: int) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = {
        "model": variant["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": schema,
        "stream": False,
        "options": {
            "temperature": variant.get("temperature", 0),
        },
    }
    if variant.get("num_ctx") is not None:
        payload["options"]["num_ctx"] = variant["num_ctx"]

    response = http_post_json(
        url=variant.get("base_url", "http://localhost:11434/api/chat"),
        payload=payload,
        headers={},
        timeout=timeout,
    )
    raw_text = ((response.get("message") or {}).get("content") or "").strip()
    if not raw_text:
        raise RuntimeError(f"Ollama response did not include message.content: {response}")
    return json.loads(raw_text), raw_text


def run_variant(variant: dict[str, Any], system_prompt: str, user_prompt: str, schema: dict[str, Any], timeout: int) -> tuple[dict[str, Any], str]:
    provider = variant["provider"]
    if provider == "openai":
        return run_openai_variant(variant, system_prompt, user_prompt, schema, timeout)
    if provider == "ollama":
        return run_ollama_variant(variant, system_prompt, user_prompt, schema, timeout)
    raise ValueError(f"Unsupported provider: {provider}")


def existing_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    records = load_predictions_file(path)
    return {(str(record["complaint_id"]), record["construct"]) for record in records if not record.get("error")}


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    variants = [variant for variant in config["variants"] if variant.get("enabled", True)]
    if args.variants:
        wanted = set(args.variants)
        variants = [variant for variant in variants if variant["name"] in wanted]
    if not variants:
        raise SystemExit("No enabled variants selected.")

    dataset = pd.read_csv(args.data)
    if args.limit:
        dataset = dataset.head(args.limit)

    run_name = args.run_name or f"{utc_timestamp_slug()}-{slugify(Path(args.config).stem)}"
    run_dir = Path(args.runs_dir) / run_name
    if run_dir.exists() and args.overwrite:
        for child in sorted(run_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    ensure_dir(run_dir)
    predictions_dir = ensure_dir(run_dir / "predictions")
    manifest = {
        "created_utc": utc_timestamp_slug(),
        "config_path": str(Path(args.config).resolve()),
        "data_path": str(Path(args.data).resolve()),
        "rows": len(dataset),
        "constructs": args.constructs,
        "variants": variants,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Run directory: {run_dir}")
    print(f"Rows: {len(dataset)} | Constructs: {', '.join(args.constructs)} | Variants: {len(variants)}")

    for variant in variants:
        variant_path = predictions_dir / f"{slugify(variant['name'])}.jsonl"
        completed = existing_pairs(variant_path)
        total_calls = len(dataset) * len(args.constructs)
        print(f"\n=== {variant['name']} ({variant['provider']} / {variant['model']}) ===")
        print(f"Prediction file: {variant_path}")
        if completed:
            print(f"Resuming with {len(completed)} completed outputs already on disk.")

        for row_idx, row in dataset.iterrows():
            complaint_id = str(row["Complaint ID"])
            narrative = str(row["Consumer complaint narrative"])
            for construct in args.constructs:
                if (complaint_id, construct) in completed:
                    continue

                schema = schema_for_construct(construct)
                system_prompt, user_prompt = build_prompt(
                    strategy=variant["prompt_strategy"],
                    construct=construct,
                    narrative=narrative,
                    schema=schema,
                )
                started = time.time()
                record: dict[str, Any] = {
                    "variant_name": variant["name"],
                    "provider": variant["provider"],
                    "model": variant["model"],
                    "prompt_strategy": variant["prompt_strategy"],
                    "row_index": int(row_idx),
                    "complaint_id": complaint_id,
                    "construct": construct,
                }
                try:
                    parsed, raw_text = run_variant(variant, system_prompt, user_prompt, schema, args.timeout)
                    record["parsed_response"] = parsed
                    record["normalized_prediction"] = normalize_prediction(construct, parsed)
                    record["raw_text"] = raw_text
                    record["latency_seconds"] = round(time.time() - started, 3)
                    record["error"] = None
                    completed.add((complaint_id, construct))
                except Exception as exc:  # noqa: BLE001
                    record["parsed_response"] = None
                    record["normalized_prediction"] = None
                    record["raw_text"] = None
                    record["latency_seconds"] = round(time.time() - started, 3)
                    record["error"] = str(exc)
                    print(f"[error] {variant['name']} | complaint {complaint_id} | {construct}: {exc}", file=sys.stderr)
                write_jsonl(variant_path, record)
                print(
                    f"[{variant['name']}] complaint {complaint_id} | {construct} "
                    f"| {len(completed)}/{total_calls} completed",
                    flush=True,
                )

    print("\nFinished. Run the evaluator next:")
    print(f"python inference_eval/evaluate_variants.py --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
