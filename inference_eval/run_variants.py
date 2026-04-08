#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_eval.common import (
    CONSTRUCTS,
    DEFAULT_DATASET,
    DEFAULT_RUNS_DIR,
    ensure_dir,
    load_dataset,
    load_predictions_file,
    normalize_prediction,
    schema_for_construct,
    slugify,
    utc_timestamp_slug,
    write_jsonl,
)
from inference_eval.prompt_templates import build_prompt
from inference_eval.retrieval import (
    DEFAULT_RETRIEVAL_CANDIDATE_K,
    DEFAULT_RETRIEVAL_BACKEND,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    RETRIEVAL_POLICY,
    RetrievalIndex,
    retrieval_metadata,
)
MAX_SCHEMA_NAME_LEN = 64
DEFAULT_OPENAI_REASONING_EFFORT = "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extractor variants against CFPB complaint narratives.")
    parser.add_argument("--config", required=True, help="Path to variant config JSON/JSONC.")
    parser.add_argument("--data", default=DEFAULT_DATASET, help="Workbook or CSV with narratives and merged labels.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Stable output directory used for predictions, manifest, and evaluation files.")
    parser.add_argument("--run-name", help="Optional subfolder inside the output directory. If omitted, outputs are written directly into --runs-dir.")
    parser.add_argument("--limit", type=int, help="Only process the first N rows.")
    parser.add_argument("--constructs", nargs="+", choices=CONSTRUCTS, default=list(CONSTRUCTS))
    parser.add_argument("--variants", nargs="+", help="Optional subset of variant names to execute.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run directory.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument("--hard-timeout", type=int, help="Wall-clock timeout for a single model call in seconds.")
    parser.add_argument("--stall-cooldown", type=int, default=0, help="Seconds to sleep after a hard-timeout before continuing or exiting.")
    parser.add_argument("--exit-on-stall", action="store_true", help="Exit the whole script after a hard-timeout so you can restart and resume later.")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(strip_json_comments(Path(path).read_text()))


def strip_json_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    escaped = False
    while i < len(text):
        char = text[i]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < len(text) and text[i] not in "\r\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        result.append(char)
        i += 1
    return "".join(result)


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


class CallTimedOutError(RuntimeError):
    pass


def run_with_hard_timeout(timeout_seconds: int | None, fn: Any, *args: Any, **kwargs: Any) -> Any:
    if not timeout_seconds:
        return fn(*args, **kwargs)
    if not hasattr(signal, "setitimer"):
        return fn(*args, **kwargs)

    def _handle_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise CallTimedOutError(f"Call exceeded hard timeout of {timeout_seconds} seconds.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


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
                parsed, end = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise RuntimeError(f"Could not parse model output as JSON object: {raw_text[:500]}")


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


def safe_schema_name(variant_name: str, construct: str) -> str:
    base = f"{variant_name}_{construct}_schema"
    if len(base) <= MAX_SCHEMA_NAME_LEN:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{construct}_{digest}"
    prefix_budget = MAX_SCHEMA_NAME_LEN - len(suffix)
    return f"{variant_name[:prefix_budget]}{suffix}"


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
                "name": safe_schema_name(variant["name"], "response"),
                "schema": schema,
                "strict": True,
            }
        },
    }
    if variant.get("max_output_tokens") is not None:
        payload["max_output_tokens"] = variant["max_output_tokens"]
    reasoning_effort = variant.get("reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT)
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    response = http_post_json(
        url=variant.get("base_url", "https://api.openai.com/v1/responses"),
        payload=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    raw_text = extract_openai_output(response)
    return parse_json_response(raw_text), raw_text


def run_ollama_variant(variant: dict[str, Any], system_prompt: str, user_prompt: str, schema: dict[str, Any], timeout: int) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = {
        "model": variant["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": schema,
        "stream": False,
        "think": variant.get("think", False),
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
    return parse_json_response(raw_text), raw_text


def run_variant(
    variant: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    timeout: int,
    hard_timeout: int | None = None,
) -> tuple[dict[str, Any], str]:
    provider = variant["provider"]
    if provider == "openai":
        return run_with_hard_timeout(hard_timeout, run_openai_variant, variant, system_prompt, user_prompt, schema, timeout)
    if provider == "ollama":
        return run_with_hard_timeout(hard_timeout, run_ollama_variant, variant, system_prompt, user_prompt, schema, timeout)
    raise ValueError(f"Unsupported provider: {provider}")


def existing_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    records = load_predictions_file(path)
    return {(str(record["complaint_id"]), record["construct"]) for record in records}


def retrieval_settings(variant: dict[str, Any]) -> tuple[str, str | None]:
    backend = str(variant.get("retrieval_backend") or DEFAULT_RETRIEVAL_BACKEND)
    embedding_model = variant.get("retrieval_embedding_model")
    if backend == "sentence_transformer":
        embedding_model = str(embedding_model or DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    else:
        embedding_model = None
    return backend, embedding_model


def retrieval_cache_key(data_path: str, backend: str, embedding_model: str | None) -> tuple[str, str, str | None]:
    return (str(Path(data_path).resolve()), backend, embedding_model)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    all_variants = list(config["variants"])
    if args.variants:
        wanted = set(args.variants)
        variants = [variant for variant in all_variants if variant["name"] in wanted]
    else:
        variants = [variant for variant in all_variants if variant.get("enabled", True)]
    if not variants:
        if args.variants:
            raise SystemExit("No matching variants selected from config.")
        raise SystemExit("No enabled variants selected.")

    dataset = load_dataset(args.data)
    if args.limit:
        dataset = dataset.head(args.limit)

    retrieval_variants = [variant for variant in variants if variant.get("prompt_strategy") == "retrieval_few_shot"]
    needs_retrieval = bool(retrieval_variants)
    retrieval_indexes: dict[tuple[str, str, str | None], RetrievalIndex] = {}
    retrieval_backends: list[str] = []
    retrieval_embedding_models: list[str] = []
    if needs_retrieval:
        for variant in retrieval_variants:
            backend, embedding_model = retrieval_settings(variant)
            cache_key = retrieval_cache_key(args.data, backend, embedding_model)
            if cache_key in retrieval_indexes:
                continue
            retrieval_indexes[cache_key] = RetrievalIndex.from_dataset(
                args.data,
                backend=backend,
                embedding_model=embedding_model or DEFAULT_SENTENCE_TRANSFORMER_MODEL,
            )
            retrieval_backends.append(backend)
            if embedding_model:
                retrieval_embedding_models.append(embedding_model)

    run_dir = Path(args.runs_dir)
    if args.run_name:
        run_dir = run_dir / args.run_name
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
        "config_name": slugify(Path(args.config).stem),
        "rows": len(dataset),
        "constructs": args.constructs,
        "calls_per_narrative": len(args.constructs),
        "call_layout": {
            "c1": "Construct 0 + Construct 1",
            "c2": "Construct 2",
            "c3": "Construct 3",
            "c4": "Construct 4",
        },
        "transductive_retrieval_experiment": needs_retrieval,
        "retrieval_policy": RETRIEVAL_POLICY if needs_retrieval else None,
        "retrieval_source_data_path": str(Path(args.data).resolve()) if needs_retrieval else None,
        "retrieval_candidate_k": DEFAULT_RETRIEVAL_CANDIDATE_K if needs_retrieval else None,
        "retrieval_backends": sorted(set(retrieval_backends)) if needs_retrieval else [],
        "retrieval_embedding_models": sorted(set(retrieval_embedding_models)) if needs_retrieval else [],
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
                exemplars = None
                retrieval_info = None
                if variant.get("prompt_strategy") == "retrieval_few_shot":
                    backend, embedding_model = retrieval_settings(variant)
                    cache_key = retrieval_cache_key(args.data, backend, embedding_model)
                    retrieval_index = retrieval_indexes.get(cache_key)
                    if retrieval_index is None:
                        raise RuntimeError("retrieval_few_shot selected but retrieval index was not initialized.")
                    retrieval_k = int(variant.get("retrieval_k", DEFAULT_RETRIEVAL_K))
                    retrieved = retrieval_index.retrieve(
                        complaint_id=complaint_id,
                        construct=construct,
                        narrative=narrative,
                        k=retrieval_k,
                        candidate_k=max(DEFAULT_RETRIEVAL_CANDIDATE_K, retrieval_k + 1),
                    )
                    exemplars = [
                        {
                            "complaint_id": item.complaint_id,
                            "row_index": item.row_index,
                            "similarity_rank": item.similarity_rank,
                            "similarity_score": item.similarity_score,
                            "narrative_excerpt": item.narrative_excerpt,
                            "gold_json": item.gold_json,
                        }
                        for item in retrieved
                    ]
                    retrieval_info = retrieval_metadata(
                        retrieved,
                        k=retrieval_k,
                        backend=backend,
                        embedding_model=embedding_model,
                    )
                system_prompt, user_prompt = build_prompt(
                    strategy=variant["prompt_strategy"],
                    construct=construct,
                    narrative=narrative,
                    schema=schema,
                    exemplars=exemplars,
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
                if retrieval_info:
                    record.update(retrieval_info)
                stalled = False
                print(
                    f"[start] {variant['name']} | complaint {complaint_id} | {construct}",
                    flush=True,
                )
                interrupted = False
                try:
                    parsed, raw_text = run_variant(
                        variant,
                        system_prompt,
                        user_prompt,
                        schema,
                        args.timeout,
                        args.hard_timeout,
                    )
                    record["parsed_response"] = parsed
                    record["normalized_prediction"] = normalize_prediction(construct, parsed)
                    record["raw_text"] = raw_text
                    record["latency_seconds"] = round(time.time() - started, 3)
                    record["error"] = None
                    completed.add((complaint_id, construct))
                    write_jsonl(variant_path, record)
                    print(
                        f"[{variant['name']}] complaint {complaint_id} | {construct} "
                        f"| {len(completed)}/{total_calls} completed",
                        flush=True,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    record["parsed_response"] = None
                    record["normalized_prediction"] = None
                    record["raw_text"] = None
                    record["latency_seconds"] = round(time.time() - started, 3)
                    record["error"] = "KeyboardInterrupt"
                    write_jsonl(variant_path, record)
                    completed.add((complaint_id, construct))
                    print(f"\n[interrupted] {variant['name']} | complaint {complaint_id} | {construct}: saved error record before exit.", file=sys.stderr)
                except Exception as exc:  # noqa: BLE001
                    stalled = isinstance(exc, CallTimedOutError)
                    record["parsed_response"] = None
                    record["normalized_prediction"] = None
                    record["raw_text"] = None
                    record["latency_seconds"] = round(time.time() - started, 3)
                    record["error"] = str(exc)
                    write_jsonl(variant_path, record)
                    completed.add((complaint_id, construct))
                    print(f"[error] {variant['name']} | complaint {complaint_id} | {construct}: {exc}", file=sys.stderr)
                if interrupted:
                    raise KeyboardInterrupt
                if stalled and args.stall_cooldown > 0:
                    print(
                        f"\nCooling down for {args.stall_cooldown} seconds after stall for "
                        f"{variant['name']} | complaint {complaint_id} | {construct}.",
                        file=sys.stderr,
                    )
                    time.sleep(args.stall_cooldown)
                if stalled and args.exit_on_stall:
                    print(
                        f"\nStopping after stall for {variant['name']} | complaint {complaint_id} | {construct}. "
                        "Restart the same command to resume.",
                        file=sys.stderr,
                    )
                    return 1

    print("\nFinished. Run the evaluator next:")
    print(f"python inference_eval/evaluate_variants.py --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
