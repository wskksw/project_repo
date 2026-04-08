#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_eval.common import (  # noqa: E402
    CONSTRUCTS,
    DEFAULT_DATASET,
    DEFAULT_RUNS_DIR,
    ensure_dir,
    load_dataset,
    schema_for_construct,
    slugify,
    utc_timestamp_slug,
)
from inference_eval.prompt_templates import build_prompt  # noqa: E402
from inference_eval.retrieval import (  # noqa: E402
    DEFAULT_RETRIEVAL_CANDIDATE_K,
    DEFAULT_RETRIEVAL_BACKEND,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    RETRIEVAL_POLICY,
    RetrievalIndex,
    retrieval_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SCHEMA_NAME_LEN = 64
DEFAULT_OPENAI_REASONING_EFFORT = "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an OpenAI Batch API JSONL file for CFPB inference runs."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATASET,
        help="Workbook or CSV with complaint narratives and merged labels.",
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="Base results directory. Batch files are written under <runs-dir>/batches/.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional batch subfolder name. Defaults to a timestamped gpt-5-nano batch folder.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N rows.")
    parser.add_argument(
        "--constructs",
        nargs="+",
        choices=CONSTRUCTS,
        default=list(CONSTRUCTS),
        help="Subset of bundled calls to include.",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=("zero_shot", "few_shot", "retrieval_few_shot"),
        default="few_shot",
        help="Prompt style used to build the requests.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Number of retrieved reference cases to inject when using retrieval_few_shot.",
    )
    parser.add_argument(
        "--retrieval-backend",
        choices=("tfidf", "sentence_transformer"),
        default=DEFAULT_RETRIEVAL_BACKEND,
        help="Retrieval similarity backend used when prompt-strategy is retrieval_few_shot.",
    )
    parser.add_argument(
        "--retrieval-embedding-model",
        default=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        help="Sentence-transformer model name used when retrieval-backend is sentence_transformer.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="OpenAI model name to place in each batch request.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Optional max_output_tokens for each Responses API request. Omit to let the model use its default output budget.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        help="Reasoning effort to include in each request body. Defaults to low.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing batch folder.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Upload requests.jsonl and create an OpenAI batch after preparing it.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window used when --submit is set.",
    )
    return parser.parse_args()


def clean_run_dir(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()


def load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def submit_batch(requests_path: Path, run_name: str, completion_window: str) -> dict[str, Any]:
    load_repo_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or place it in .env before using --submit.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with requests_path.open("rb") as fh:
        upload = client.files.create(file=fh, purpose="batch")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata={"run_name": run_name},
    )
    return {
        "submitted_utc": utc_timestamp_slug(),
        "file_id": upload.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/responses",
        "completion_window": completion_window,
    }


def build_request_body(
    *,
    model: str,
    prompt_strategy: str,
    construct: str,
    complaint_id: str,
    narrative: str,
    variant_name: str,
    row_index: int,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    exemplars: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema = schema_for_construct(construct)
    system_prompt, user_prompt = build_prompt(
        strategy=prompt_strategy,
        construct=construct,
        narrative=narrative,
        schema=schema,
        exemplars=exemplars,
    )
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": safe_schema_name(variant_name, construct),
                "schema": schema,
                "strict": True,
            }
        },
        "metadata": {
            "variant_name": str(variant_name),
            "prompt_strategy": str(prompt_strategy),
            "complaint_id": str(complaint_id),
            "construct": str(construct),
            "row_index": str(row_index),
        },
    }
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def safe_schema_name(variant_name: str, construct: str) -> str:
    base = f"{slugify(variant_name)}_{construct}_schema"
    if len(base) <= MAX_SCHEMA_NAME_LEN:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{construct}_{digest}"
    prefix_budget = MAX_SCHEMA_NAME_LEN - len(suffix)
    return f"{slugify(variant_name)[:prefix_budget]}{suffix}"


def retrieval_variant_name(model: str, prompt_strategy: str, backend: str, embedding_model: str) -> str:
    base = slugify(f"openai-{model}-{prompt_strategy}")
    if prompt_strategy != "retrieval_few_shot" or backend == DEFAULT_RETRIEVAL_BACKEND:
        return base
    embedding_slug = slugify(embedding_model.split("/")[-1])
    return slugify(f"{base}-{backend}-{embedding_slug}")


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.data)
    if args.limit:
        dataset = dataset.head(args.limit)

    retrieval_index = None
    retrieval_embedding_model = args.retrieval_embedding_model if args.retrieval_backend == "sentence_transformer" else None
    if args.prompt_strategy == "retrieval_few_shot":
        retrieval_index = RetrievalIndex.from_dataset(
            args.data,
            backend=args.retrieval_backend,
            embedding_model=args.retrieval_embedding_model,
        )

    variant_name = retrieval_variant_name(
        args.model,
        args.prompt_strategy,
        args.retrieval_backend,
        args.retrieval_embedding_model,
    )
    run_name = args.run_name or f"{variant_name}-batch-{utc_timestamp_slug()}"
    batches_root = Path(args.runs_dir) / "batches"
    run_dir = batches_root / run_name

    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Batch directory already exists: {run_dir}. Use --overwrite to replace it.")
        clean_run_dir(run_dir)
    ensure_dir(run_dir)

    requests_path = run_dir / "requests.jsonl"
    index_path = run_dir / "request_index.jsonl"

    rows_considered = len(dataset)
    rows_with_text = 0
    request_count = 0

    with requests_path.open("w", encoding="utf-8") as requests_fh, index_path.open(
        "w",
        encoding="utf-8",
    ) as index_fh:
        for row_idx, row in dataset.iterrows():
            narrative = row.get("Consumer complaint narrative", "")
            if narrative is None:
                continue
            narrative_text = str(narrative).strip()
            if not narrative_text:
                continue

            complaint_id = str(row["Complaint ID"])
            rows_with_text += 1

            for construct in args.constructs:
                exemplars = None
                request_retrieval = None
                if args.prompt_strategy == "retrieval_few_shot":
                    if retrieval_index is None:
                        raise RuntimeError("retrieval_few_shot selected but retrieval index was not initialized.")
                    retrieved = retrieval_index.retrieve(
                        complaint_id=complaint_id,
                        construct=construct,
                        narrative=narrative_text,
                        k=args.retrieval_k,
                        candidate_k=max(DEFAULT_RETRIEVAL_CANDIDATE_K, args.retrieval_k + 1),
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
                    request_retrieval = retrieval_metadata(
                        retrieved,
                        k=args.retrieval_k,
                        backend=args.retrieval_backend,
                        embedding_model=retrieval_embedding_model,
                    )
                custom_id = f"{slugify(variant_name)}::{row_idx}::{complaint_id}::{construct}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": build_request_body(
                        model=args.model,
                        prompt_strategy=args.prompt_strategy,
                        construct=construct,
                        complaint_id=complaint_id,
                        narrative=narrative_text,
                        variant_name=variant_name,
                        row_index=int(row_idx),
                        max_output_tokens=args.max_output_tokens,
                        reasoning_effort=args.reasoning_effort,
                        exemplars=exemplars,
                    ),
                }
                requests_fh.write(json.dumps(request, ensure_ascii=False) + "\n")
                request_index_record = {
                    "custom_id": custom_id,
                    "row_index": int(row_idx),
                    "complaint_id": complaint_id,
                    "construct": construct,
                    "variant_name": variant_name,
                }
                if request_retrieval:
                    request_index_record.update(request_retrieval)
                index_fh.write(
                    json.dumps(
                        request_index_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                request_count += 1

    manifest = {
        "created_utc": utc_timestamp_slug(),
        "data_path": str(Path(args.data).resolve()),
        "rows_considered": rows_considered,
        "rows_with_narratives": rows_with_text,
        "constructs": args.constructs,
        "calls_per_narrative": len(args.constructs),
        "variant_name": variant_name,
        "model": args.model,
        "prompt_strategy": args.prompt_strategy,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "request_count": request_count,
        "requests_path": str(requests_path.resolve()),
        "request_index_path": str(index_path.resolve()),
        "transductive_retrieval_experiment": args.prompt_strategy == "retrieval_few_shot",
        "retrieval_policy": RETRIEVAL_POLICY if args.prompt_strategy == "retrieval_few_shot" else None,
        "retrieval_k": args.retrieval_k if args.prompt_strategy == "retrieval_few_shot" else None,
        "retrieval_candidate_k": DEFAULT_RETRIEVAL_CANDIDATE_K if args.prompt_strategy == "retrieval_few_shot" else None,
        "retrieval_backend": args.retrieval_backend if args.prompt_strategy == "retrieval_few_shot" else None,
        "retrieval_embedding_model": retrieval_embedding_model,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    submission = None
    if args.submit:
        submission = submit_batch(requests_path, run_name, args.completion_window)
        submission_path = run_dir / "submission.json"
        submission_path.write_text(json.dumps(submission, indent=2))
        manifest["submission_path"] = str(submission_path.resolve())
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Batch directory: {run_dir}")
    print(f"Rows considered: {rows_considered}")
    print(f"Rows with narratives: {rows_with_text}")
    print(f"Constructs: {', '.join(args.constructs)}")
    print(f"Variant: {variant_name}")
    print(f"Requests written: {request_count}")
    print(f"Requests JSONL: {requests_path}")
    print(f"Request index: {index_path}")
    print()
    if submission is None:
        print("Next step:")
        print(
            "  Re-run with --submit, or upload requests.jsonl with the OpenAI Files API and create a batch against /v1/responses."
        )
    else:
        print(f"Submitted file_id: {submission['file_id']}")
        print(f"Submitted batch_id: {submission['batch_id']}")
        print(f"Initial status: {submission['status']}")
        print()
        print("Next step:")
        print(f"  python inference_eval/poll_openai_batch.py --batch-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
