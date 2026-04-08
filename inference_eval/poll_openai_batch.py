#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll an OpenAI batch and optionally download results into the batch folder."
    )
    parser.add_argument("--batch-dir", help="Batch directory created by prepare_openai_batch.py.")
    parser.add_argument("--batch-id", help="OpenAI batch ID. If omitted, read it from <batch-dir>/submission.json.")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the batch reaches a terminal state.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Polling interval in seconds when --wait is used.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download output and error files into the batch directory once available.",
    )
    return parser.parse_args()


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


def load_submission(batch_dir: Path) -> dict[str, Any]:
    submission_path = batch_dir / "submission.json"
    if not submission_path.exists():
        raise SystemExit(f"Missing submission metadata: {submission_path}")
    return json.loads(submission_path.read_text(encoding="utf-8"))


def summarize_batch(batch: Any) -> dict[str, Any]:
    counts = getattr(batch, "request_counts", None)
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "request_counts": {
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
        },
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def download_file(client: Any, file_id: str, path: Path) -> None:
    content = client.files.content(file_id)
    path.write_bytes(content.content)


def main() -> int:
    args = parse_args()
    if not args.batch_dir and not args.batch_id:
        raise SystemExit("Provide --batch-dir or --batch-id.")

    batch_dir = Path(args.batch_dir) if args.batch_dir else None
    batch_id = args.batch_id
    if batch_id is None and batch_dir is not None:
        batch_id = load_submission(batch_dir)["batch_id"]
    if batch_id is None:
        raise SystemExit("Could not determine batch_id.")

    load_repo_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or place it in .env before polling.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    summary = summarize_batch(batch)

    while args.wait and summary["status"] not in TERMINAL_STATES:
        print(
            f"[{summary['status']}] total={summary['request_counts']['total']} "
            f"completed={summary['request_counts']['completed']} failed={summary['request_counts']['failed']}",
            flush=True,
        )
        time.sleep(args.interval)
        batch = client.batches.retrieve(batch_id)
        summary = summarize_batch(batch)

    print(json.dumps(summary, indent=2))

    if batch_dir is not None:
        (batch_dir / "status.json").write_text(json.dumps(summary, indent=2))

    if args.download:
        if batch_dir is None:
            raise SystemExit("--download requires --batch-dir so files have a destination.")
        if summary["output_file_id"]:
            download_file(client, summary["output_file_id"], batch_dir / "output.jsonl")
        if summary["error_file_id"]:
            download_file(client, summary["error_file_id"], batch_dir / "errors.jsonl")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
