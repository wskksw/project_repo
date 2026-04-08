# Inference And Evaluation Pipeline

This folder contains the prompt-variant evaluation scaffold for the CFPB fraud annotation project.

The current target file is [`annotation_folder/annotated_results_merged_100.xlsx`](/Users/kevin/school/532/project_repo/annotation_folder/annotated_results_merged_100.xlsx). The runner and evaluator now align to that merged workbook and the March 2026 annotation guide.

## What Changed

- The default dataset is now the merged Excel workbook, not the old provisional CSV.
- The pipeline still makes exactly `4` model calls per narrative.
- Call `c1` now bundles Construct `0` and Construct `1` so the four-call budget is preserved.
- Prompt strategies are now `zero_shot`, `few_shot`, and `retrieval_few_shot`.
- `retrieval_few_shot` uses leave-one-out retrieval over the current merged workbook, so it should be treated as a transductive experiment rather than a leakage-free benchmark.
- Local variants are the default matrix:
  - `qwen3:8b` x `zero_shot`
  - `qwen3:8b` x `few_shot`
  - `qwen3:8b` x `retrieval_few_shot`
- OpenAI variants are left commented out in the config file.
- Structured output is requested for every call. If a local model drifts out of schema, the runner falls back to JSON extraction/parsing from the raw text.
- Outputs now go to one stable directory by default: `inference_eval/results/`.
- The runner writes predictions row by row and resumes automatically from previously completed `complaint x construct x variant` outputs.
- `c3` evaluation now matches the merged target format:
  - `c3_code`
  - comma-separated emotion set
  - one overall tier
  - one set of inference-pattern labels

## Four-Call Layout

| Call Key | Annotation Scope | Target Columns |
| --- | --- | --- |
| `c1` | Construct 0 + Construct 1 | `c0_label`, `c0_evidence quotes`, `c1_score`, `c1_label`, `c1_evidence_quotes` |
| `c2` | Construct 2 | `c2_abs_code`, `c2_abs_amount`, `c2_rel_score`, `c2_evidence_quotes` |
| `c3` | Construct 3 | `c3_code`, `c3_emotions`, `c3_tier`, `c3_inference_pattern`, `c3_evidence_quotes` |
| `c4` | Construct 4 | `c4_score`, `c4_label`, `c4_evidence_quotes` |


## Quick Start

### 1. Pull local Ollama models

```bash
pip install -r inference_eval/requirements.txt
ollama pull gemma3:4b
ollama pull qwen3:8b
```

### 2. Run a smoke pass

```bash
python inference_eval/run_variants.py \
  --config inference_eval/configs/ollama_first.json \
  --limit 5
```

This writes into the stable output folder `inference_eval/results/` with:

- `manifest.json`
- `predictions/<variant>.jsonl`
- `evaluation/*` after you run the evaluator

If the process is interrupted, rerunning the same command will skip completed rows and continue from the next unfinished item.

If local inference sometimes hangs, use a hard timeout and stop-on-stall mode:

```bash
python inference_eval/run_variants.py \
  --config inference_eval/configs/ollama_first.json \
  --hard-timeout 300 \
  --stall-cooldown 300 \
  --exit-on-stall
```

That will abort any single call that hangs for more than 5 minutes, wait 5 minutes, then stop the script. Running the same command again resumes from the next unfinished row.

### 3. Prepare or Submit an OpenAI batch

This repo now includes a batch-prep utility that reuses the same prompt builder and schemas as the sync runner. It can either just write the batch files or write and submit them in one step.

```bash
python inference_eval/prepare_openai_batch.py \
  --model gpt-5-nano \
  --prompt-strategy few_shot
```

That writes a timestamped folder under `inference_eval/results/batches/` containing:

- `requests.jsonl`
- `request_index.jsonl`
- `manifest.json`

To prepare and submit in one command:

```bash
python inference_eval/prepare_openai_batch.py \
  --model gpt-5-nano \
  --prompt-strategy few_shot \
  --submit
```

When `--submit` is used, the batch folder also gets:

- `submission.json`

Use `--limit 5` for a smoke run or `--run-name my-batch --overwrite` to regenerate a specific batch folder.

OpenAI batch prep defaults now use `reasoning_effort=low` and do not send `max_output_tokens` unless you pass that flag explicitly.

### 4. Poll and download OpenAI batch results

Poll once:

```bash
python inference_eval/poll_openai_batch.py \
  --batch-dir inference_eval/results/batches/my-batch
```

Wait until completion:

```bash
python inference_eval/poll_openai_batch.py \
  --batch-dir inference_eval/results/batches/my-batch \
  --wait
```

Wait and download outputs into the same folder:

```bash
python inference_eval/poll_openai_batch.py \
  --batch-dir inference_eval/results/batches/my-batch \
  --wait \
  --download
```

That writes:

- `status.json`
- `output.jsonl` when the batch completes successfully
- `errors.jsonl` when OpenAI returns an error file

### 5. Import batch outputs into the prediction format

Batch polling writes raw OpenAI batch files. Convert them into the same normalized `predictions/*.jsonl` format used by local runs:

```bash
python inference_eval/import_openai_batch.py \
  --batch-dir inference_eval/results/batches/my-batch \
  --run-dir inference_eval/results \
  --overwrite
```

That writes `predictions/<variant>.jsonl`, which can then be scored alongside local-model outputs.

### 6. Evaluate the run

```bash
python inference_eval/evaluate_variants.py \
  --run-dir inference_eval/results
```

This writes:

- `evaluation/metrics_by_variant.csv`
- `evaluation/metrics_summary.md`

The evaluator scans every `predictions/*.jsonl` file in the run directory, so imported batch variants are included even if they were not listed in the current `manifest.json`.

## Output Semantics

The runner emits one prediction per `complaint x call bundle x variant`.

- `c1` is the bundled `C0 + C1` call.
- `c2`, `c3`, and `c4` remain one call each.
- The manifest records `calls_per_narrative=4`.
- Restarting the runner is safe because already-written outputs are reused.
- Retrieval variants also record `retrieval_policy`, `retrieval_k`, and the retrieved complaint ids in per-request outputs.

## Evaluation Metrics

- `c1`
  - `c0_label_accuracy`
  - `c1_score_accuracy`
- `c2`
  - `abs_code_accuracy`
  - `abs_amount_exact_match`
  - `rel_score_accuracy`
- `c3`
  - `code_accuracy`
  - emotion-label micro precision / recall / F1
  - emotion exact match
  - `tier_accuracy_on_positive`
  - pattern exact match and label metrics on tier-2 rows
- `c4`
  - `score_accuracy`

## Files

- [`run_variants.py`](/Users/kevin/school/532/project_repo/inference_eval/run_variants.py): executes the variant matrix and writes normalized predictions.
- [`evaluate_variants.py`](/Users/kevin/school/532/project_repo/inference_eval/evaluate_variants.py): scores predictions against the merged workbook and adds a majority baseline.
- [`prompt_templates.py`](/Users/kevin/school/532/project_repo/inference_eval/prompt_templates.py): prompt construction for `zero_shot`, `few_shot`, and `retrieval_few_shot`.
- [`retrieval.py`](/Users/kevin/school/532/project_repo/inference_eval/retrieval.py): LangChain TF-IDF retrieval and construct-specific exemplar rendering.
- [`common.py`](/Users/kevin/school/532/project_repo/inference_eval/common.py): target schemas, label normalization, dataset loading, and shared helpers.
- [`configs/ollama_first.json`](/Users/kevin/school/532/project_repo/inference_eval/configs/ollama_first.json): local-first variant config with commented OpenAI examples.

## Related Docs

- [`README_retrieval.md`](/Users/kevin/school/532/project_repo/inference_eval/README_retrieval.md): dedicated retrieval-few-shot guide with local and batch workflows.
- [`diagrams/retrieval_pipeline.md`](/Users/kevin/school/532/project_repo/inference_eval/diagrams/retrieval_pipeline.md): high-level retrieval pipeline diagram.
- [`analysis/README.md`](/Users/kevin/school/532/project_repo/inference_eval/analysis/README.md): notebook-based metrics analysis workspace and generated tables.
