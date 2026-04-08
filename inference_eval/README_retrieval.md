# Retrieval-Few-Shot Pipeline

This document isolates the retrieval-specific path inside the inference pipeline: how retrieved reference cases are built, how they are injected into prompts, how to run retrieval variants locally or through OpenAI batch mode, and where retrieval metadata lands in the outputs.

## What The Retrieval Pipeline Does

The retrieval pipeline uses [`retrieval.py`](/Users/kevin/school/532/project_repo/inference_eval/retrieval.py) to build an index over the merged adjudicated workbook. It currently supports two similarity backends:

- `tfidf`: sparse TF-IDF vectors with cosine similarity
- `sentence_transformer`: dense sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2` with cosine similarity

For each target complaint and construct:

- the pipeline finds the nearest similar narratives under the configured backend
- it applies a leave-one-out rule so the target complaint is never retrieved as its own reference
- it keeps only examples that have usable ground-truth labels for that construct
- it injects the top `k` retrieved cases into the `retrieval_few_shot` prompt

Important constraint: this is a transductive setup, not a leakage-free benchmark. The retrieval pool comes from the same merged workbook used for evaluation. Treat it as a retrieval-augmented annotation experiment, not as a final generalization estimate.

## High-Level Flow

See the dedicated diagram in [`diagrams/retrieval_pipeline.md`](/Users/kevin/school/532/project_repo/inference_eval/diagrams/retrieval_pipeline.md).

## Core Files

- [`retrieval.py`](/Users/kevin/school/532/project_repo/inference_eval/retrieval.py): builds the retrieval index and construct-specific gold JSON payloads
- [`prompt_templates.py`](/Users/kevin/school/532/project_repo/inference_eval/prompt_templates.py): adds the retrieved examples to the `retrieval_few_shot` prompt
- [`run_variants.py`](/Users/kevin/school/532/project_repo/inference_eval/run_variants.py): executes local retrieval variants and writes retrieval metadata into prediction rows
- [`prepare_openai_batch.py`](/Users/kevin/school/532/project_repo/inference_eval/prepare_openai_batch.py): prepares retrieval-aware OpenAI batch requests
- [`evaluate_variants.py`](/Users/kevin/school/532/project_repo/inference_eval/evaluate_variants.py): scores the normalized prediction files after the run
- [`configs/ollama_first.json`](/Users/kevin/school/532/project_repo/inference_eval/configs/ollama_first.json): includes both the TF-IDF and MiniLM retrieval variants

## Retrieval Rules

- Retrieval policy: `leave_one_out`
- Default retrieved exemplars: `k=2`
- Candidate pool before filtering: `3`
- Default backend: `tfidf`
- Optional semantic backend: `sentence_transformer`
- Default embedding model for semantic retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Similarity scoring: cosine similarity
- Reference payloads are construct-specific:
  - `c1`: bundled `C0 + C1`
  - `c2`: harms
  - `c3`: emotions
  - `c4`: financial activities

If a candidate lacks usable gold labels for a construct, it is skipped for that construct.

## Retrieval Backend Comparison

### TF-IDF Baseline

- lexical similarity
- sparse vectors
- no embedding model download
- current default backend

### Semantic Retrieval With MiniLM

- dense sentence embeddings
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- more robust when similar complaints use different wording
- requires `sentence-transformers`
- first run may download the embedding model weights

This MiniLM option is the new comparison variant for checking whether semantic retrieval improves over lexical TF-IDF retrieval.

## Local Retrieval Run

Install dependencies and make sure the local model is available:

```bash
pip install -r inference_eval/requirements.txt
ollama pull qwen3:8b
```

### Run The Existing TF-IDF Retrieval Variant

Run only the TF-IDF retrieval variant as a local smoke pass:

```bash
python inference_eval/run_variants.py \
  --config inference_eval/configs/ollama_first.json \
  --variants ollama-qwen3-retrieval-few-shot \
  --run-name retrieval-local-smoke \
  --limit 5 \
  --overwrite
```

### Run The New MiniLM Semantic Retrieval Variant

The config now includes:

- `ollama-qwen3-retrieval-few-shot-minilm`

It is set to `enabled: false` so it does not alter your default run matrix. To use it, switch that variant to `enabled: true` in [`ollama_first.json`](/Users/kevin/school/532/project_repo/inference_eval/configs/ollama_first.json), then run:

```bash
python inference_eval/run_variants.py \
  --config inference_eval/configs/ollama_first.json \
  --variants ollama-qwen3-retrieval-few-shot-minilm \
  --run-name retrieval-local-smoke-minilm \
  --limit 5 \
  --overwrite
```

If local inference stalls, use the hard-timeout controls:

```bash
python inference_eval/run_variants.py \
  --config inference_eval/configs/ollama_first.json \
  --variants ollama-qwen3-retrieval-few-shot-minilm \
  --run-name retrieval-local-smoke-minilm \
  --limit 5 \
  --hard-timeout 300 \
  --stall-cooldown 300 \
  --exit-on-stall \
  --overwrite
```

Evaluate either run:

```bash
python inference_eval/evaluate_variants.py \
  --run-dir inference_eval/results/retrieval-local-smoke
```

```bash
python inference_eval/evaluate_variants.py \
  --run-dir inference_eval/results/retrieval-local-smoke-minilm
```

The current TF-IDF smoke artifacts are already in:

- [`results/retrieval-local-smoke/manifest.json`](/Users/kevin/school/532/project_repo/inference_eval/results/retrieval-local-smoke/manifest.json)
- [`results/retrieval-local-smoke/predictions/ollama-qwen3-retrieval-few-shot.jsonl`](/Users/kevin/school/532/project_repo/inference_eval/results/retrieval-local-smoke/predictions/ollama-qwen3-retrieval-few-shot.jsonl)
- [`results/retrieval-local-smoke/evaluation/metrics_summary.md`](/Users/kevin/school/532/project_repo/inference_eval/results/retrieval-local-smoke/evaluation/metrics_summary.md)

## OpenAI Batch Retrieval Run

### TF-IDF Retrieval Batch

Prepare a retrieval-aware OpenAI batch:

```bash
python inference_eval/prepare_openai_batch.py \
  --model gpt-5-nano \
  --prompt-strategy retrieval_few_shot \
  --retrieval-k 2 \
  --run-name retrieval-batch
```

Prepare and submit in one step:

```bash
python inference_eval/prepare_openai_batch.py \
  --model gpt-5-nano \
  --prompt-strategy retrieval_few_shot \
  --retrieval-k 2 \
  --run-name retrieval-batch \
  --submit
```

Default OpenAI batch settings now use `reasoning_effort=low` and omit `max_output_tokens` unless you pass it explicitly. The same LLM-side defaults apply to the MiniLM retrieval batch.

### MiniLM Semantic Retrieval Batch

Prepare a semantic-retrieval batch with MiniLM:

```bash
python inference_eval/prepare_openai_batch.py \
  --model gpt-5-nano \
  --prompt-strategy retrieval_few_shot \
  --retrieval-backend sentence_transformer \
  --retrieval-embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-k 2 \
  --run-name retrieval-batch-minilm
```

Prepare and submit in one step:

```bash
python inference_eval/prepare_openai_batch.py \
  --model gpt-5-nano \
  --prompt-strategy retrieval_few_shot \
  --retrieval-backend sentence_transformer \
  --retrieval-embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-k 2 \
  --run-name retrieval-batch-minilm \
  --submit
```

Poll and download the batch:

```bash
python inference_eval/poll_openai_batch.py \
  --batch-dir inference_eval/results/batches/retrieval-batch \
  --wait \
  --download
```

For the MiniLM retrieval batch:

```bash
python inference_eval/poll_openai_batch.py \
  --batch-dir inference_eval/results/batches/retrieval-batch-minilm \
  --wait \
  --download
```

Import the batch outputs into the prediction format used by the evaluator:

```bash
python inference_eval/import_openai_batch.py \
  --batch-dir inference_eval/results/batches/retrieval-batch \
  --run-dir inference_eval/results \
  --overwrite
```

For the MiniLM retrieval batch:

```bash
python inference_eval/import_openai_batch.py \
  --batch-dir inference_eval/results/batches/retrieval-batch-minilm \
  --run-dir inference_eval/results \
  --overwrite
```

Then evaluate:

```bash
python inference_eval/evaluate_variants.py \
  --run-dir inference_eval/results
```

To also append a raw `retrieval-batch` appendix with completion counts and parsed sample outputs into `evaluation/metrics_summary.md`, run:

```bash
python inference_eval/evaluate_variants.py \
  --run-dir inference_eval/results \
  --batch-summary-dirs inference_eval/results/batches/retrieval-batch
```

If the batch has not been imported yet, refresh both the prediction file and `metrics_summary.md` in one shell command:

```bash
python inference_eval/import_openai_batch.py --batch-dir inference_eval/results/batches/retrieval-batch --run-dir inference_eval/results --overwrite && \
python inference_eval/evaluate_variants.py --run-dir inference_eval/results --batch-summary-dirs inference_eval/results/batches/retrieval-batch
```

There is already a TF-IDF retrieval batch-prep smoke example in:

- [`results/batches/retrieval-smoke/manifest.json`](/Users/kevin/school/532/project_repo/inference_eval/results/batches/retrieval-smoke/manifest.json)
- [`results/batches/retrieval-smoke/request_index.jsonl`](/Users/kevin/school/532/project_repo/inference_eval/results/batches/retrieval-smoke/request_index.jsonl)

## What Gets Written

### Manifest-Level Retrieval Metadata

Run manifests can include:

- `transductive_retrieval_experiment`
- `retrieval_policy`
- `retrieval_source_data_path`
- `retrieval_candidate_k`
- `retrieval_backends`
- `retrieval_embedding_models`

### Per-Prediction Retrieval Metadata

Retrieval prediction rows can include:

- `retrieval_policy`
- `retrieval_k`
- `retrieval_backend`
- `retrieval_embedding_model`
- `retrieved_complaint_ids`
- `retrieval_scores`
- `retrieved_row_indexes`

That makes it possible to audit which exemplars were shown to the model for a given prediction.

## Recommended Workflow

1. Run a small local TF-IDF retrieval smoke pass with `--limit 5`.
2. Run the same smoke pass again with the MiniLM semantic retrieval variant.
3. Inspect the prediction JSONL to confirm the retrieved complaint ids, retrieval backend, embedding model, and similarity scores look plausible.
4. Evaluate both smoke runs before spending time on a full run.
5. If you need larger-scale OpenAI retrieval runs, prepare a batch and archive the batch folder alongside the imported predictions.
6. Keep retrieval results separate from non-retrieval runs when comparing experiments unless you explicitly want a mixed leaderboard.
