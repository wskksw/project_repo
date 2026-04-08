# Inference Pipelines For Presentation Generation

This document is a presentation-facing explanation of the inference stack implemented in `inference_eval/`. It is written to be fed into another LLM or copied into slides, so it focuses on the system design, the variant taxonomy, the prompt strategies, the retrieval path, and the evaluation logic without assuming the reader has the code open.

## Executive Summary

The project evaluates multiple LLM-based annotation pipelines on CFPB fraud complaint narratives. The system is not a single monolithic prompt. It is a controlled evaluation framework with shared data handling and scoring, plus interchangeable inference variants.

Across all variants, the pipeline keeps the following core structure fixed:

1. Load the same merged adjudicated complaint workbook.
2. Split annotation into four model calls per complaint.
3. Build a construct-specific prompt and JSON schema for each call.
4. Send the request through a selected inference provider and model.
5. Normalize the raw model output into a common prediction format.
6. Score the predictions against the merged ground truth using construct-specific metrics.

What changes between variants is not the task definition itself, but the inference strategy:

- model family
- provider
- prompt strategy
- retrieval backend for retrieval-augmented prompting
- execution mode for OpenAI: synchronous calls vs Batch API

This lets the project compare pipeline design choices while keeping data, annotation targets, schemas, and evaluation logic aligned.

## The Constant Backbone Of Every Pipeline

### Dataset

The default dataset is:

- `annotation_folder/annotated_results_merged_100.xlsx`

This workbook contains the complaint narratives and the merged/adjudicated target annotations used for evaluation.

### Four-call decomposition

The system uses exactly four model calls per narrative. This is a deliberate design decision that balances construct separation against runtime cost.

| Call key | What it codes | Why it is grouped this way |
| --- | --- | --- |
| `c1` | Construct 0 and Construct 1 together | C0 and C1 are bundled to preserve a four-call budget |
| `c2` | Construct 2 | Economic harm is handled independently |
| `c3` | Construct 3 | Emotion coding requires its own decision logic |
| `c4` | Construct 4 | Financial activity / digital asset engagement is separated from harms and emotions |

This means the pipeline is not doing one giant end-to-end annotation prompt. It is doing a structured multi-call inference workflow.

### Shared schema discipline

Every construct has a strict JSON schema defined in `common.py`. The prompt tells the model to output only JSON, and the runner normalizes the response into a common shape before evaluation.

The schemas matter because they make all variants comparable. Even when models differ, the output contract stays constant.

### Shared normalization and evaluation

All variants are funneled through the same normalization functions in `common.py` and the same evaluator in `evaluate_variants.py`.

That means the comparison is controlled:

- same dataset
- same construct definitions
- same output schema targets
- same normalization rules
- same metrics

## What Actually Varies Across Pipelines

The codebase explores several axes of variation.

### 1. Provider axis

- `ollama`: local inference through the Ollama chat API
- `openai`: hosted inference through the OpenAI Responses API

### 2. Model axis

The repo currently includes or references these model families:

- `qwen3:8b`
- `gemma3:4b`
- `gpt-5-nano`
- optional commented OpenAI variants such as `gpt-5.4-mini` and `gpt-5.4`

### 3. Prompt strategy axis

There are three implemented prompt strategies:

- `zero_shot`
- `few_shot`
- `retrieval_few_shot`

These are the most important slide-worthy variants, because they change how much task guidance and example structure the model receives.

### 4. Retrieval backend axis

For `retrieval_few_shot`, the retrieval subsystem has two backends:

- `tfidf`
- `sentence_transformer` using `sentence-transformers/all-MiniLM-L6-v2`

### 5. Execution mode axis

For OpenAI, there are two operational modes:

- synchronous execution through `run_variants.py`
- asynchronous batch execution through `prepare_openai_batch.py`, `poll_openai_batch.py`, and `import_openai_batch.py`

These execution modes do not change the task definition, but they do change how inference is operationalized at scale.

## High-Level Variant Taxonomy

The cleanest way to explain the system in a presentation is:

1. Fixed annotation/evaluation framework.
2. Multiple interchangeable inference pipelines inside that framework.

Those pipelines can be described as combinations of:

- provider: local or hosted
- model: Qwen, Gemma, GPT-5
- prompting: zero-shot, few-shot, or retrieval-augmented few-shot
- retrieval backend: TF-IDF or MiniLM embeddings when retrieval is enabled
- execution mode: online sync or offline batch for OpenAI

In other words, the project is comparing pipeline configurations, not just comparing models.

## Prompting Strategies In Detail

## `zero_shot`

`zero_shot` gives the model construct-specific instructions, allowed labels, and the strict schema, but no worked examples.

What it contains:

- explanation of the construct being coded
- label inventory
- construct-specific decision rules
- the exact JSON schema
- the target narrative

What it does not contain:

- worked examples
- retrieved exemplars from the dataset

Interpretation:

- this is the cleanest test of whether the model can follow the annotation instructions directly
- it is the lowest-context prompting condition
- it gives the weakest scaffolding but also the least prompt overhead

## `few_shot`

`few_shot` adds worked decision rules and example mappings inside the prompt instructions, but those examples are hand-authored prompt content, not dynamically retrieved complaint cases.

What it contains:

- everything in `zero_shot`
- richer annotation heuristics
- worked examples embedded in the instruction text

Why it matters:

- it tests whether richer task framing improves annotation consistency
- it provides more concept-level guidance without using dataset retrieval
- it is still a static prompt strategy

Important nuance:

The few-shot examples are not inserted as separate JSON demonstrations from the evaluation dataset. They are verbal worked examples embedded in the prompt template.

## `retrieval_few_shot`

`retrieval_few_shot` starts from the few-shot instruction set, then injects dynamically retrieved reference complaints and their gold annotations into the prompt.

What it contains:

- the full few-shot instruction block
- top-k similar retrieved reference cases
- complaint excerpts from those cases
- the gold JSON annotation for each retrieved case
- the target narrative

What makes it different:

- it is not just more instruction; it is retrieval-augmented inference
- the examples are dynamically chosen based on narrative similarity
- each target complaint gets a different reference set

Why this is interesting experimentally:

- it tests whether similar annotated cases help the model map ambiguous narratives to the coding scheme
- it approximates a retrieval-augmented annotation assistant rather than a static prompt-only annotator

Important caveat:

This retrieval setup is transductive, not leakage-free. The retrieval pool comes from the same merged workbook used for evaluation. The target complaint is excluded from retrieving itself, but retrieved examples still come from the evaluation corpus. So this should be presented as a retrieval-augmented annotation experiment, not as a strict out-of-sample generalization benchmark.

## Construct-by-Construct Prompt Logic

Each of the four calls has distinct prompt content because the constructs are semantically different.

### `c1`: Construct 0 plus Construct 1

This bundled call asks the model to jointly annotate:

- what type of user information was leaked
- the perceived information vulnerability level

Why this is bundled:

- C0 and C1 are tightly related
- bundling preserves a four-call runtime budget

The prompt emphasizes:

- choose the highest leaked-information category present for C0
- assess C1 from the victim's framing of ongoing risk, not just objective data sensitivity
- return evidence separately for the two sub-constructs

### `c2`: Construct 2

This call handles harms, especially economic harms.

The prompt separates:

- whether financial loss is present
- whether the amount is exact, unspecified, or redacted
- how severe the loss is from the victim's own framing

The critical design point is that relative significance is not judged from dollar value alone. It is grounded in the narrative framing.

### `c3`: Construct 3

This is the emotion pipeline and is one of the most complex calls.

The prompt asks the model to produce:

- emotion count code
- canonical emotion labels
- evidence tier
- inference patterns
- evidence quotes

This call has more nuanced logic than the others because it distinguishes:

- no emotion
- one emotion vs multiple emotions
- explicit emotion language vs inferred emotion
- different inference pattern categories for tier-2 emotion evidence

This is why C3 often deserves its own slide in a presentation.

### `c4`: Construct 4

This call codes financial activities / digital asset engagement.

The prompt distinguishes between:

- no digital asset engagement
- online transfers and transactions
- broader digital asset engagement

The design intent is to code the victim's financial behavior and engagement profile, not merely the fraud channel.

## Retrieval-Augmented Pipeline In Detail

The retrieval pipeline is implemented in `retrieval.py`.

### Retrieval source

The retrieval index is built from the same merged annotated workbook used for evaluation.

### Retrieval policy

- leave-one-out retrieval
- the target complaint cannot retrieve itself

### Retrieval filtering

Retrieved candidates are only usable if they have valid gold labels for the target construct.

This is important because retrieval is construct-specific. A complaint can be usable as a reference for one construct but unusable for another if labels are missing.

### Retrieval backends

#### TF-IDF retrieval

TF-IDF is the default backend.

Characteristics:

- sparse lexical representation
- cosine similarity
- cheaper and simpler
- does not require downloading an embedding model

Interpretation:

- good baseline for wording-level similarity
- likely to work best when similar complaints use similar surface phrasing

#### Sentence-transformer retrieval with MiniLM

The alternative backend uses dense embeddings from:

- `sentence-transformers/all-MiniLM-L6-v2`

Characteristics:

- dense semantic representation
- cosine similarity over normalized embeddings
- more robust to paraphrase than TF-IDF
- requires the sentence-transformers dependency and model weights

Interpretation:

- this variant tests whether semantic retrieval improves over lexical retrieval
- it is especially useful when similar fraud cases are described with different wording

### Retrieval payload injected into prompts

For each retrieved example, the prompt includes:

- complaint id
- similarity score
- narrative excerpt
- gold JSON for the relevant construct

This means the model sees both analogous narrative text and the corresponding gold annotation structure.

### Retrieval metadata written to outputs

Retrieval variants also store metadata in prediction records, including:

- retrieval policy
- retrieval k
- retrieval backend
- embedding model when relevant
- retrieved complaint ids
- retrieval scores
- retrieved row indexes

This is useful for both auditing and slide generation because it makes the retrieval path inspectable.

## Local Ollama Pipeline

The local execution path is `run_variants.py` with provider `ollama`.

High-level flow:

1. Load config variants from JSON/JSONC.
2. Build the shared dataset and optional retrieval indexes.
3. For each complaint and construct, build the schema and prompt.
4. Send the prompt to Ollama's chat endpoint.
5. Parse and normalize the response.
6. Append one JSONL record per complaint-construct pair.

Key operational properties:

- local inference through `http://localhost:11434/api/chat`
- structured output requested from the model
- JSON parsing fallback for models that drift slightly from schema
- row-by-row writing for resumability
- optional hard timeout and stall handling

Why this matters in a presentation:

- this is the self-hosted experimental path
- it makes it easy to compare cheaper local models against OpenAI-hosted models under the same task framing

## OpenAI Synchronous Pipeline

The synchronous OpenAI path also runs through `run_variants.py`, but uses the OpenAI Responses API instead of Ollama.

High-level flow:

1. Build the same prompt and schema used in the local path.
2. Send the request to `/v1/responses`.
3. Use JSON schema output formatting with strict mode.
4. Parse and normalize the result.
5. Write it into the same normalized prediction format.

Key operational properties:

- same task logic as local inference
- different provider and model family
- OpenAI reasoning effort defaults to `low` in current tooling
- output records are directly comparable with Ollama outputs

This is important for the presentation because it separates model/provider effects from pipeline-definition effects.

## OpenAI Batch Pipeline

The batch path is operationally different, even though the prompt logic is the same.

It has three stages:

1. `prepare_openai_batch.py`
2. `poll_openai_batch.py`
3. `import_openai_batch.py`

### Stage 1: Prepare requests

`prepare_openai_batch.py` builds a `requests.jsonl` file for the OpenAI Batch API.

It uses:

- the same construct schemas
- the same prompt templates
- the same retrieval logic when `retrieval_few_shot` is selected

This is a scaling mechanism, not a different annotation definition.

### Stage 2: Poll batch status

`poll_openai_batch.py` monitors the batch and optionally downloads:

- `status.json`
- `output.jsonl`
- `errors.jsonl`

### Stage 3: Import into the common prediction format

`import_openai_batch.py` converts raw batch outputs into the same normalized `predictions/*.jsonl` format used by synchronous runs.

This is one of the strongest design choices in the repo because it unifies downstream evaluation. Once imported, batch outputs are scored exactly like local or synchronous outputs.

## Resilience And Reproducibility Features

These are useful talking points because they show the project is an engineering pipeline, not just a one-off prompt script.

### Stable output layout

Outputs are written under:

- `inference_eval/results/`

Typical artifacts include:

- `manifest.json`
- `predictions/*.jsonl`
- `evaluation/metrics_by_variant.csv`
- `evaluation/metrics_summary.md`
- batch-specific folders under `results/batches/`

### Resume behavior

The runner writes outputs incrementally and skips complaint-construct pairs that already exist on disk.

That means interrupted runs can resume without restarting from scratch.

### Error capture

If a call fails, the record still gets written with error metadata. This keeps the run auditable and prevents silent data loss.

### Manifest recording

Run manifests store:

- config path
- data path
- construct set
- calls per narrative
- retrieval settings
- active variants

This is useful for reproducibility and presentation traceability.

## Evaluation Pipeline

Inference outputs are evaluated by `evaluate_variants.py`.

The evaluator:

1. Loads the merged ground truth.
2. Loads all prediction files in the run directory.
3. Normalizes and aligns prediction records.
4. Scores each variant by construct-specific metrics.
5. Writes a CSV and a human-readable Markdown summary.

### Why the evaluation is construct-specific

Different constructs require different notions of correctness. A single metric would collapse important distinctions.

### Metrics by construct

#### `c1`

- `c0_label_accuracy`
- `c1_score_accuracy`

#### `c2`

- `abs_code_accuracy`
- `abs_amount_exact_match`
- `rel_score_accuracy`

#### `c3`

- `code_accuracy`
- emotion precision, recall, and F1
- emotion exact match
- `tier_accuracy_on_positive`
- inference-pattern exact match and label metrics on tier-2 rows

#### `c4`

- `score_accuracy`

This metric design is presentation-worthy because it shows the evaluation respects the structure of the annotation task rather than flattening everything into one accuracy number.

## How To Explain The Pipeline Variants In Slides

A clean slide narrative is:

1. Start with the shared task framing.
2. Explain the four-call decomposition.
3. Explain the three prompt strategies.
4. Explain retrieval as the main pipeline augmentation.
5. Explain local vs OpenAI vs batch as execution environments.
6. End with construct-specific evaluation metrics.

One useful framing is:

- baseline question: can a model follow the annotation codebook directly
- prompt-engineering question: does richer prompting help
- retrieval question: do analogous annotated cases help
- systems question: how do local and hosted inference pipelines compare operationally

## Current Variant Families Reflected In Repo Artifacts

The repo currently contains artifacts for variants such as:

- majority baseline
- `ollama-qwen3-zero-shot`
- `ollama-qwen3-few-shot`
- `ollama-qwen3-retrieval-few-shot`
- `ollama-gemma3-zero-shot`
- `ollama-gemma3-few-shot`
- `openai-gpt-5-nano-zero-shot`
- OpenAI few-shot artifacts such as `gpt5nano`
- `openai-gpt-5-nano-retrieval-few-shot`
- `openai-gpt-5-nano-retrieval-few-shot-sentence-transformer-all-minilm-l6-v2`

There are also config stubs for optional future variants like:

- `gpt-5.4-mini`
- `gpt-5.4`

This is another useful presentation point: the framework is extensible. Adding a new model usually means adding a variant config, not rewriting the pipeline.

## Important Caveats To State Clearly

### Retrieval is transductive

The retrieval pool comes from the same merged workbook used for evaluation. This is helpful for annotation assistance experiments, but it is not a pure held-out retrieval benchmark.

### The pipeline measures structured annotation performance, not general reasoning

Strict JSON schema outputs and normalization are a feature, not a limitation. The goal is controlled annotation performance.

### Construct difficulty is uneven

Some constructs are inherently harder than others, especially emotion coding in `c3`. Presenting metric breakdowns by construct is more honest than presenting one aggregate score.

## Relevant Source Files

- `inference_eval/run_variants.py`
- `inference_eval/prepare_openai_batch.py`
- `inference_eval/poll_openai_batch.py`
- `inference_eval/import_openai_batch.py`
- `inference_eval/prompt_templates.py`
- `inference_eval/retrieval.py`
- `inference_eval/common.py`
- `inference_eval/evaluate_variants.py`
- `inference_eval/results/evaluation/metrics_summary.md`
- `inference_eval/analysis/metrics_analysis.py`

## One-Sentence Thesis

The project implements a controlled multi-call annotation framework where the main experimental variables are prompt strategy, retrieval augmentation, model/provider choice, and execution mode, all evaluated under a shared schema-normalized scoring pipeline.
