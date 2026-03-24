# CFPB Fraud Complaint Annotation Pipeline

Annotates CFPB consumer complaint narratives across 4 constructs using **GPT-5.4** (OpenAI Responses API + Batch API).

| Construct | What it measures |
|-----------|-----------------|
| C1 | Perceived Information Vulnerability (0–4) |
| C2 | Harms Experienced — absolute loss, relative harm, psychological harm |
| C3 | Emotions Reported — 8 emotions × present / tier / inference pattern |
| C4 | Investment & Digital Asset Engagement (0–2) |

---

## Setup

```bash
# Install dependencies
pip install openai pandas

# Set your OpenAI API key (or edit the OPENAI_API_KEY line in annotation_pipeline.py)
export OPENAI_API_KEY="sk-..."
```

---

## Commands

### 1. Test on a single narrative (all 4 constructs)

Run this first to verify your API key and prompt quality before spending batch credits.

```bash
python annotation_pipeline.py --test
```

Output: pretty-printed JSON for each construct on the first narrative in `human_review_sheet_100_samples.csv`.

---

### 2. Submit a batch job (all narratives × all 4 constructs)

```bash
python annotation_pipeline.py --batch
```

This will:
1. Write `annotation_folder/batch_requests.jsonl` (~400 requests for 100 narratives)
2. Upload the file to OpenAI Files API
3. Submit the batch and print a **Batch ID** — save it

Example output:
```
Uploading batch file to OpenAI Files API...
  File ID:  file-abc123
Submitting batch...
  Batch ID: batch_xyz789
  Status:   validating

Save this ID → run with --retrieve batch_xyz789 when complete.
```

---

### 3. Check batch status

```bash
python annotation_pipeline.py --poll batch_xyz789
```

Prints status (`validating` → `in_progress` → `finalizing` → `completed`) and request counts. Polls every 30 seconds until terminal state.

---

### 4. Download results and write annotated CSV

```bash
python annotation_pipeline.py --retrieve batch_xyz789
```

This will:
1. Poll until batch is `completed`
2. Download results to `annotation_folder/batch_results.jsonl`
3. Merge AI annotations into the review sheet
4. Save `annotation_folder/annotated_results.csv`

---

## Human Review Workflow

After `--retrieve` completes:

1. Open `annotation_folder/annotated_results.csv`
2. Each row has `review_status = "pending"`
3. Review the AI-filled columns (`c1_score`, `c2_abs_code`, `c3_emotions_json`, etc.)
4. If you disagree, fill in the corresponding `*_human_override` column
5. Change `review_status` to `"reviewed"` and add notes in `review_notes`

The `*_human_override` columns are the ground truth — the AI annotation is just the first pass.

---

## File Layout

```
annotation_folder/
  human_review_sheet_100_samples.csv   ← input: 100 narratives
  annotated_results.csv                ← output: AI annotations (created by --retrieve)
  batch_requests.jsonl                 ← intermediate: batch input (created by --batch)
  batch_results.jsonl                  ← intermediate: raw API output (created by --retrieve)
  prompts/
    construct_1_perceived_info_vulnerability.md
    construct_2_harms_experienced.md
    construct_3_emotions_reported.md
    construct_4_investment_engagement.md
annotation_pipeline.py                 ← main script
```

---

## Configuration

At the top of `annotation_pipeline.py`:

```python
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
MODEL            = "gpt-5.4"
REASONING_EFFORT = "medium"   # none | low | medium | high | xhigh
MAX_TOKENS       = 25000      # reasoning + output tokens combined
```
