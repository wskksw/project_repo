# Metrics Analysis Workspace

This folder contains a notebook-driven analysis workspace for the evaluation outputs produced by [`evaluate_variants.py`](/Users/kevin/school/532/project_repo/inference_eval/evaluate_variants.py).

## Contents

- [`metrics_analysis.ipynb`](/Users/kevin/school/532/project_repo/inference_eval/analysis/metrics_analysis.ipynb): notebook for interactive exploration
- [`presentation_tables.ipynb`](/Users/kevin/school/532/project_repo/inference_eval/analysis/presentation_tables.ipynb): self-contained notebook for slide-ready tables and per-metric winners
- [`metrics_analysis.py`](/Users/kevin/school/532/project_repo/inference_eval/analysis/metrics_analysis.py): reusable analysis helper and output generator
- [`outputs/`](/Users/kevin/school/532/project_repo/inference_eval/analysis/outputs): generated tables in CSV and Markdown

## Default Input

By default the analysis targets:

- [`results/evaluation/metrics_by_variant.csv`](/Users/kevin/school/532/project_repo/inference_eval/results/evaluation/metrics_by_variant.csv)

You can point it at another evaluated run by changing `RUN_DIR` in the notebook or by passing `--run-dir` to the script.

## Regenerate Outputs

```bash
python -m inference_eval.analysis.metrics_analysis \
  --run-dir inference_eval/results \
  --output-dir inference_eval/analysis/outputs
```

## Notebook Import Bootstrap

If the notebook kernel starts inside `inference_eval/analysis`, inserting that directory or the `inference_eval/` package directory into `sys.path` is not enough for:

```python
from inference_eval.analysis.metrics_analysis import build_analysis_outputs
```

The import needs the repo root on `sys.path`. Use this bootstrap block in the notebook:

```python
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

CWD = Path.cwd().resolve()
REPO_ROOT = next((path for path in [CWD, *CWD.parents] if (path / "inference_eval").exists()), None)
if REPO_ROOT is None:
    raise RuntimeError(f"Could not locate repo root from {CWD}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_eval.analysis.metrics_analysis import build_analysis_outputs
```

The older pattern of setting `REPO_ROOT = Path.cwd()` and then falling back to `REPO_ROOT.parent` can fail when the current working directory is already `inference_eval/analysis`, because it inserts `.../inference_eval` instead of the actual repo root.

## Generated Tables

- `summary_table`: key metrics in the same row-oriented format as `metrics_summary.md`
- `prompt_strategy_comparison`: mean key metrics grouped by prompt strategy
- `family_comparison`: mean key metrics grouped by model family
- `metric_view`: metric-centric pivot with metrics as rows and variants as columns
- `primary_metric_leaderboard`: rankings for the primary metrics used in the summary
- `primary_metric_coverage`: scorable pair counts `n` for the primary metrics

These tables are intentionally flat so they can be reused in reports, copied into docs, or reloaded from the notebook.

## Presentation Notebook

Use [`presentation_tables.ipynb`](/Users/kevin/school/532/project_repo/inference_eval/analysis/presentation_tables.ipynb) when you want cleaner labels and slide-oriented rollups:

- a main results table with model and prompt-strategy labels instead of variant slugs
- a prompt-strategy comparison averaged over models rather than raw variants
- a family comparison that collapses duplicate retrieval backends before averaging
- an optional C3 deep-dive table for emotion precision, recall, F1, and tier accuracy
- a plain logged winner table under each view so the best result for each metric is explicit
