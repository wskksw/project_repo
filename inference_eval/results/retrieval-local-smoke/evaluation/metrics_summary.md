# Variant Evaluation Summary

The first call bundles Construct 0 and Construct 1 so the run still uses four model calls per narrative.

`NA` means the evaluator found no scorable ground-truth/prediction pairs for that metric.

| Variant | Provider | Model | Prompt Strategy | C0 Acc | C1 Acc | C2 Abs Acc | C2 Rel Acc | C3 Code Acc | C3 Emotion F1 | C4 Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | baseline | deterministic | majority_class | NA | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| ollama-qwen3-retrieval-few-shot | ollama | qwen3:8b | retrieval_few_shot | NA | NA | 1.0 | 0.0 | NA | NA | 1.0 |
