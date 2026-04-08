| provider | family | variant_count | prompt_strategies | C0 Acc | C1 Acc | C2 Abs Acc | C2 Rel Acc | C3 Code Acc | C3 Emotion F1 | C4 Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | 1 | majority_class | 0.3385 | 0.65 | 0.71 | 0.55 | 0.52 | 0.4216 | 0.57 |
| ollama | gemma3 | 4 | zero_shot, few_shot, retrieval_few_shot | 0.2408 | 0.0684 | 0.7469 | 0.2559 | 0.34 | 0.3679 | 0.5779 |
| ollama | qwen3 | 4 | zero_shot, few_shot, retrieval_few_shot | 0.3824 | 0.3587 | 0.8281 | 0.179 | 0.4325 | 0.3757 | 0.6555 |
| openai | gpt-5-nano | 4 | zero_shot, few_shot, retrieval_few_shot | 0.4308 | 0.435 | 0.8275 | 0.3125 | 0.3755 | 0.2993 | 0.7003 |
| openai | gpt-5.4-mini | 2 | few_shot, retrieval_few_shot | 0.4231 | 0.63 | 0.87 | 0.48 | 0.45 | 0.4849 | 0.745 |
