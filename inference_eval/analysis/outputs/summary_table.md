| variant_name | provider | model | family | prompt_strategy | C0 Acc | C1 Acc | C2 Abs Acc | C2 Rel Acc | C3 Code Acc | C3 Emotion F1 | C4 Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | baseline | deterministic | baseline | majority_class | 0.3385 | 0.65 | 0.71 | 0.55 | 0.52 | 0.4216 | 0.57 |
| ollama-gemma3-zero-shot | ollama | gemma3:4b | gemma3 | zero_shot | 0.2462 | 0.04 | 0.68 | 0.06 | 0.28 | 0.3732 | 0.54 |
| ollama-qwen3-zero-shot | ollama | qwen3:8b | qwen3 | zero_shot | 0.3385 | 0.13 | 0.7755 | 0.1224 | 0.52 | 0.3578 | 0.7071 |
| openai-gpt-5-nano-zero-shot | openai | gpt-5-nano-2025-08-07 | gpt-5-nano | zero_shot | 0.4615 | 0.47 | 0.75 | 0.25 | 0.3519 | 0.0816 | 0.641 |
| ollama-gemma3-few-shot | ollama | gemma3:4b | gemma3 | few_shot | 0.1077 | 0.03 | 0.7677 | 0.3535 | 0.34 | 0.338 | 0.6162 |
| ollama-qwen3-few-shot | ollama | qwen3:8b | qwen3 | few_shot | 0.3231 | 0.43 | 0.8283 | 0.0808 | 0.4 | 0.3321 | 0.4949 |
| openai-gpt-5-nano-few-shot | openai | gpt-5-nano-2025-08-07 | gpt-5-nano | few_shot | 0.3385 | 0.37 | 0.86 | 0.31 | 0.35 | 0.381 | 0.71 |
| openai-gpt-5-4-mini-few-shot | openai | gpt-5.4-mini-2026-03-17 | gpt-5.4-mini | few_shot | 0.4308 | 0.65 | 0.84 | 0.46 | 0.45 | 0.4861 | 0.74 |
| ollama-gemma3-retrieval-few-shot | ollama | gemma3:4b | gemma3 | retrieval_few_shot | 0.2812 | 0.0722 | 0.81 | 0.29 | 0.33 | 0.3617 | 0.5354 |
| ollama-gemma3-retrieval-few-shot-minilm | ollama | gemma3:4b | gemma3 | retrieval_few_shot | 0.3281 | 0.1313 | 0.73 | 0.32 | 0.41 | 0.3985 | 0.62 |
| ollama-qwen3-retrieval-few-shot | ollama | qwen3:8b | qwen3 | retrieval_few_shot | 0.4462 | 0.41 | 0.8687 | 0.2828 | 0.38 | 0.4071 | 0.68 |
| ollama-qwen3-retrieval-few-shot-minilm | ollama | qwen3:8b | qwen3 | retrieval_few_shot | 0.4219 | 0.4646 | 0.84 | 0.23 | 0.43 | 0.4059 | 0.74 |
| openai-gpt-5-nano-retrieval-few-shot | openai | gpt-5-nano-2025-08-07 | gpt-5-nano | retrieval_few_shot | 0.4462 | 0.41 | 0.84 | 0.32 | 0.44 | 0.386 | 0.73 |
| openai-gpt-5-nano-retrieval-few-shot-sentence-transformer-all-minilm-l6-v2 | openai | gpt-5-nano-2025-08-07 | gpt-5-nano | retrieval_few_shot | 0.4769 | 0.49 | 0.86 | 0.37 | 0.36 | 0.3486 | 0.72 |
| openai-gpt-5-4-mini-retrieval-few-shot-sentence-transformer-all-minilm-l6-v2 | openai | gpt-5.4-mini-2026-03-17 | gpt-5.4-mini | retrieval_few_shot | 0.4154 | 0.61 | 0.9 | 0.5 | 0.45 | 0.4836 | 0.75 |
