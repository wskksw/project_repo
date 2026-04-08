# Prompt Strategy Comparison

Use this on the experiment-design slide to show exactly what each prompt condition adds.

```mermaid
flowchart LR
    A["Same target narrative<br/>same construct schema"] --> B["Zero-shot"]
    A --> C["Few-shot"]
    A --> D["Retrieval few-shot"]

    B --> B1["Short task framing"]
    B --> B2["Allowed label set"]
    B --> B3["Return JSON matching schema"]

    C --> C1["Everything in zero-shot"]
    C --> C2["Construct-specific decision rules"]
    C --> C3["Worked examples from the annotation guide"]
    C --> C4["Disambiguation heuristics for tricky labels"]

    D --> D1["Everything in few-shot"]
    D --> D2["Top-k similar annotated complaints"]
    D --> D3["Gold JSON from retrieved reference cases"]
    D --> D4["Explicit rule: ground final answer only in the target narrative"]
```

What each strategy entails:
- `Zero-shot`: tests the model with the task definition, label vocabulary, and schema only.
- `Few-shot`: adds richer annotation guidance and hand-authored examples inside the prompt.
- `Retrieval few-shot`: keeps the few-shot scaffold, then injects dynamically retrieved complaint examples and their gold annotations.
