# High-Level Inference Pipeline

Use this on the methodology slide to separate what stays fixed from what changes across conditions.

```mermaid
flowchart LR
    A["CFPB complaint narratives<br/>100 adjudicated examples"] --> B["Controlled experiment setup<br/>same 4-call decomposition<br/>same JSON schemas<br/>same normalization + scoring"]
    B --> C["Experimental condition<br/>prompt strategy + model + retrieval backend"]
    C --> D["Per narrative: 4 construct calls<br/>C0+C1, C2, C3, C4"]
    D --> E["Schema-checked JSON outputs"]
    E --> F["Canonical normalization"]
    F --> G["Evaluation against gold labels"]
    G --> H["Presentation tables and figures"]
```

Key message:
- The dataset, task decomposition, output contracts, and scoring stay constant.
- Only the inference condition changes, so performance differences are attributable to model and prompting choices.
