# Retrieval-Augmented Prompting

Use this on the retrieval slide to show the flow without dropping into implementation details.

```mermaid
flowchart LR
    A["Target complaint narrative"] --> B["Similarity search over annotated corpus"]
    B --> C["Backend choice<br/>TF-IDF or MiniLM embeddings"]
    C --> D["Leave-one-out filtering"]
    D --> E["Top-k matched complaints<br/>plus gold annotations"]
    E --> F["Inject retrieved cases into the retrieval few-shot prompt"]
    F --> G["Model predicts JSON for the target narrative"]
    G --> H["Normalization and evaluation"]
```

Retrieval-specific points to say aloud:
- Retrieved examples are analogous references, not labels to copy.
- Leave-one-out retrieval prevents the target complaint from being returned as its own example.
- The retrieval backend is part of the experimental condition, so TF-IDF and MiniLM can be compared directly.
