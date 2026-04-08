from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from inference_eval.common import (
    C0_LABELS,
    C1_LABELS,
    C2_REL_LABELS,
    C4_LABELS,
    DEFAULT_DATASET,
    build_ground_truth,
    label_for_code,
    load_dataset,
    normalize_amount,
    parse_pipe_separated_evidence,
)

DEFAULT_RETRIEVAL_K = 2
DEFAULT_RETRIEVAL_CANDIDATE_K = 3
DEFAULT_EXCERPT_CHARS = 900
RETRIEVAL_POLICY = "leave_one_out"
DEFAULT_RETRIEVAL_BACKEND = "tfidf"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_BACKENDS = ("tfidf", "sentence_transformer")
REFERENCE_RATIONALE = "Reference annotation from a similar complaint narrative."


@dataclass(frozen=True)
class RetrievedExample:
    complaint_id: str
    row_index: int
    similarity_rank: int
    similarity_score: float
    narrative_excerpt: str
    gold_json: dict[str, Any]


class RetrievalIndex:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        backend: str = DEFAULT_RETRIEVAL_BACKEND,
        retriever: TFIDFRetriever | None = None,
        sentence_model: Any = None,
        vector_matrix: Any = None,
        embedding_model: str | None = None,
        excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    ) -> None:
        self.rows = rows
        self.backend = backend
        self.retriever = retriever
        self.sentence_model = sentence_model
        self.vector_matrix = vector_matrix
        self.embedding_model = embedding_model
        self.excerpt_chars = excerpt_chars

    @classmethod
    def from_dataset(
        cls,
        path: str | Path = DEFAULT_DATASET,
        *,
        backend: str = DEFAULT_RETRIEVAL_BACKEND,
        embedding_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
        candidate_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K,
    ) -> RetrievalIndex:
        if backend not in RETRIEVAL_BACKENDS:
            raise ValueError(f"Unsupported retrieval backend: {backend}")
        raw_df = load_dataset(path)
        gt_df = build_ground_truth(raw_df)
        rows: list[dict[str, Any]] = []
        documents: list[Document] = []

        for row_idx, raw_row in raw_df.iterrows():
            narrative = str(raw_row.get("Consumer complaint narrative") or "").strip()
            if not narrative:
                continue
            gt_row = gt_df.loc[row_idx]
            complaint_id = str(raw_row["Complaint ID"])
            payloads = {
                "c1": _build_c1_payload(raw_row, gt_row),
                "c2": _build_c2_payload(raw_row, gt_row),
                "c3": _build_c3_payload(raw_row, gt_row),
                "c4": _build_c4_payload(raw_row, gt_row),
            }
            row = {
                "complaint_id": complaint_id,
                "row_index": int(row_idx),
                "narrative": narrative,
                "construct_payloads": payloads,
            }
            rows.append(row)
            documents.append(
                Document(
                    page_content=narrative,
                    metadata={"complaint_id": complaint_id, "row_index": int(row_idx)},
                )
            )

        if backend == "tfidf":
            retriever = TFIDFRetriever.from_documents(documents, k=max(candidate_k, DEFAULT_RETRIEVAL_CANDIDATE_K))
            return cls(
                rows=rows,
                backend=backend,
                retriever=retriever,
                embedding_model=None,
                excerpt_chars=excerpt_chars,
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "sentence-transformers is required for retrieval_backend='sentence_transformer'. "
                "Install inference_eval/requirements.txt again after adding that dependency."
            ) from exc

        sentence_model = SentenceTransformer(embedding_model)
        vector_matrix = sentence_model.encode(
            [row["narrative"] for row in rows],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cls(
            rows=rows,
            backend=backend,
            sentence_model=sentence_model,
            vector_matrix=vector_matrix,
            embedding_model=embedding_model,
            excerpt_chars=excerpt_chars,
        )

    def retrieve(
        self,
        *,
        complaint_id: str,
        construct: str,
        narrative: str,
        k: int = DEFAULT_RETRIEVAL_K,
        candidate_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K,
    ) -> list[RetrievedExample]:
        if not narrative.strip():
            return []

        if self.backend == "tfidf":
            if self.retriever is None:
                raise RuntimeError("TF-IDF retrieval backend selected without an initialized retriever.")
            query_vector = self.retriever.vectorizer.transform([narrative])
            similarity_scores = cosine_similarity(query_vector, self.retriever.tfidf_array).ravel()
        elif self.backend == "sentence_transformer":
            if self.sentence_model is None or self.vector_matrix is None:
                raise RuntimeError("Sentence-transformer retrieval backend selected without initialized embeddings.")
            query_vector = self.sentence_model.encode(
                [narrative],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            similarity_scores = cosine_similarity(query_vector, self.vector_matrix).ravel()
        else:
            raise RuntimeError(f"Unknown retrieval backend: {self.backend}")
        ranked_indexes = sorted(
            range(len(self.rows)),
            key=lambda index: (-float(similarity_scores[index]), self.rows[index]["row_index"]),
        )

        candidate_indexes = ranked_indexes[: max(candidate_k, k + 1)]
        selected = self._select_examples(
            candidate_indexes,
            similarity_scores,
            complaint_id=complaint_id,
            construct=construct,
            k=k,
        )
        if len(selected) < k:
            selected = self._select_examples(
                ranked_indexes,
                similarity_scores,
                complaint_id=complaint_id,
                construct=construct,
                k=k,
            )
        return selected

    def _select_examples(
        self,
        ranked_indexes: list[int],
        similarity_scores: Any,
        *,
        complaint_id: str,
        construct: str,
        k: int,
    ) -> list[RetrievedExample]:
        examples: list[RetrievedExample] = []
        seen_ids: set[str] = set()

        for index in ranked_indexes:
            row = self.rows[index]
            candidate_id = row["complaint_id"]
            if candidate_id == complaint_id or candidate_id in seen_ids:
                continue
            gold_json = row["construct_payloads"].get(construct)
            if not gold_json:
                continue
            seen_ids.add(candidate_id)
            examples.append(
                RetrievedExample(
                    complaint_id=candidate_id,
                    row_index=row["row_index"],
                    similarity_rank=len(examples) + 1,
                    similarity_score=_clean_score(similarity_scores[index]),
                    narrative_excerpt=truncate_text(row["narrative"], self.excerpt_chars),
                    gold_json=gold_json,
                )
            )
            if len(examples) >= k:
                break
        return examples


def retrieval_metadata(
    examples: list[RetrievedExample],
    *,
    k: int,
    backend: str = DEFAULT_RETRIEVAL_BACKEND,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "retrieval_policy": RETRIEVAL_POLICY,
        "retrieval_k": int(k),
        "retrieval_backend": backend,
        "retrieval_embedding_model": embedding_model,
        "retrieved_complaint_ids": [item.complaint_id for item in examples],
        "retrieval_scores": [item.similarity_score for item in examples],
        "retrieved_row_indexes": [item.row_index for item in examples],
    }
    return metadata


def _build_c1_payload(raw_row: Any, gt_row: Any) -> dict[str, Any] | None:
    c0_code = gt_row.get("gt_c0_label")
    c1_code = gt_row.get("gt_c1_score")
    if c0_code is None or c1_code is None or _is_nan(c0_code) or _is_nan(c1_code):
        return None
    c0_code = int(c0_code)
    c1_code = int(c1_code)
    return {
        "user_info_leaked_code": c0_code,
        "user_info_leaked_label": label_for_code(C0_LABELS, c0_code),
        "user_info_evidence_quotes": parse_pipe_separated_evidence(raw_row.get("c0_evidence quotes")),
        "perceived_info_vulnerability_score": c1_code,
        "perceived_info_vulnerability_label": label_for_code(C1_LABELS, c1_code),
        "perceived_vulnerability_evidence_quotes": parse_pipe_separated_evidence(raw_row.get("c1_evidence_quotes")),
        "rationale": REFERENCE_RATIONALE,
    }


def _build_c2_payload(raw_row: Any, gt_row: Any) -> dict[str, Any] | None:
    abs_code = gt_row.get("gt_c2_abs_code")
    rel_score = gt_row.get("gt_c2_rel_score")
    if abs_code is None or rel_score is None or _is_nan(abs_code) or _is_nan(rel_score):
        return None
    abs_code = int(abs_code)
    rel_score = int(rel_score)
    amount = normalize_amount(gt_row.get("gt_c2_abs_amount"))
    if abs_code == 0:
        amount = None
    return {
        "economic_absolute_code": abs_code,
        "economic_absolute_amount_usd": amount,
        "economic_relative_significance_score": rel_score,
        "economic_relative_label": label_for_code(C2_REL_LABELS, rel_score),
        "economic_evidence_quotes": parse_pipe_separated_evidence(raw_row.get("c2_evidence_quotes")),
        "rationale": REFERENCE_RATIONALE,
    }


def _build_c3_payload(raw_row: Any, gt_row: Any) -> dict[str, Any] | None:
    code = gt_row.get("gt_c3_code")
    tier = gt_row.get("gt_c3_tier")
    if code is None or tier is None or _is_nan(code) or _is_nan(tier):
        return None
    return {
        "emotion_count_code": int(code),
        "emotions": list(gt_row.get("gt_c3_emotions") or []),
        "evidence_tier": int(tier),
        "inference_patterns": list(gt_row.get("gt_c3_inference_patterns") or []),
        "evidence_quotes": parse_pipe_separated_evidence(raw_row.get("c3_evidence_quotes")),
        "rationale": REFERENCE_RATIONALE,
    }


def _build_c4_payload(raw_row: Any, gt_row: Any) -> dict[str, Any] | None:
    score = gt_row.get("gt_c4_score")
    if score is None or _is_nan(score):
        return None
    score = int(score)
    return {
        "investment_engagement_score": score,
        "investment_engagement_label": label_for_code(C4_LABELS, score),
        "evidence_quotes": parse_pipe_separated_evidence(raw_row.get("c4_evidence_quotes")),
        "rationale": REFERENCE_RATIONALE,
    }


def truncate_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _clean_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(score):
        return 0.0
    return round(score, 6)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)
