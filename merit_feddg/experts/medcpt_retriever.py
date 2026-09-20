"""Local MedCPT retrieval expert over an auditable biomedical knowledge base.

The retriever supplies literature context, not a patient-specific diagnosis.
Dense retrieval scores and cross-encoder scores are relevance signals only.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..capabilities import CapabilityResult, EvidenceItem


class MedCPTRetrievalExpert:
    def __init__(
        self,
        model_id: str,
        expert_id: str,
        kb_manifest: str,
        scope: str = "biomedical_literature",
        cross_encoder_path: str | None = None,
        device: str = "auto",
        candidate_k: int = 32,
        top_k: int = 3,
        max_query_tokens: int = 64,
        max_pair_tokens: int = 512,
        source_balance: bool = True,
    ):
        if min(candidate_k, top_k, max_query_tokens, max_pair_tokens) < 1:
            raise ValueError("MedCPT retrieval budgets must be positive")
        if top_k > candidate_k:
            raise ValueError("top_k cannot exceed candidate_k")
        self.query_path = Path(model_id).expanduser().resolve()
        self.kb_manifest_path = Path(kb_manifest).expanduser().resolve()
        self.cross_path = (
            Path(cross_encoder_path).expanduser().resolve()
            if cross_encoder_path
            else None
        )
        if not self.query_path.is_dir():
            raise FileNotFoundError(
                "MedCPT Query Encoder is missing. Run "
                "python scripts/prepare_medcpt_assets.py --print-commands."
            )
        if not self.kb_manifest_path.is_file():
            raise FileNotFoundError(
                "MedCPT knowledge base manifest is missing. Build it with "
                "python scripts/build_medcpt_kb.py --help after downloading PubMed."
            )
        self.expert_id, self.scope = expert_id, scope
        self.device_request = device
        self.candidate_k, self.top_k = candidate_k, top_k
        self.max_query_tokens, self.max_pair_tokens = max_query_tokens, max_pair_tokens
        if type(source_balance) is not bool:
            raise TypeError("source_balance must be boolean")
        self.source_balance = source_balance
        self._query_model = self._query_tokenizer = None
        self._reranker = self._reranker_tokenizer = None
        self._manifest = self._read_manifest()
        self._db = None
        self._faiss_index = None
        self._faiss_rowids = None
        self._faiss_source_indices = {}

    def _read_manifest(self):
        payload = json.loads(self.kb_manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "merit-medcpt-kb-v1":
            raise ValueError("unsupported MedCPT knowledge-base manifest")
        if payload.get("embedding_dim") != 768:
            raise ValueError("MedCPT knowledge base must use 768-D article embeddings")
        if not isinstance(payload.get("shards"), list) or not payload["shards"]:
            raise ValueError("MedCPT knowledge base has no embedding shards")
        root = self.kb_manifest_path.parent
        database = root / payload.get("sqlite", "")
        if not database.is_file():
            raise FileNotFoundError(f"knowledge-base SQLite store is missing: {database}")
        for shard in payload["shards"]:
            for key in ("embedding", "rowids", "count"):
                if key not in shard:
                    raise ValueError(f"knowledge-base shard missing {key}")
            embedding = root / shard["embedding"]
            rowids = root / shard["rowids"]
            if not embedding.is_file() or not rowids.is_file():
                raise FileNotFoundError("knowledge-base embedding shard is incomplete")
            if shard.get("source_ids"):
                source_ids = root / shard["source_ids"]
                if not source_ids.is_file():
                    raise FileNotFoundError("knowledge-base source-id shard is incomplete")
        return payload

    @property
    def kb_root(self):
        return self.kb_manifest_path.parent

    def _load_models(self):
        if self._query_model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT expert requires the existing research environment with torch/transformers"
            ) from exc
        if self.device_request == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device_request
        self.torch, self.device = torch, torch.device(device)
        self._query_tokenizer = AutoTokenizer.from_pretrained(
            self.query_path, local_files_only=True
        )
        self._query_model = AutoModel.from_pretrained(
            self.query_path, local_files_only=True
        ).to(self.device).eval().requires_grad_(False)
        if self.cross_path is not None and self.cross_path.is_dir():
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.cross_path, local_files_only=True
            )
            self._reranker = AutoModelForSequenceClassification.from_pretrained(
                self.cross_path, local_files_only=True
            ).to(self.device).eval().requires_grad_(False)

    def _query_embedding(self, text):
        self._load_models()
        encoded = self._query_tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=self.max_query_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            vector = self._query_model(**encoded).last_hidden_state[:, 0, :]
        result = vector[0].detach().float().cpu().numpy()
        if result.shape != (768,) or not np.isfinite(result).all():
            raise ValueError("MedCPT query encoder returned an invalid vector")
        return result

    @staticmethod
    def _topk(scores, k):
        values = np.asarray(scores, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("knowledge-base similarity scores must be finite")
        k = min(k, values.size)
        if not k:
            return np.empty(0, dtype=np.int64)
        indices = np.argpartition(values, values.size - k)[-k:]
        return indices[np.argsort(values[indices])[::-1]]

    def _faiss_candidates(self, query):
        spec = self._manifest.get("faiss")
        if not spec:
            return None
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "This knowledge base was built with FAISS. Install faiss-cpu/faiss-gpu "
                "in the retrieval environment, or rebuild a small exact-scan pilot."
            ) from exc

        if self.source_balance and spec.get("source_indices"):
            result = []
            for source in spec["source_indices"]:
                source_id = int(source["source_id"])
                if source_id not in self._faiss_source_indices:
                    index_path = self.kb_root / source["index"]
                    rowids_path = self.kb_root / source["rowids"]
                    if not index_path.is_file() or not rowids_path.is_file():
                        raise FileNotFoundError(
                            "source-specific FAISS knowledge index is incomplete"
                        )
                    index = faiss.read_index(str(index_path))
                    rowids = np.load(rowids_path, mmap_mode="r")
                    if index.ntotal != len(rowids):
                        raise ValueError(
                            "source FAISS index and rowid mapping have different sizes"
                        )
                    if source.get("backend", spec.get("backend")) == "hnsw":
                        index.hnsw.efSearch = max(64, self.candidate_k * 2)
                    self._faiss_source_indices[source_id] = (index, rowids)
                index, rowids = self._faiss_source_indices[source_id]
                scores, indices = index.search(
                    np.asarray(query, dtype=np.float32)[None, :],
                    min(self.candidate_k, index.ntotal),
                )
                for score, index_value in zip(scores[0], indices[0], strict=True):
                    if index_value < 0:
                        continue
                    result.append(
                        (float(score), int(rowids[index_value]), source_id, int(index_value))
                    )
            result.sort(key=lambda value: (-value[0], value[1]))
            return result

        if self._faiss_index is None:
            index_path = self.kb_root / spec["index"]
            rowids_path = self.kb_root / spec["rowids"]
            if not index_path.is_file() or not rowids_path.is_file():
                raise FileNotFoundError("FAISS knowledge-base index is incomplete")
            self._faiss_index = faiss.read_index(str(index_path))
            self._faiss_rowids = np.load(rowids_path, mmap_mode="r")
            if self._faiss_index.ntotal != len(self._faiss_rowids):
                raise ValueError("FAISS index and rowid mapping have different sizes")
            if spec.get("backend") == "hnsw":
                self._faiss_index.hnsw.efSearch = max(64, self.candidate_k * 2)
        scores, indices = self._faiss_index.search(
            np.asarray(query, dtype=np.float32)[None, :],
            min(self.candidate_k, self._faiss_index.ntotal),
        )
        result = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            result.append((float(score), int(self._faiss_rowids[index]), -1, int(index)))
        return result

    def _dense_candidates(self, query):
        indexed = self._faiss_candidates(query)
        if indexed is not None:
            return indexed

        source_vocab = self._manifest.get("source_vocab", [])
        source_aware = self.source_balance and len(source_vocab) > 1
        candidates = []
        per_source = {source_id: [] for source_id in range(len(source_vocab))}
        for shard_id, shard in enumerate(self._manifest["shards"]):
            embeddings = np.load(
                self.kb_root / shard["embedding"], mmap_mode="r"
            )
            rowids = np.load(self.kb_root / shard["rowids"], mmap_mode="r")
            if embeddings.shape != (int(shard["count"]), 768):
                raise ValueError("knowledge-base embedding shape does not match manifest")
            if rowids.shape != (int(shard["count"]),):
                raise ValueError("knowledge-base rowid shape does not match manifest")
            scores = np.asarray(embeddings @ query, dtype=np.float32)

            if source_aware:
                source_path = shard.get("source_ids")
                if not source_path:
                    source_aware = False
                else:
                    source_ids = np.load(self.kb_root / source_path, mmap_mode="r")
                    if source_ids.shape != rowids.shape:
                        raise ValueError("knowledge-base source-id shape does not match rowids")
                    for source_id in per_source:
                        positions = np.flatnonzero(np.asarray(source_ids) == source_id)
                        if not positions.size:
                            continue
                        local = self._topk(scores[positions], self.candidate_k)
                        for local_index in local:
                            index = int(positions[local_index])
                            per_source[source_id].append(
                                (
                                    float(scores[index]),
                                    int(rowids[index]),
                                    shard_id,
                                    index,
                                )
                            )
                    continue

            for index in self._topk(scores, self.candidate_k):
                candidates.append(
                    (float(scores[index]), int(rowids[index]), shard_id, int(index))
                )

        if source_aware:
            candidates = []
            for source_id, values in per_source.items():
                values.sort(key=lambda value: (-value[0], value[1]))
                candidates.extend(values[: self.candidate_k])
        candidates.sort(key=lambda value: (-value[0], value[1]))
        return candidates if source_aware else candidates[: self.candidate_k]

    def _connection(self):
        if self._db is None:
            path = self.kb_root / self._manifest["sqlite"]
            self._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _documents(self, candidates):
        rowids = [rowid for _, rowid, _, _ in candidates]
        placeholders = ",".join("?" for _ in rowids)
        rows = self._connection().execute(
            f"SELECT row_id, doc_id, title, content, source, year FROM documents "
            f"WHERE row_id IN ({placeholders})",
            rowids,
        ).fetchall()
        by_id = {int(row["row_id"]): dict(row) for row in rows}
        if len(by_id) != len(set(rowids)):
            raise ValueError("knowledge-base metadata is missing retrieved documents")
        return [by_id[rowid] for _, rowid, _, _ in candidates]

    def _rerank(self, query, candidates, documents):
        if self._reranker is None:
            return [
                {**doc, "dense_score": float(candidate[0]), "rerank_score": None}
                for candidate, doc in zip(candidates, documents, strict=True)
            ]
        pairs = [
            [query, (doc["title"] + ". " + doc["content"]).strip()]
            for doc in documents
        ]
        encoded = self._reranker_tokenizer(
            pairs,
            truncation=True,
            padding=True,
            max_length=self.max_pair_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self._reranker(**encoded).logits.reshape(-1)
        scores = logits.detach().float().cpu().numpy()
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise ValueError("MedCPT cross encoder returned invalid scores")
        records = [
            {
                **doc,
                "dense_score": float(candidate[0]),
                "rerank_score": float(score),
            }
            for candidate, doc, score in zip(candidates, documents, scores, strict=True)
        ]
        records.sort(key=lambda value: (-value["rerank_score"], -value["dense_score"]))
        return records

    def _select_topk(self, ranked):
        """Prefer source diversity before filling remaining relevance slots.

        This is deterministic diversity, not a learned source prior. With a
        single-source KB it is identical to ordinary top-k.
        """
        if not self.source_balance or self.top_k <= 1:
            return ranked[: self.top_k]
        selected, deferred, seen = [], [], set()
        for row in ranked:
            source = str(row.get("source", ""))
            if source and source not in seen and len(selected) < self.top_k:
                selected.append(row)
                seen.add(source)
            else:
                deferred.append(row)
        if len(selected) < self.top_k:
            selected.extend(deferred[: self.top_k - len(selected)])
        return selected[: self.top_k]


    def infer(self, request):
        if request.capability != "retrieval" or request.scope != self.scope:
            return CapabilityResult(
                self.expert_id, request.capability, (), "wrong_capability_or_scope"
            )
        if request.region is not None:
            return CapabilityResult(
                self.expert_id, request.capability, (), "unsupported_region"
            )
        query = (request.query or request.question).strip()
        if not query:
            return CapabilityResult(
                self.expert_id, request.capability, (), "empty_query"
            )
        vector = self._query_embedding(query)
        candidates = self._dense_candidates(vector)
        if not candidates:
            return CapabilityResult(
                self.expert_id, request.capability, (), "empty_knowledge_base"
            )
        documents = self._documents(candidates)
        ranked = self._select_topk(self._rerank(query, candidates, documents))
        items = []
        for rank, row in enumerate(ranked, 1):
            reference = {
                "document_id": row["doc_id"],
                "source": row["source"],
                "year": row["year"],
                "title": row["title"],
                "content": row["content"],
                "dense_score": row["dense_score"],
                "rerank_score": row["rerank_score"],
                "score_semantics": (
                    "MedCPT_cross_encoder_relevance"
                    if row["rerank_score"] is not None
                    else "MedCPT_inner_product_relevance"
                ),
                "patient_specific_claim": "not_established",
                "retrieval_rank": rank,
            }
            safe_id = str(row["doc_id"]).replace(":", "-").replace("/", "-")
            items.append(
                EvidenceItem(
                    evidence_id=(
                        f"{self.expert_id}:{request.sample_id}:literature:"
                        f"{rank}:{safe_id}"
                    ),
                    expert_id=self.expert_id,
                    capability="retrieval",
                    scope=self.scope,
                    payload={
                        "references": [reference],
                        "retrieval_is_patient_diagnosis": False,
                        "source_answers_included": False,
                    },
                    summary=(
                        "Retrieved biomedical literature passage; relevance is "
                        "not patient-specific clinical truth."
                    ),
                    confidence=None,
                    provenance={
                        "adapter": "medcpt_pubmed",
                        "kb_schema": self._manifest["schema"],
                        "kb_fingerprint": self._manifest.get("fingerprint"),
                        "documents_indexed": self._manifest.get("documents"),
                        "index_backend": (
                            self._manifest.get("faiss", {}).get("backend")
                            if self._manifest.get("faiss") else "exact_sharded_scan"
                        ),
                        "query_encoder": str(self.query_path),
                        "cross_encoder_used": self._reranker is not None,
                        "source_balance": self.source_balance,
                        "retrieval_query": query,
                        "retrieval_rank": rank,
                        "target_answers_used": False,
                        "generated_prefix_used": False,
                    },
                )
            )
        return CapabilityResult(
            self.expert_id, request.capability, tuple(items)
        )
