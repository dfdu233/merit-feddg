"""Build an auditable MedCPT knowledge base from real PubMed baseline XML or JSONL.

No target VQA answers are accepted.  Documents are embedded with the frozen
MedCPT Article Encoder and stored in sharded NumPy arrays plus SQLite metadata.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_of(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def pubmed_records(directory):
    files = sorted(Path(directory).glob("*.xml.gz"))
    if not files:
        raise FileNotFoundError(f"no PubMed .xml.gz files found under {directory}")
    for path in files:
        with gzip.open(path, "rb") as handle:
            for _event, element in ET.iterparse(handle, events=("end",)):
                if element.tag.rsplit("}", 1)[-1] != "PubmedArticle":
                    continue
                citation = element.find(".//MedlineCitation")
                article = element.find(".//Article")
                if citation is None or article is None:
                    element.clear()
                    continue
                pmid = text_of(citation.find("PMID"))
                title = text_of(article.find("ArticleTitle"))
                abstracts = [
                    text_of(value)
                    for value in article.findall(".//Abstract/AbstractText")
                    if text_of(value)
                ]
                content = " ".join(abstracts).strip()
                year = text_of(article.find(".//JournalIssue/PubDate/Year"))
                if not year:
                    year = text_of(article.find(".//JournalIssue/PubDate/MedlineDate"))[:4]
                if pmid and content:
                    yield {
                        "id": f"PMID:{pmid}",
                        "title": title,
                        "content": content,
                        "source": "PubMed",
                        "year": year,
                        "provenance": {
                            "pubmed_baseline_file": path.name,
                            "pmid": pmid,
                        },
                    }
                element.clear()


def jsonl_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if any(key in row for key in ("answer", "answers", "label", "reference")):
                raise ValueError("knowledge-base JSONL must not contain benchmark answers")
            doc_id = str(row.get("id", "")).strip()
            content = str(row.get("content", "")).strip()
            if not doc_id or not content:
                raise ValueError(f"invalid JSONL document at line {line_number}")
            yield {
                "id": doc_id,
                "title": str(row.get("title", "")).strip(),
                "content": content,
                "source": str(row.get("source", "custom")).strip() or "custom",
                "year": str(row.get("year", "")).strip(),
                "provenance": {"jsonl": str(Path(path).resolve()), "line": line_number},
            }


class ArticleEncoder:
    def __init__(self, path, device="auto", batch_size=32):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Build the knowledge base in the existing torch/transformers environment") from exc
        self.torch = torch
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"MedCPT Article Encoder missing: {self.path}. "
                "Run python scripts/prepare_medcpt_assets.py --print-commands."
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True)
        self.model = self.model.to(self.device).eval().requires_grad_(False)

    def encode(self, rows):
        pairs = [[row["title"], row["content"]] for row in rows]
        encoded = self.tokenizer(
            pairs,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            vector = self.model(**encoded).last_hidden_state[:, 0, :]
        result = vector.detach().float().cpu().numpy().astype("<f4", copy=False)
        if result.shape != (len(rows), 768) or not np.isfinite(result).all():
            raise ValueError("MedCPT Article Encoder returned invalid embeddings")
        return result


def init_db(path):
    db = sqlite3.connect(path)
    db.execute(
        """CREATE TABLE documents (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            year TEXT NOT NULL,
            provenance_json TEXT NOT NULL
        )"""
    )
    db.execute("CREATE INDEX documents_doc_id ON documents(doc_id)")
    return db


def flush_shard(root, shard_index, embeddings, rowids):
    if not embeddings:
        return None
    matrix = np.concatenate(embeddings, axis=0).astype("<f4", copy=False)
    ids = np.asarray(rowids, dtype="<i8")
    emb_rel = Path("embeddings") / f"shard-{shard_index:05d}.npy"
    ids_rel = Path("rowids") / f"shard-{shard_index:05d}.npy"
    (root / emb_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / ids_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(root / emb_rel, matrix, allow_pickle=False)
    np.save(root / ids_rel, ids, allow_pickle=False)
    return {
        "embedding": emb_rel.as_posix(),
        "rowids": ids_rel.as_posix(),
        "count": int(matrix.shape[0]),
        "embedding_sha256": sha256(root / emb_rel),
        "rowids_sha256": sha256(root / ids_rel),
    }


def build(records, output, article_encoder, *, limit=0, batch_size=32, shard_size=50000):
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty knowledge-base directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "documents.sqlite3"
    db = init_db(db_path)
    encoder = ArticleEncoder(article_encoder, batch_size=batch_size)

    shards = []
    embed_parts, shard_rowids, pending = [], [], []
    total = duplicate = 0
    shard_index = 0

    def encode_pending():
        nonlocal pending, embed_parts, shard_rowids, total, duplicate, shard_index
        if not pending:
            return
        accepted = []
        for row in pending:
            cursor = db.execute(
                "INSERT OR IGNORE INTO documents(doc_id,title,content,source,year,provenance_json) "
                "VALUES (?,?,?,?,?,?)",
                (
                    row["id"], row["title"], row["content"], row["source"], row["year"],
                    json.dumps(row["provenance"], ensure_ascii=False, sort_keys=True),
                ),
            )
            if cursor.rowcount:
                accepted.append((row, int(cursor.lastrowid)))
            else:
                duplicate += 1
        if accepted:
            rows = [row for row, _ in accepted]
            vectors = encoder.encode(rows)
            embed_parts.append(vectors)
            shard_rowids.extend(rowid for _, rowid in accepted)
            total += len(accepted)
        pending = []
        if len(shard_rowids) >= shard_size:
            db.commit()
            shard = flush_shard(root, shard_index, embed_parts, shard_rowids)
            if shard:
                shards.append(shard)
                shard_index += 1
            embed_parts, shard_rowids = [], []

    for row in records:
        pending.append(row)
        if len(pending) >= batch_size:
            encode_pending()
        if limit and total >= limit:
            break
    encode_pending()
    if shard_rowids:
        db.commit()
        shard = flush_shard(root, shard_index, embed_parts, shard_rowids)
        if shard:
            shards.append(shard)
    db.commit()
    db.close()
    if not total or not shards:
        raise ValueError("knowledge-base build produced no documents")

    payload = {
        "schema": "merit-medcpt-kb-v1",
        "source": "biomedical_literature",
        "documents": total,
        "duplicates_skipped": duplicate,
        "embedding_dim": 768,
        "article_encoder": str(Path(article_encoder).expanduser().resolve()),
        "sqlite": db_path.name,
        "sqlite_sha256": sha256(db_path),
        "shards": shards,
        "target_answers_used": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    payload["fingerprint"] = f"sha256:{fingerprint}"
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pubmed-dir")
    source.add_argument("--jsonl")
    parser.add_argument("--article-encoder", required=True)
    parser.add_argument("--output", default="artifacts/knowledge/medcpt-pubmed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()
    if args.limit < 0 or min(args.batch_size, args.shard_size) < 1:
        parser.error("limit must be nonnegative and batch/shard sizes positive")
    records = (
        pubmed_records(args.pubmed_dir)
        if args.pubmed_dir
        else jsonl_records(args.jsonl)
    )
    result = build(
        records,
        args.output,
        args.article_encoder,
        limit=args.limit,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
