"""Import official precomputed MedCPT PubMed chunks into the MERIT KB format.

Expected files for each chunk N are the files published by NCBI MedCPT:
  embeds_chunk_N.npy
  pmids_chunk_N.json
  pubmed_chunk_N.json

The original content JSON maps PMID strings to compact records with title/date/
abstract fields (t/d/a). No VQA labels or target answers are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunk(root, index):
    root = Path(root)
    embedding_path = root / f"embeds_chunk_{index}.npy"
    pmids_path = root / f"pmids_chunk_{index}.json"
    content_path = root / f"pubmed_chunk_{index}.json"
    for path in (embedding_path, pmids_path, content_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing official MedCPT PubMed chunk asset: {path}")
    embeddings = np.load(embedding_path, mmap_mode="r")
    pmids = json.loads(pmids_path.read_text(encoding="utf-8"))
    content = json.loads(content_path.read_text(encoding="utf-8"))
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError("official MedCPT embedding chunk must have shape [N,768]")
    if not isinstance(pmids, list) or len(pmids) != embeddings.shape[0]:
        raise ValueError("PMID list and embedding chunk length differ")
    if not isinstance(content, dict):
        raise TypeError("official MedCPT article content must be a PMID mapping")
    return embedding_path, pmids_path, content_path, embeddings, pmids, content


def article(content, pmid):
    value = content.get(str(pmid), {})
    if not isinstance(value, dict):
        raise TypeError(f"invalid MedCPT article metadata for PMID {pmid}")
    title = str(value.get("t", value.get("title", ""))).strip()
    abstract = str(value.get("a", value.get("abstract", ""))).strip()
    date = str(value.get("d", value.get("date", ""))).strip()
    if not abstract:
        return None
    return {
        "doc_id": f"PMID:{pmid}",
        "title": title,
        "content": abstract,
        "source": "PubMed-MedCPT-precomputed",
        "year": date[:4] if len(date) >= 4 and date[:4].isdigit() else "",
        "provenance": {
            "pmid": str(pmid),
            "medcpt_precomputed": True,
            "date": date,
        },
    }


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


def build_faiss(root, shards, backend, hnsw_m):
    if backend == "none":
        return None
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is optional. Install faiss-cpu/faiss-gpu explicitly or use "
            "--index-backend none."
        ) from exc
    if backend == "flat":
        index = faiss.IndexFlatIP(768)
    elif backend == "hnsw":
        index = faiss.IndexHNSWFlat(768, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
    else:
        raise ValueError("unsupported FAISS backend")
    rowids = []
    for shard in shards:
        matrix = np.load(root / shard["embedding"], mmap_mode="r")
        ids = np.load(root / shard["rowids"], mmap_mode="r")
        index.add(np.asarray(matrix, dtype=np.float32))
        rowids.append(np.asarray(ids, dtype=np.int64))
    mapping = np.concatenate(rowids)
    faiss.write_index(index, str(root / "medcpt.faiss"))
    np.save(root / "faiss-rowids.npy", mapping, allow_pickle=False)
    return {
        "backend": backend,
        "index": "medcpt.faiss",
        "rowids": "faiss-rowids.npy",
        "count": int(mapping.size),
        "index_sha256": sha256(root / "medcpt.faiss"),
        "rowids_sha256": sha256(root / "faiss-rowids.npy"),
        "hnsw_m": hnsw_m if backend == "hnsw" else None,
    }


def import_chunks(source, output, chunks, *, index_backend="none", hnsw_m=32):
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty KB directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir()
    (root / "rowids").mkdir()
    db_path = root / "documents.sqlite3"
    db = init_db(db_path)
    shards, source_assets = [], []
    total = skipped = 0

    for shard_index, chunk in enumerate(chunks):
        paths = load_chunk(source, chunk)
        emb_path, pmids_path, content_path, embeddings, pmids, content = paths
        accepted_indices, rowids = [], []
        for index, pmid in enumerate(pmids):
            record = article(content, pmid)
            if record is None:
                skipped += 1
                continue
            cursor = db.execute(
                "INSERT OR IGNORE INTO documents(doc_id,title,content,source,year,provenance_json) "
                "VALUES (?,?,?,?,?,?)",
                (
                    record["doc_id"],
                    record["title"],
                    record["content"],
                    record["source"],
                    record["year"],
                    json.dumps(record["provenance"], sort_keys=True),
                ),
            )
            if not cursor.rowcount:
                skipped += 1
                continue
            accepted_indices.append(index)
            rowids.append(int(cursor.lastrowid))
        db.commit()
        matrix = np.asarray(embeddings[accepted_indices], dtype="<f4")
        ids = np.asarray(rowids, dtype="<i8")
        emb_rel = Path("embeddings") / f"chunk-{chunk:03d}.npy"
        row_rel = Path("rowids") / f"chunk-{chunk:03d}.npy"
        np.save(root / emb_rel, matrix, allow_pickle=False)
        np.save(root / row_rel, ids, allow_pickle=False)
        shards.append(
            {
                "source_chunk": int(chunk),
                "embedding": emb_rel.as_posix(),
                "rowids": row_rel.as_posix(),
                "count": int(matrix.shape[0]),
                "embedding_sha256": sha256(root / emb_rel),
                "rowids_sha256": sha256(root / row_rel),
            }
        )
        source_assets.append(
            {
                "chunk": int(chunk),
                "embeddings_sha256": sha256(emb_path),
                "pmids_sha256": sha256(pmids_path),
                "content_sha256": sha256(content_path),
            }
        )
        total += int(matrix.shape[0])

    db.close()
    if not total:
        raise ValueError("official MedCPT import produced no abstract-bearing documents")
    faiss_spec = build_faiss(root, shards, index_backend, hnsw_m)
    payload = {
        "schema": "merit-medcpt-kb-v1",
        "source": "official_ncbi_medcpt_precomputed_pubmed",
        "documents": total,
        "duplicates_or_no_abstract_skipped": skipped,
        "embedding_dim": 768,
        "article_encoder": "official_precomputed_MedCPT_Article_Encoder",
        "sqlite": db_path.name,
        "sqlite_sha256": sha256(db_path),
        "shards": shards,
        "faiss": faiss_spec,
        "source_assets": source_assets,
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
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="artifacts/knowledge/medcpt-pubmed")
    parser.add_argument("--chunks", type=int, nargs="+", required=True)
    parser.add_argument("--index-backend", choices=("none", "flat", "hnsw"), default="none")
    parser.add_argument("--hnsw-m", type=int, default=32)
    args = parser.parse_args()
    if args.hnsw_m < 1 or len(set(args.chunks)) != len(args.chunks):
        parser.error("HNSW size must be positive and chunks must be unique")
    result = import_chunks(
        args.source,
        args.output,
        args.chunks,
        index_backend=args.index_backend,
        hnsw_m=args.hnsw_m,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
