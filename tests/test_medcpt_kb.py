import gzip
import json
import sqlite3

import numpy as np

from merit_feddg.capabilities import CapabilityRequest
from merit_feddg.experts.medcpt_retriever import MedCPTRetrievalExpert
from merit_feddg.native_evidence import compile_evidence
from scripts import build_medcpt_kb


def write_kb(tmp_path):
    query = tmp_path / "query"
    query.mkdir()
    root = tmp_path / "kb"
    (root / "embeddings").mkdir(parents=True)
    (root / "rowids").mkdir()
    db = sqlite3.connect(root / "documents.sqlite3")
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
    rows = [
        ("PMID:1", "heart", "cardiac finding", "PubMed", "2025", "{}"),
        ("PMID:2", "lung", "pulmonary nodule", "PubMed", "2024", "{}"),
        ("PMID:3", "brain", "intracranial finding", "PubMed", "2023", "{}"),
    ]
    db.executemany(
        "INSERT INTO documents(doc_id,title,content,source,year,provenance_json) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    db.close()
    embeds = np.zeros((3, 768), dtype=np.float32)
    embeds[0, 0] = 0.1
    embeds[1, 0] = 0.9
    embeds[2, 0] = 0.2
    np.save(root / "embeddings/shard-00000.npy", embeds)
    np.save(root / "rowids/shard-00000.npy", np.array([1, 2, 3], dtype=np.int64))
    manifest = {
        "schema": "merit-medcpt-kb-v1",
        "embedding_dim": 768,
        "documents": 3,
        "fingerprint": "sha256:test",
        "sqlite": "documents.sqlite3",
        "shards": [
            {
                "embedding": "embeddings/shard-00000.npy",
                "rowids": "rowids/shard-00000.npy",
                "count": 3,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return query, root


def request():
    return CapabilityRequest(
        sample_id="s",
        image="/tmp/unused.png",
        question="What pulmonary finding is described?",
        modality="cxr",
        task="open_vqa",
        domain="source",
        group_id="g",
        capability="retrieval",
        scope="biomedical_literature",
        query="pulmonary nodule",
    )


def test_medcpt_retriever_reads_real_kb_contract_without_patient_truth(tmp_path, monkeypatch):
    query, root = write_kb(tmp_path)
    expert = MedCPTRetrievalExpert(
        str(query),
        "medcpt",
        str(root / "manifest.json"),
        candidate_k=3,
        top_k=2,
    )
    monkeypatch.setattr(
        expert,
        "_query_embedding",
        lambda _text: np.r_[np.array([1.0], dtype=np.float32), np.zeros(767, dtype=np.float32)],
    )
    result = expert.infer(request())
    assert result.reason == "ok"
    assert len(result.items) == 2
    first = result.items[0]
    reference = first.payload["references"][0]
    assert reference["document_id"] == "PMID:2"
    assert reference["patient_specific_claim"] == "not_established"
    assert reference["retrieval_rank"] == 1
    assert first.confidence is None
    assert first.provenance["target_answers_used"] is False
    assert result.items[0].evidence_id != result.items[1].evidence_id


def test_medcpt_missing_model_has_actionable_download_hint(tmp_path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    try:
        MedCPTRetrievalExpert(
            str(tmp_path / "missing"),
            "medcpt",
            str(root / "manifest.json"),
        )
    except FileNotFoundError as exc:
        assert "prepare_medcpt_assets.py" in str(exc)
    else:
        raise AssertionError("missing model must fail")


def test_pubmed_xml_stream_extracts_real_citation_fields(tmp_path):
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>123</PMID><Article><ArticleTitle>Example <i>study</i></ArticleTitle>
    <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
    <Abstract><AbstractText Label="BACKGROUND">First sentence.</AbstractText>
    <AbstractText>Second sentence.</AbstractText></Abstract></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    path = tmp_path / "pubmed26n0001.xml.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(xml)
    rows = list(build_medcpt_kb.pubmed_records(tmp_path))
    assert rows == [
        {
            "id": "PMID:123",
            "title": "Example study",
            "content": "First sentence. Second sentence.",
            "source": "PubMed",
            "year": "2026",
            "provenance": {
                "pubmed_baseline_file": "pubmed26n0001.xml.gz",
                "pmid": "123",
            },
        }
    ]


def test_builder_limit_is_exact_with_fake_article_encoder(tmp_path, monkeypatch):
    class FakeEncoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, rows):
            result = np.zeros((len(rows), 768), dtype=np.float32)
            for index in range(len(rows)):
                result[index, 0] = index + 1
            return result

    monkeypatch.setattr(build_medcpt_kb, "ArticleEncoder", FakeEncoder)
    source = (
        {
            "id": f"D{i}",
            "title": f"T{i}",
            "content": f"C{i}",
            "source": "test",
            "year": "",
            "provenance": {},
        }
        for i in range(20)
    )
    payload = build_medcpt_kb.build(
        source,
        tmp_path / "out",
        tmp_path / "article",
        limit=5,
        batch_size=2,
        shard_size=3,
    )
    assert payload["documents"] == 5
    assert sum(shard["count"] for shard in payload["shards"]) == 5


def test_medcpt_documents_are_atomic_for_whole_record_token_packing(tmp_path, monkeypatch):
    query, root = write_kb(tmp_path)
    expert = MedCPTRetrievalExpert(
        str(query),
        "medcpt",
        str(root / "manifest.json"),
        candidate_k=3,
        top_k=3,
    )
    monkeypatch.setattr(
        expert,
        "_query_embedding",
        lambda _text: np.r_[np.array([1.0], dtype=np.float32), np.zeros(767, dtype=np.float32)],
    )
    result = expert.infer(request())
    assert len(result.items) == 3
    assert all(len(item.payload["references"]) == 1 for item in result.items)
    assert [
        item.payload["references"][0]["retrieval_rank"] for item in result.items
    ] == [1, 2, 3]


def test_medcpt_literature_survives_generic_native_retrieval_compiler(tmp_path, monkeypatch):
    query, root = write_kb(tmp_path)
    expert = MedCPTRetrievalExpert(
        str(query),
        "medcpt",
        str(root / "manifest.json"),
        candidate_k=3,
        top_k=1,
    )
    monkeypatch.setattr(
        expert,
        "_query_embedding",
        lambda _text: np.r_[np.array([1.0], dtype=np.float32), np.zeros(767, dtype=np.float32)],
    )
    result = expert.infer(request())
    records = compile_evidence(result.items, request().question, max_chars=5000)
    assert len(records) == 1
    assert records[0]["payload"]["references"][0]["title"] == "lung"
    assert "pulmonary nodule" in records[0]["payload"]["references"][0]["content"]
