"""Local, provenance-preserving text retrieval. No patients, embeddings or downloads."""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path


def build(seed, output):
    raw = Path(seed).read_bytes()
    rows = json.loads(raw)
    required = {'id', 'text', 'source_url', 'license', 'scope', 'source_family'}
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique seed IDs required')
    for row in rows:
        if set(row) != required or not all(isinstance(v, str) and v for v in row.values()):
            raise ValueError('seed requires reviewed text, source, license and scope')
        if not row['source_url'].startswith('https://') or row['scope'] != 'general_knowledge':
            raise ValueError('only sourced general knowledge seeds are supported')
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace a frozen corpus.
    with path.open('xb'):
        pass
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE metadata (sha256 TEXT)')
        db.execute('INSERT INTO metadata VALUES (?)', (hashlib.sha256(raw).hexdigest(),))
        db.execute('CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, text, record UNINDEXED)')
        db.executemany('INSERT INTO docs VALUES (?, ?, ?)',
                       [(r['id'], r['text'], json.dumps(r)) for r in rows])
    return {'documents': len(rows), 'seed_sha256': hashlib.sha256(raw).hexdigest()}


def query(index, text, limit=3):
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be 1..20')
    words = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    if not words:
        return []
    match = ' OR '.join('"' + w + '"' for w in words)
    with sqlite3.connect(Path(index).resolve().as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT record, bm25(docs) FROM docs WHERE docs MATCH ? '
                          'ORDER BY bm25(docs), id LIMIT ?', (match, limit)).fetchall()
    return [{**json.loads(record), 'lexical_rank_score': score,
             'current_patient_fact': False} for record, score in rows]


class TextKnowledgeExpert:
    def __init__(self, model_id, expert_id='text_knowledge', scope='general_knowledge'):
        self.index, self.expert_id, self.scope = model_id, expert_id, scope
        if not Path(model_id).is_file():
            raise FileNotFoundError('build the reviewed text index first')

    def infer(self, request):
        from .capabilities import CapabilityResult, EvidenceItem
        hits = query(self.index, request.question)
        if not hits:
            return CapabilityResult(self.expert_id, 'retrieval', (), reason='no_text_match')
        payload = {'knowledge': hits, 'scope': 'general_knowledge', 'patient_observation': False}
        item = EvidenceItem(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            self.expert_id, 'retrieval', self.scope, payload, confidence=None,
            provenance={'source_type': 'reviewed_public_text', 'independent_patient_vote': False})
        return CapabilityResult(self.expert_id, 'retrieval', (item,))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--seed', required=True); b.add_argument('--output', required=True)
    q = sub.add_parser('query'); q.add_argument('--index', required=True); q.add_argument('--text', required=True)
    args = p.parse_args()
    print(json.dumps(build(args.seed, args.output) if args.command == 'build'
                     else query(args.index, args.text), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
