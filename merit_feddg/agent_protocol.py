"""Label-free manifest, baseline provenance, atomic result and shard checks."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .agent_runtime import image_digest
from .evidence_agent import digest

INPUT_KEYS = {
    'id', 'image', 'question', 'image_sha256', 'answer_type', 'task', 'modality',
    'domain', 'domain_kind', 'role', 'group_id', 'subject_id', 'study_id',
}
SOURCE_KEYS = {
    'id', 'image', 'question', 'image_sha256', 'modality', 'domain', 'group_id',
    'subject_id', 'study_id', 'task', 'role', 'split', 'reference',
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def json_rows(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('nonempty JSONL dictionaries required')
    ids = [row.get('id') for row in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('unique nonempty IDs required')
    return rows


def read_inputs(path):
    rows = json_rows(path)
    seen = {}
    for row in rows:
        if set(row) - INPUT_KEYS:
            raise ValueError('unknown/label-bearing inference fields are forbidden')
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ('image', 'question', 'image_sha256')):
            raise ValueError('image, question and image_sha256 required')
        image = Path(row['image']).expanduser()
        if not image.is_absolute():
            image = Path(path).parent / image
        row['image'] = str(image.resolve())
        if row['image'] not in seen:
            seen[row['image']] = (file_hash(image), image_digest(image))
        fh, ph = seen[row['image']]
        if row['image_sha256'] not in (fh, ph):
            raise ValueError(f"image hash mismatch: {row['id']}")
        row['_file_sha256'], row['_pixel_sha256'] = fh, ph
    return rows


def read_sources(path, queries):
    """Small source-only corpus. No auto-download, target labels or target masks.

    Pixel checks apply to the full query manifest, not only the current shard.
    Patient/study checks apply where both corpora expose these identifiers.
    """
    if path is None:
        return [], {}, {'enabled': False, 'reason': 'no_source_manifest'}
    rows = json_rows(path)
    query_pixels = {r['_pixel_sha256'] for r in queries}
    query_ids = {r['id'] for r in queries}
    groups = {str(r.get('group_id', r['image_sha256'])) for r in queries}
    subjects = {str(r['subject_id']) for r in queries if r.get('subject_id') is not None}
    studies = {str(r['study_id']) for r in queries if r.get('study_id') is not None}
    references = {}
    for row in rows:
        if set(row) - SOURCE_KEYS or row.get('role') != 'source':
            raise ValueError('invalid source schema or role')
        if row.get('split') not in {'train', 'external'}:
            raise ValueError('retrieval corpus must be declared train/external')
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ('image', 'domain', 'group_id', 'question', 'modality')):
            raise ValueError('source image/domain/group/question/modality required')
        image = Path(row['image']).expanduser()
        if not image.is_absolute():
            image = Path(path).parent / image
        row['image'] = str(image.resolve())
        ph, fh = image_digest(image), file_hash(image)
        if row.get('image_sha256') not in (ph, fh):
            raise ValueError('source image hash mismatch or missing hash')
        if (row['id'] in query_ids or ph in query_pixels or row['group_id'] in groups
                or (row.get('subject_id') is not None and str(row['subject_id']) in subjects)
                or (row.get('study_id') is not None and str(row['study_id']) in studies)):
            raise ValueError('query/source image, group, study or patient overlap')
        row['image_sha256'] = ph
        if 'reference' in row:
            references[row['id']] = row.pop('reference')
    return rows, references, {
        'enabled': True, 'n': len(rows), 'manifest_sha256': file_hash(path),
        'content_sha256': digest([rows, references]),
        'patient_exclusion_available': bool(subjects) and all('subject_id' in r for r in rows),
        'patient_exclusion_guaranteed_without_ids': False,
    }


def load_incumbent(base_run, method, rows):
    base = Path(base_run)
    protocol = json.loads((base / 'protocol.json').read_text())
    if protocol.get('shards_complete') is not True:
        raise ValueError('base run must be finalized, not a partial shard')
    if method not in {'semantic_all', 'compact_rows', 'compact_all'}:
        raise ValueError('select an existing no-gate semantic/compact incumbent')
    ids = [row['id'] for row in rows]
    values = json.loads((base / f'{method}.json').read_text())
    if protocol.get('n') != len(ids) or set(values) != set(ids):
        raise ValueError('base run must match the complete inference manifest')
    if not protocol.get('identity') or not isinstance(protocol.get('config'), dict):
        raise ValueError('base protocol identity/config required')
    for value in values.values():
        if (not isinstance(value.get('text'), str) or not value.get('token_ids')
                or not isinstance(value.get('evidence'), list)
                or not isinstance(value.get('generation_config'), dict)):
            raise ValueError('base answer needs text, token IDs, native evidence and generation config')
        cfg = value['generation_config']
        if (cfg.get('evidence_style') != 'semantic' or cfg.get('vector_gate', 'off') != 'off'
                or cfg.get('semantic_spatial') or cfg.get('native_entry_transport')
                or cfg.get('block_tokens') != cfg.get('max_new_tokens')):
            raise ValueError('incumbent must be full-answer semantic transport without gate/spatial edits')
    return protocol, values


def merge_shards(root, identity, ids, methods, count):
    root = Path(root)
    outputs = {name: {} for name in methods}
    for shard in range(count):
        path = root / 'shards' / str(shard) / 'results.json'
        if not path.is_file():
            raise ValueError(f'missing shard {shard}; no partial result will be published')
        data = json.loads(path.read_text())
        expected = set(ids[shard::count])
        if (data.get('identity') != identity or data.get('shard_count') != count
                or data.get('shard_index') != shard or data.get('complete') is not True
                or set(data.get('outputs', {})) != set(methods)):
            raise ValueError('shard identity/methods/completion mismatch')
        for name in methods:
            values = data['outputs'][name]
            if set(values) != expected or set(outputs[name]).intersection(values):
                raise ValueError('shard rows incomplete, duplicated or incorrectly assigned')
            outputs[name].update(values)
    for name, values in outputs.items():
        atomic_json(root / f'{name}.json', {key: values[key] for key in ids})
    return outputs
