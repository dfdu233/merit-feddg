"""Fixed 24-case TRAIN-only-image pilot. No answers/references are read here."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from merit_feddg.anchored_revision import (
    assert_same_delivery,
    revision_prompt,
    select_train_probe,
)
from merit_feddg.open_study import atomic_json, fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data')
BASE = Path('/home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/'
            '0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6')
GPUS = {'GPU-3846413a-4238-d307-b1f3-10c2dfbe002c',
        'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023'}
SCORERS = ('anchor/corrected_sgta/evaluate_medheval_answers.py',
           'anchor/medeval/evaluate_mixed_vqa_table.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources():
    rows = [json.loads(s) for s in (DATA / 'train/manifest.jsonl').read_text().splitlines()]
    test = [json.loads(s) for s in (DATA / 'test/manifest.jsonl').read_text().splitlines()]
    protocol, old, base = [read(BASE / p) for p in
                           ('protocol.json', 'compact_rows.json', 'generalist.json')]
    if (not protocol['shards_complete'] or protocol['n'] != len(rows)
            or set(old) != set(base) or set(old) != {r['id'] for r in rows}):
        raise ValueError('Complete matching historical TRAIN outputs required')
    selected = select_train_probe(rows, old, {r['image_sha256'] for r in test}, 24)
    from merit_feddg.generalist_factory import generalist_provenance
    cfg = {
        'schema': 'anchored-revision-train-v1', 'selected': selected,
        'manifest': str(DATA / 'train/manifest.jsonl'), 'base': str(BASE),
        'input_hashes': {str(p): sha(p) for p in (DATA / 'train/manifest.jsonl',
            DATA / 'test/manifest.jsonl', BASE / 'protocol.json',
            BASE / 'compact_rows.json', BASE / 'generalist.json')},
        'source': {str(p.relative_to(ROOT)): sha(p) for p in
                   [Path(__file__), *sorted((ROOT / 'merit_feddg').glob('*.py'))]},
        'scorers': {p: sha(Path('/home/dbw/ANCHOR') / p) for p in SCORERS},
        'model': generalist_provenance(protocol['config']['generalist'], str(ROOT / 'artifacts')),
        'selection': 'hash order; one question per TRAIN-only image with historical evidence; no scores',
        'patient_disjoint': 'unknown: patient identifiers unavailable',
        'arms': ['generalist', 'compact', 'revision_no_evidence', 'revision_evidence'],
        'pilot_only': True, 'full_benchmark': False,
        'hypothesis': 'anchored evidence revision reduces harm while retaining expert gains',
        'no_training': True, 'no_decision_threshold': True,
        'literature': 'RARR ACL2023 editor preservation principle; NOT full RARR replication',
    }
    return cfg, {r['id']: r for r in rows}, old, base, protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--gpu-uuid', choices=sorted(GPUS))
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--max-cases', type=int, default=0, help='Scheduling stop only, not a new selection')
    a = p.parse_args()
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard')
    cfg, rows, old, base, protocol = resources()
    identity = fingerprint(cfg)
    a.output.mkdir(parents=True, exist_ok=True)
    frozen = a.output / 'frozen.json'
    if frozen.exists() and read(frozen) != cfg:
        raise ValueError('Frozen experiment changed; preserve old output')
    if not frozen.exists():
        atomic_json(frozen, cfg)
    print('PREFLIGHT', identity, len(cfg['selected']), flush=True)
    if a.check_only:
        return
    actual = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
                                     '--format=csv,noheader'], text=True, timeout=10).strip()
    if actual != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Unauthorized or unexpected device mapping')
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    if free < 22000:
        raise RuntimeError('Insufficient safe memory headroom; do not disturb existing jobs')
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    from merit_feddg.matched_evaluation import generation_prompt
    from merit_feddg.open_data import pixel_digest
    torch.set_num_threads(4)
    torch.ones(1, device='cuda:0').sum().item()
    start = time.perf_counter()
    probe = load_generalist(protocol['config']['generalist'], str(ROOT / 'artifacts'))
    probe.model.eval().requires_grad_(False)
    atomic_json(a.output / f'load-{time.time_ns()}.json', {
        'identity': identity, 'gpu_uuid': actual, 'seconds': time.perf_counter() - start,
        'torch': torch.__version__, 'cuda': torch.version.cuda})
    completed = 0
    for i, key in enumerate(cfg['selected']):
        if i % a.shard_count != a.shard_index:
            continue
        target = a.output / 'cases' / (key + '.json')
        if target.exists():
            if read(target)['identity'] != identity:
                raise ValueError('Saved identity mismatch')
            continue
        started = time.perf_counter()
        row, incumbent, baseline = rows[key], old[key], base[key]
        if pixel_digest(row['image']) != row['image_sha256']:
            raise ValueError('TRAIN image identity changed')
        gen = ValueGenerationConfig(**incumbent['generation_config'])
        if gen.semantic_spatial or gen.visual_views or gen.evidence_style != 'semantic':
            raise ValueError('This isolated pilot requires the existing text delivery protocol')
        items = tuple(EvidenceItem(**e) for e in incumbent['evidence'])
        prompt = generation_prompt(row, protocol['config'])
        revision = revision_prompt(prompt, baseline['text'])
        arms = {}
        # Exact controls first. A mismatch stops this case before experimental generation.
        for name, text_prompt, evidence, generation in (
            ('generalist', prompt, (), ValueGenerationConfig(**baseline['generation_config'])),
            ('compact', prompt, items, gen),
            ('revision_no_evidence', revision, (), gen),
            ('revision_evidence', revision, items, gen),
        ):
            session = NativeSession(probe, row['image'], text_prompt, row['question'], generation)
            image, compiled = session.context(NativeState(items=evidence))
            if name == 'revision_evidence':
                assert_same_delivery(arms['compact']['transport'], session.last_transport)
                if not session.last_transport['presented']:
                    raise ValueError('No evidence delivered; not a valid evidence intervention')
            set_seed(42)
            t = time.perf_counter()
            with torch.inference_mode():
                block = probe.new_answer_session(image, compiled).propose(
                    (), count=1, length=generation.max_new_tokens)[0]
            value = {'text': block.text, 'token_ids': list(block.tokens),
                     'seconds': time.perf_counter() - t, 'finished': block.finished,
                     'transport': session.last_transport, 'input_prompt_sha256':
                     hashlib.sha256(compiled.encode()).hexdigest(), 'model_calls': 1}
            if name in ('generalist', 'compact'):
                prior = baseline if name == 'generalist' else incumbent
                value['historical_token_parity'] = value['token_ids'] == prior['token_ids']
                value['historical_text_parity'] = value['text'] == prior['text']
                if not value['historical_token_parity'] or not value['historical_text_parity']:
                    atomic_json(a.output / 'failures' / (key + '-' + name + '.json'), value)
                    raise RuntimeError('Historical control parity failed; no threshold relaxation')
            if not value['text'].strip():
                atomic_json(a.output / 'failures' / (key + '-' + name + '.json'), value)
                raise RuntimeError('Empty real candidate; stop rather than claim fallback success')
            arms[name] = value
        atomic_json(target, {'id': key, 'identity': identity, 'image_sha256': row['image_sha256'],
            'arms': arms, 'gpu_uuid': actual, 'wall_seconds': time.perf_counter() - started,
            'inherited_compact_seconds': incumbent['seconds'],
            'inherited_generalist_seconds': baseline['seconds'],
            'native_evidence_sha256': fingerprint(incumbent['evidence'])})
        print('DONE', i + 1, key, 'delivered', len(arms['compact']['transport']['presented']),
              'changed', arms['revision_evidence']['text'] != baseline['text'], flush=True)
        completed += 1
        if a.max_cases and completed >= a.max_cases:
            break
    print('SCHEDULING_STOP', completed, 'new cases; not full benchmark completion', flush=True)


if __name__ == '__main__':
    main()
