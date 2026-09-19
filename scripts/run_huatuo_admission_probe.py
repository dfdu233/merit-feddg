"""Fixed TRAIN-only Huatuo probe; original MERIT routing and expert acquisition.

Read-only reuse of the current server backend, pinned before any GPU stage.
No references are read by this program. No patient data is exported to git.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
COVERAGE = Path('/home/dbw/merit-feddg-expert-coverage')
sys.path.insert(0, str(COVERAGE / 'scripts'))
import run_huatuo_merit_full as full

checked, native = full.checked, full.native
import merit_feddg

merit_feddg.__path__.append(str(HERE / 'merit_feddg'))
from merit_feddg.context_admission import admission_prompt, admitted_view

DATA = Path('/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data')
SELECTION = Path('/home/dbw/merit-feddg-anchored-revision/runs/pilot-v4/frozen.json')


def rows(a):
    from anchor.corrected_sgta.protocol_v2 import build_prompt

    from merit_feddg.open_data import pixel_digest
    if os.environ.get('HUATUO_TRAIN_MANIFEST'):
        manifest = Path(os.environ['HUATUO_TRAIN_MANIFEST'])
        schedule = native.read(Path(os.environ['HUATUO_TRAIN_SCHEDULE']))
        if schedule['dataset'] != a.dataset:
            raise ValueError('Schedule dataset mismatch')
        values = {r['id']:r for r in map(json.loads,manifest.read_text().splitlines())}
        selected = [values[k] for k in schedule['ids']]
        for row in selected:
            if set(row) & {'answer','answers','label','reference','references'}:
                raise ValueError('Labels in generation manifest')
            if native.sha(row['image']) != row['image_sha256'] or pixel_digest(row['image']) != row['pixel_sha256']:
                raise ValueError('Scheduled image changed')
        return selected
    start = int(os.environ.get('HUATUO_TRAIN_CASE_START', '0'))
    count = int(os.environ.get('HUATUO_TRAIN_CASE_COUNT', '4'))
    candidates = native.read(SELECTION)['selected']
    if start < 0 or count < 1 or start + count > len(candidates):
        raise ValueError('Invalid scheduling slice of existing TRAIN probe')
    selected = candidates[start:start + count]
    all_rows = {r['id']: r for r in map(json.loads, (DATA / 'train/manifest.jsonl').read_text().splitlines())}
    test_images = {r['image_sha256'] for r in map(json.loads, (DATA / 'test/manifest.jsonl').read_text().splitlines())}
    result = []
    for key in selected:
        row = dict(all_rows[key])
        if row['image_sha256'] in test_images or pixel_digest(row['image']) != row['image_sha256']:
            raise ValueError('Require unchanged TRAIN-only image')
        if set(row) & {'answer', 'answers', 'label', 'reference', 'references'}:
            raise ValueError('Labels in generation input')
        row['benchmark_prompt'] = build_prompt({'question': row['question'],
            'question_type': 'binary' if row['answer_type'] == 'closed' else 'open'})
        row['image_sha256'] = native.sha(row['image'])
        result.append(row)
    return result


def protocol(a):
    from merit_feddg.capability_runtime import ValueGenerationConfig
    from merit_feddg.matched_evaluation import experiment_arms
    specs, config, excluded = native.registry(a)
    base = checked.base()
    arm = experiment_arms(ValueGenerationConfig(**base['capability_value']['generation']),
                          'verified_packets')['compact_rows']
    if arm.max_new_tokens != 1024:
        raise ValueError('Formal 1024-token answer budget required')
    paths = [Path(__file__), HERE / 'merit_feddg/context_admission.py',
             Path(full.__file__), Path(checked.__file__), Path(native.__file__),
             COVERAGE / 'merit_feddg/huatuo_generalist.py',
             COVERAGE / 'scripts/native_quilt_infer.py',
             COVERAGE / 'merit_feddg/experts/native_coverage.py',
             Path('/home/dbw/ANCHOR/anchor/corrected_sgta/protocol_v2.py')]
    payload = {'method': 'Huatuo native MERIT plus isolated packet admission',
        'rows': rows(a), 'base': base, 'registry': specs, 'coverage_config': config,
        'excluded': excluded, 'generation_config': asdict(arm),
        'selection': 'explicit label-free full-TRAIN manifest and fixed image schedule' if os.environ.get('HUATUO_TRAIN_MANIFEST') else 'predeclared scheduling slice of existing TRAIN-only-image probe; not scores',
        'schedule_start': None if os.environ.get('HUATUO_TRAIN_MANIFEST') else int(os.environ.get('HUATUO_TRAIN_CASE_START', '0')),
        'schedule_count': len(rows(a)),
        'native_only': os.environ.get('HUATUO_NATIVE_ONLY') == '1',
        'source': {str(p): native.sha(p) for p in paths},
        'formal_sources': {str(p): native.sha(p) for p in (native.FORMAL / 'merit_feddg').glob('*.py')},
        'manifest_sha256': native.sha(Path(os.environ.get('HUATUO_TRAIN_MANIFEST', DATA / 'train/manifest.jsonl'))),
        'schedule_sha256': native.sha(os.environ['HUATUO_TRAIN_SCHEDULE']) if os.environ.get('HUATUO_TRAIN_SCHEDULE') else None,
        'test_image_exclusion_sha256': native.read(os.environ['HUATUO_TRAIN_SCHEDULE'])['source_test_sha256'] if os.environ.get('HUATUO_TRAIN_SCHEDULE') else native.sha(DATA / 'test/manifest.jsonl'),
        'gate': 'same prior frozen prompts; relevance A/C/D, scope A/B/C/D; 8 tokens; no threshold',
        'new_training': False, 'no_test_tuning': True}
    identity = native.fingerprint(payload)
    path = a.output / 'protocol.json'
    value = dict(identity=identity, **payload)
    if path.exists() and native.read(path) != value:
        raise ValueError('Pinned dependencies/config/data changed')
    if not path.exists():
        native.atomic_json(path, value)
    return identity, specs, config, arm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['check', 'route', 'experts', 'admission'], required=True)
    p.add_argument('--dataset', choices=['vqa_rad','slake'], default='vqa_rad')
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.shard_index, a.canary = 0, False
    full.rows, full.protocol = rows, protocol
    native.rows_for, native.protocol, native.base = rows, protocol, checked.base
    original_native_row = native.native_row
    def train_native_row(row, route):
        return original_native_row(row, route) | {'domain': 'official-train', 'role': 'source'}
    native.native_row = train_native_row
    identity, specs, config, arm = protocol(a)
    print('PREFLIGHT', identity, len(rows(a)), flush=True)
    if a.stage == 'check':
        return
    if a.stage in ('route', 'experts'):
        full.main()
        return
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeState
    native.gpu_check(a.gpu_uuid)
    started = time.perf_counter()
    probe = checked.HuatuoGeneralist(checked.base()['generalist']['checkpoint_path'])
    native.atomic_json(a.output / 'model-load.json', {'seconds': time.perf_counter() - started})
    for index, row in enumerate(rows(a)):
        path = a.output / 'cases' / (row['id'] + '.json')
        if path.exists():
            saved = native.read(path)
            if saved['identity'] != identity or not saved['complete']:
                raise ValueError('Incomplete or incompatible prior case; preserve it')
            continue
        route = full.saved_route(a, row, identity)
        shared = checked.SharedExpertPool(checked.CacheOnly(), a.output / 'expert-cache' /
                                        native.fingerprint(row['id']), identity)
        set_seed(42)
        session = checked.NativeSession(probe, row['image'], row['benchmark_prompt'], row['question'], arm)
        engine = checked.CapabilityRuntime(session, shared, native.native_row(row, route),
                                           native.eligible_registry(specs, config, row), arm)
        start = time.perf_counter()
        with torch.inference_mode():
            compact = engine.run('all_evidence')
        compact['wall_seconds'] = time.perf_counter() - start
        if any('runtime_error' in str(t.get('reason', '')) for t in compact['trace']):
            native.atomic_json(a.output / 'failed-native.json', compact)
            raise RuntimeError('Native specialist failure, no silent fallback')
        native_items = tuple(EvidenceItem(**e) for e in compact['evidence'])
        replay = checked.NativeSession(probe, row['image'], row['benchmark_prompt'], row['question'], arm)
        _, original = replay.context(NativeState(items=native_items))
        delivered = {(e['expert_id'], e['evidence_id']) for e in replay.last_transport['presented']}
        items = tuple(e for e in native_items if (e.expert_id, e.evidence_id) in delivered)
        _, reduced = replay.context(NativeState(items=items))
        if reduced != original:
            raise RuntimeError('Delivered-universe restriction changed the original prompt')
        set_seed(42)
        start = time.perf_counter()
        baseline = probe.generate_with_usage(row['image'], row['benchmark_prompt'], arm.max_new_tokens)
        baseline['seconds'] = time.perf_counter() - start
        out = {'id': row['id'], 'identity': identity, 'complete': False,
               'route': route, 'compact_raw': compact, 'arms': {'generalist': baseline, 'compact': compact},
               'gates': {}, 'delivered_packets': len(items)}
        if index == 0:
            set_seed(42)
            replayed = probe.generate_with_usage(row['image'], original, arm.max_new_tokens)
            out['compact_parity'] = (replayed['token_ids'] == compact['token_ids'] and
                                    replayed['text'] == compact['text'])
            if not out['compact_parity']:
                native.atomic_json(path, out)
                raise RuntimeError('Native MERIT token/text replay parity failed')
        native.atomic_json(a.output / 'native-controls' / (row['id'] + '.json'), out)
        for policy in (() if os.environ.get('HUATUO_NATIVE_ONLY') == '1' else ('relevance', 'scope')):
            judgments = []
            for item in items:
                prompt = admission_prompt(row['question'], item, specs[item.expert_id], policy)
                if not probe.context_token_budget(row['image'], prompt, 8)['fits']:
                    raise RuntimeError('Judge input unavailable; do not count a fallback as a decision')
                set_seed(42)
                start = time.perf_counter()
                value = probe.generate_with_usage(row['image'], prompt, 8,
                    allowed_texts=['A', 'C', 'D'] if policy == 'relevance' else ['A', 'B', 'C', 'D'])
                judgments.append({'expert_id': item.expert_id, 'evidence_id': item.evidence_id,
                    'label': value['text'], 'seconds': time.perf_counter() - start, 'usage': value})
            out['gates'][policy] = judgments
            kept = admitted_view(items, [v['label'] for v in judgments])
            if not kept or kept == items:
                answer = dict(baseline if not kept else compact)
                answer['reuse'] = 'generalist' if not kept else 'compact'
                answer['new_answer_calls'] = 0
            else:
                view = checked.NativeSession(probe, row['image'], row['benchmark_prompt'], row['question'], arm)
                image, prompt = view.context(NativeState(items=kept))
                actual = [(e['expert_id'], e['evidence_id']) for e in view.last_transport['presented']]
                if actual != [(e.expert_id, e.evidence_id) for e in kept]:
                    raise RuntimeError('Admitted evidence not delivered exactly')
                set_seed(42)
                start = time.perf_counter()
                answer = probe.generate_with_usage(image, prompt, arm.max_new_tokens)
                answer.update(seconds=time.perf_counter() - start, new_answer_calls=1,
                              reuse=None, transport=view.last_transport)
            out['arms'][policy] = answer
        out['complete'] = True
        native.atomic_json(path, out)
        print('DONE', row['id'], {k: [v['label'] for v in j] for k, j in out['gates'].items()}, flush=True)
    native.atomic_json(a.output / 'complete.json', {'identity': identity, 'n': len(rows(a)),
                                                  'pilot_only': True})


if __name__ == '__main__':
    main()
