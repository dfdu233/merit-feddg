"""ACD on the unchanged historical full VQA-RAD channel branches.

No new experts, prompts, parser, verifier, or fitted parameters. Fixed controls
are inherited from the completed channel experiment, not regenerated as results.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from run_channel_soft_canary import visible
from run_class_text_guidance import cad_decode
from run_soft_guidance_full import sha
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist, generalist_provenance
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.open_data import pixel_digest
from merit_feddg.persistent_scores import PersistentScores
from merit_feddg.uncertainty_decode import decode_adaptive

HISTORY = Path('/home/dbw/merit-feddg-soft-guidance/runs/class-text-guidance-v1/13a3e59cbcb99caa714fea919b3b50fb99c80ae6b9aa90e3c8b8879872e35f35')
GPUS = ('GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023', 'GPU-3846413a-4238-d307-b1f3-10c2dfbe002c')
CHANNELS = ('classification', 'generation')


def validate_content(root):
    source = json.loads((root/'frozen.json').read_text())
    assert fingerprint(source) == root.name
    for path, digest in source['source'].items():
        assert sha(path) == digest, path
    assert sha(Path(__file__).with_name('run_soft_guidance_full.py')) == source['runner_sha256']
    assert source['runtime'] == {'torch': torch.__version__, 'cuda': torch.version.cuda, 'threads': 4}
    for data in source['datasets'].values():
        assert sha(data['manifest']) == data['manifest_sha256']
    spec = source['datasets']['vqarad']['protocol']['config']['generalist']
    actual = json.loads(json.dumps(generalist_provenance(spec, 'artifacts')))
    expected = source['model']
    hashes = {}
    for current, previous in ((actual, expected), (actual['vision_tower'], expected['vision_tower'])):
        assert current['checkpoint_path'] == previous['checkpoint_path']
        assert [(x[0], x[1]) for x in current['file_stats']] == [(x[0], x[1]) for x in previous['file_stats']]
        for name, size, mtime in current['file_stats']:
            old_mtime = next(x[2] for x in previous['file_stats'] if x[0] == name)
            if mtime == old_mtime:
                continue
            path = Path(current['checkpoint_path'])/name
            assert name.endswith(('.safetensors', '.bin')), 'unexpected non-weight change'
            metadata = path.parent/'.cache/huggingface/download'/(name+'.metadata')
            expected_hash = metadata.read_text().splitlines()[1] if metadata.exists() else path.resolve().name
            assert len(expected_hash) == 64, 'missing original content identity'
            digest = sha(path)
            assert digest == expected_hash, 'weight content mismatch'
            hashes[str(path)] = digest
        current['file_stats'] = previous['file_stats']
    assert actual == expected, 'non-mtime model/source identity mismatch'
    return source, spec, hashes


def resources():
    previous = json.loads((HISTORY/'frozen.json').read_text())
    assert fingerprint(previous) == HISTORY.name
    assert json.loads((HISTORY/'complete.json').read_text())['full_dataset_complete']
    source, spec, verified_hashes = validate_content(Path(previous['source_run']))
    data = source['datasets']['vqarad']
    ids = {r['id'] for r in data['rows']}
    assert len(ids) == len(data['rows']) == 451
    for channel in CHANNELS:
        assert {p.stem for p in (HISTORY/'vqarad'/channel).glob('*.json')} == ids
    frozen = {'schema': 'historical-channel-acd-v1', 'history': str(HISTORY),
        'source_run': previous['source_run'], 'manifest_sha256': data['manifest_sha256'],
        'channels': CHANNELS, 'mode': 'acd', 'formula': 's=H0/(H0+HE); z=(1-s)*z0+s*zE',
        'source_modes': 'unavailable: historical delivered catalog/text lacks matched native source uncertainty',
        'scorer': previous['scorer'], 'fit': False, 'test_parameter_selection': False,
        'mtime_drift_verified_weight_hashes': verified_hashes,
        'files': {p: sha(p) for p in ('scripts/run_vqarad_dynamic.py',
            'merit_feddg/uncertainty_decode.py', 'merit_feddg/persistent_scores.py')},
        'controls': 'completed historical text/without_channel/blend/cad; no overwrite',
        'history_records': {c: {k: sha(HISTORY/'vqarad'/c/(k+'.json')) for k in sorted(ids)} for c in CHANNELS}}
    return data, spec, json.loads(json.dumps(frozen))


def run_case(probe, row, old, protocol, historic, audit):
    start = time.perf_counter()
    cfg = ValueGenerationConfig(**old['generation_config'])
    assert cfg.evidence_style == 'semantic' and not cfg.semantic_spatial and cfg.vector_gate == 'off'
    native = tuple(EvidenceItem(**e) for e in old['evidence'])
    prompt = generation_prompt(row, protocol['config'])
    def context(items):
        session = NativeSession(probe, row['image'], prompt, row['question'], cfg)
        image, rendered = session.context(NativeState(items=items))
        return probe.new_answer_session(image, rendered), session.last_transport
    conditioned, with_transport = context(native)
    delivered = visible(with_transport)
    assert delivered == visible(historic['original_transport'])
    selected = {tuple(x) for x in historic['selected_evidence']}
    other = tuple(e for e in native if (e.expert_id, e.evidence_id) in delivered-selected)
    base, without_transport = context(other)
    assert visible(without_transport) == delivered-selected == visible(historic['without_transport'])
    eos = probe.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    checks = {}
    if audit:
        for name, session, expected in [('text', conditioned, historic['current_incumbent']),
                                        ('without_channel', base, historic['without_channel'])]:
            generated = session.propose((), count=1, length=cfg.max_new_tokens)[0]
            checks[name] = list(generated.tokens) == expected['token_ids'] and generated.text.strip() == expected['text'].strip()
            assert checks[name], 'historical endpoint parity: '+name
            cached = PersistentScores(session)
            for prefix in ((), tuple(expected['token_ids'][:2])):
                assert np.array_equal(session.next_scores(prefix), cached.next_scores(prefix)), 'KV score mismatch'
            checks[name+'_cache_exact'] = True
            del cached
        cad = cad_decode(PersistentScores(base), PersistentScores(conditioned), cfg.max_new_tokens, eos_ids)
        checks['fixed_cad'] = cad['token_ids'] == historic['cad']['token_ids']
        assert checks['fixed_cad'], 'historical fixed CAD parity'
    verification_seconds = time.perf_counter()-start
    a, b = PersistentScores(base), PersistentScores(conditioned)
    result = decode_adaptive(a, b, max_tokens=cfg.max_new_tokens, eos_ids=eos_ids, mode='acd')
    assert result['text'] and result['token_ids']
    return {'status': 'real_candidate', 'acd': result, 'checks': checks,
        'selected_evidence': sorted(selected), 'original_transport': with_transport,
        'without_transport': without_transport, 'verification_and_context_seconds': verification_seconds,
        'wall_seconds': time.perf_counter()-start, 'new_expert_calls': 0,
        'prefills': a.prefills+b.prefills, 'incremental_forwards': a.incremental_calls+b.incremental_calls,
        'source_modes_status': 'unavailable_native_source_uncertainty'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--shard-index', required=True, type=int, choices=(0, 1))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    data, spec, frozen = resources()
    root = args.output/fingerprint(frozen)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'frozen.json').exists():
        assert json.loads((root/'frozen.json').read_text()) == frozen
    else:
        atomic_json(root/'frozen.json', frozen)
    print('ROOT', root, flush=True)
    if args.check_only:
        return
    with (root/f'.shard-{args.shard_index}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = args.shard_index
        actual = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
        assert actual == GPUS[index] and os.environ['CUDA_VISIBLE_DEVICES'] == str(index)
        torch.set_num_threads(4)
        torch.ones(1, device='cuda:0').sum().item()
        started = time.perf_counter()
        probe = load_generalist(spec, 'artifacts')
        probe.model.eval().requires_grad_(False)
        atomic_json(root/f'load-{index}-{time.time_ns()}.json', {'seconds': time.perf_counter()-started, 'gpu_uuid': actual})
        for channel in CHANNELS:
            audited = False
            dest = root/channel
            dest.mkdir(exist_ok=True)
            for ordinal, row in enumerate(data['rows']):
                if ordinal % 2 != index:
                    continue
                path = dest/(row['id']+'.json')
                if path.exists():
                    saved = json.loads(path.read_text())
                    assert saved['identity'] == root.name and saved['id'] == row['id']
                    audited = audited or bool(saved.get('checks'))
                    if args.canary and audited:
                        break
                    continue
                historic = json.loads((HISTORY/'vqarad'/channel/(row['id']+'.json')).read_text())
                assert historic['identity'] == HISTORY.name and historic['id'] == row['id']
                if historic['status'] != 'real_candidate':
                    result = {'status': 'unavailable', 'reason': historic['reason'],
                              'acd': historic['current_incumbent'], 'new_expert_calls': 0, 'wall_seconds': 0.0}
                else:
                    assert pixel_digest(row['image']) == row['image_sha256'], 'image changed'
                    entry = json.loads((Path(data['base'])/'case-cache/compact_rows'/(fingerprint(row['id'])+'.json')).read_text())
                    assert entry['identity'] == data['protocol']['identity']
                    with torch.inference_mode():
                        result = run_case(probe, row, entry['output'], data['protocol'], historic, not audited)
                    audited = True
                atomic_json(path, {'identity': root.name, 'id': row['id'], 'channel': channel,
                    'shard': index, 'gpu_uuid': actual, **result})
                print('DONE', channel, row['id'], result['status'], result['wall_seconds'], flush=True)
                if args.canary and audited:
                    break
        if args.canary:
            print('CANARY COMPLETE', index, flush=True)
            return
        with (root/'.merge.lock').open('a') as merge:
            fcntl.flock(merge, fcntl.LOCK_EX)
            ids = {r['id'] for r in data['rows']}
            if all({p.stem for p in (root/c).glob('*.json')} == ids for c in CHANNELS):
                atomic_json(root/'complete.json', {'identity': root.name, 'full_dataset_complete': True, 'n': 451})
                print('FULL COMPLETE', flush=True)
            else:
                print('SHARD COMPLETE; merge pending', flush=True)


if __name__ == '__main__':
    main()
