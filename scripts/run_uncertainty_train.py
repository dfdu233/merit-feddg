"""Fixed TRAIN mechanism pilot; native source entropy and independent visual judge.

No reference is loaded by preparation/generation. Independent new experiment,
never mutates the earlier fixed-alpha runs or their frozen scorer.
"""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

from run_soft_guidance_full import sha, output
from run_channel_soft_canary import visible
from run_class_text_guidance import cad_decode
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.experts.native_xrv import XrvCapabilityAdapter
from merit_feddg.generalist import QwenLayerProbe
from merit_feddg.generalist_factory import load_generalist, generalist_provenance
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.persistent_scores import PersistentScores
from merit_feddg.soft_guidance import decode
from merit_feddg.uncertainty_decode import binary_uncertainty, decode_adaptive
from merit_feddg.uncertainty_experiment import select_cases, judge_prompt, swap_decision

BASE = Path('/home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6')
MANIFEST = Path('/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl')
XRV = Path('/home/dbw/merit-feddg/artifacts/models/xrv/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt')
XRV_SHA = '56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899'
JUDGE = Path('/home/dbw/merit-feddg/artifacts/models/OpenMed--Qwen2.5-3B-MedVL')
GPUS = ('GPU-3846413a-4238-d307-b1f3-10c2dfbe002c', 'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023')


def gpu_check(uuid):
    actual = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    if actual != uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('unauthorized GPU mapping')
    torch.set_num_threads(4)
    torch.ones(1, device='cuda:0').sum().item()


def resources():
    protocol = json.loads((BASE/'protocol.json').read_text())
    rows = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    old = json.loads((BASE/'compact_rows.json').read_text())
    if not protocol['shards_complete'] or protocol['n'] != len(rows) or set(old) != {r['id'] for r in rows}:
        raise RuntimeError('complete matching TRAIN incumbent required')
    return protocol, rows, old


def source_fingerprint():
    names = ('merit_feddg/uncertainty_decode.py', 'merit_feddg/uncertainty_experiment.py',
        'merit_feddg/persistent_scores.py', 'merit_feddg/llava_generalist.py',
        'merit_feddg/generalist.py', 'scripts/run_uncertainty_train.py')
    return {name: sha(name) for name in names}


def prepare(root, uuid):
    if root.exists():
        raise RuntimeError('prepare requires a fresh output directory')
    protocol, rows, old = resources()
    selected = select_cases(rows, old, per_label=4)
    from evaluate_soft_guidance_full import scorer_hashes
    config = {'schema': 'native-uncertainty-train-v1', 'manifest': str(MANIFEST),
        'manifest_sha256': sha(MANIFEST), 'base_run': str(BASE), 'base_sha256': sha(BASE/'compact_rows.json'),
        'protocol_sha256': sha(BASE/'protocol.json'), 'selected': selected, 'selection':
        'first four distinct TRAIN image hashes per explicit native attribute; full manifest retained',
        'source': source_fingerprint(), 'scorer': scorer_hashes(),
        'actor': generalist_provenance(protocol['config']['generalist'], 'artifacts'),
        'xrv_sha256': XRV_SHA, 'xrv_native_labels': 'official densenet121-res224-all; independent sigmoid; op_threshs=None',
        'judge_model': str(JUDGE), 'judge_config_sha256': sha(JUDGE/'config.json'),
        'judge_weights_index_sha256': sha(JUDGE/'model.safetensors.index.json'),
        'judge_weights_files': {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                for p in JUDGE.glob('*.safetensors')},
        'judge_training_overlap': 'not independently ruled out; SynthVision medical VQA fine-tune',
        'primary_candidate': 'source_acd', 'verifier': 'swap-consistent replacement; otherwise retain original incumbent',
        'arms': ['incumbent', 'unscoped_text', 'scope_text', 'fixed_blend', 'fixed_cad',
                 'acd', 'source_only', 'source_acd', 'source_constant', 'source_shuffled'],
        'uncertainty': 'binary entropy of question-matched raw sigmoid / ln(2); NOT calibrated correctness',
        'ablations': 'constant mean (1-U) and cyclic shuffled U within the same native label; TRAIN source outputs only',
        'max_new_tokens': 64, 'fit': False, 'test_used': False,
        'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'threads': 4}}
    config = json.loads(json.dumps(config))
    gpu_check(uuid)
    start = time.perf_counter()
    expert = XrvCapabilityAdapter(XRV, 'classification', device='cuda', sha256=XRV_SHA)
    load_seconds = time.perf_counter()-start
    by_id = {r['id']: r for r in rows}
    source = {}
    for selection in selected:
        key, label = selection['id'], selection['applicability']['label']
        start = time.perf_counter()
        labels, scores, transform = expert.classify(by_id[key]['image'])
        p = float(scores[list(labels).index(label)])
        source[key] = {'label': label, 'probability': p, 'uncertainty': binary_uncertainty(p),
            'labels': list(labels), 'scores': scores.tolist(), 'transform': transform,
            'seconds': time.perf_counter()-start, 'source_calls': 1}
    for label in {v['label'] for v in source.values()}:
        keys = [k for k in source if source[k]['label'] == label]
        mean = statistics.mean(1-source[k]['uncertainty'] for k in keys)
        for i, key in enumerate(keys):
            source[key]['constant_weight'] = mean
            source[key]['shuffled_u'] = source[keys[(i+1) % len(keys)]]['uncertainty']
            source[key]['shuffled_from'] = keys[(i+1) % len(keys)]
    root.mkdir(parents=True)
    (root/'cases').mkdir()
    atomic_json(root/'frozen.json', config)
    atomic_json(root/'sources.json', {'identity': fingerprint(config), 'gpu_uuid': uuid,
        'load_seconds': load_seconds, 'cases': source})
    print('PREPARED', root, fingerprint(config), 'cases', len(source), flush=True)


def evidence(key, source, focused):
    values = [(source['label'], source['probability'])] if focused else zip(source['labels'], source['scores'])
    return EvidenceItem('xrv-native:'+key, 'xrv_native_uncertainty', 'classification', 'cxr_findings',
        {'findings': [{'finding': label, 'score': float(score)} for label, score in values if label],
         'score_semantics': 'uncalibrated_independent_sigmoid', 'unlisted_findings': 'unknown',
         'query_diagnosis': 'not_established', 'image_transform': source['transform']},
        'Unverified chest radiograph finding scores; low scores do not exclude disease.',
        provenance={'adapter': 'xrv_classification', 'target_answers_used': False,
                    'question_attribute_focused': focused})


def context(probe, row, protocol, cfg, items):
    session = NativeSession(probe, row['image'], generation_prompt(row, protocol['config']), row['question'], cfg)
    image, prompt = session.context(NativeState(items=items))
    return PersistentScores(probe.new_answer_session(image, prompt)), session.last_transport


def real_case(probe, judge, row, old, protocol, source, key):
    started = time.perf_counter()
    cfg = ValueGenerationConfig(**old['generation_config'])
    if cfg.max_new_tokens != 64 or cfg.evidence_style != 'semantic' or cfg.vector_gate != 'off':
        raise RuntimeError('unexpected frozen generation configuration')
    native = tuple(EvidenceItem(**e) for e in old['evidence'])
    initial, transport = context(probe, row, protocol, cfg, native)
    original_ids = visible(transport)
    kept = tuple(e for e in native if (e.expert_id, e.evidence_id) in original_ids)
    base, base_transport = context(probe, row, protocol, cfg, kept)
    if visible(base_transport) != original_ids:
        raise RuntimeError('inherited evidence changed')
    focused, focused_transport = context(probe, row, protocol, cfg, kept+(evidence(key, source, True),))
    unscoped, unscoped_transport = context(probe, row, protocol, cfg, kept+(evidence(key, source, False),))
    expected = original_ids | {('xrv_native_uncertainty', 'xrv-native:'+key)}
    if visible(focused_transport) != expected or visible(unscoped_transport) != expected:
        raise RuntimeError('new evidence not delivered or inherited evidence displaced')
    eos = probe.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    arms = {}
    def run(mode, **kw):
        return decode_adaptive(base, focused, max_tokens=64, eos_ids=eos_ids, mode=mode, **kw)
    arms['incumbent'] = run('fixed', fixed_weight=0)
    start = time.perf_counter()
    direct = initial.propose((), count=1, length=64)[0]
    if list(direct.tokens) != arms['incumbent']['token_ids']:
        raise RuntimeError('incumbent endpoint parity failed')
    verification_seconds = time.perf_counter()-start
    arms['scope_text'] = run('fixed', fixed_weight=1)
    start = time.perf_counter()
    arms['unscoped_text'] = output(unscoped.propose((), count=1, length=64)[0], time.perf_counter()-start)
    arms['fixed_blend'] = run('fixed', fixed_weight=.5)
    start = time.perf_counter()
    legacy_blend = decode(base, focused, alpha=.5, max_tokens=64, eos_ids=eos_ids)
    verification_seconds += time.perf_counter()-start
    arms['fixed_cad'] = run('fixed', fixed_weight=1.5)
    start = time.perf_counter()
    legacy_cad = cad_decode(base, focused, 64, eos_ids)
    verification_seconds += time.perf_counter()-start
    if any(arms[a]['token_ids'] != ref['token_ids'] for a, ref in
           [('fixed_blend', legacy_blend), ('fixed_cad', legacy_cad)]):
        raise RuntimeError('reference decoder parity failed')
    for mode in ('acd', 'source_only', 'source_acd'):
        arms[mode] = run(mode, source_u=source['uncertainty'])
    arms['source_constant'] = run('fixed', fixed_weight=source['constant_weight'])
    arms['source_shuffled'] = run('source_only', source_u=source['shuffled_u'])
    if any(not a['text'].strip() for a in arms.values()):
        raise RuntimeError('empty candidate')
    incumbent, candidate = arms['incumbent']['text'], arms['source_acd']['text']
    verdicts = {}
    for kind in ('cross_visual', 'cross_blank', 'self_visual'):
        calls = []
        for a,b in ((incumbent,candidate),(candidate,incumbent)):
            prompt = judge_prompt(row['question'], a, b)
            start = time.perf_counter()
            if kind == 'self_visual':
                block = probe.new_answer_session(row['image'], prompt).propose((), count=1, length=16)[0]
                response = {'text': block.text, 'output_tokens': len(block.tokens)}
            else:
                image = row['image'] if kind == 'cross_visual' else Image.new('RGB', Image.open(row['image']).size, (127,127,127))
                response = judge.generate_with_usage(image, prompt, max_new_tokens=16,
                                                     allowed_texts=['A','B','TIE','UNCERTAIN'])
            calls.append({**response, 'seconds': time.perf_counter()-start})
        decision = swap_decision(calls[0]['text'], calls[1]['text'])
        verdicts[kind] = {**decision, 'calls': calls, 'selected_arm': 'source_acd' if decision['replace'] else 'incumbent',
                         'independent_correctness_guarantee': False}
    return {'id': key, 'arms': arms, 'verifiers': verdicts, 'source': source,
            'transport': {'incumbent': base_transport, 'scope_text': focused_transport, 'unscoped_text': unscoped_transport},
            'historical_token_parity': arms['incumbent']['token_ids'] == old['token_ids'],
            'old_incumbent_seconds_inherited': old['seconds'],
            'verification_seconds': verification_seconds, 'wall_seconds': time.perf_counter()-started}


def generate(root, shard, uuid, max_cases=0):
    frozen = json.loads((root/'frozen.json').read_text())
    if source_fingerprint() != frozen['source'] or sha(MANIFEST) != frozen['manifest_sha256'] or sha(BASE/'compact_rows.json') != frozen['base_sha256']:
        raise RuntimeError('frozen experiment changed')
    protocol, rows, old = resources()
    sources = json.loads((root/'sources.json').read_text())
    identity = fingerprint(frozen)
    if sources['identity'] != identity:
        raise RuntimeError('source identity mismatch')
    with (root/f'.shard-{shard}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_check(uuid)
        start = time.perf_counter()
        probe = load_generalist(protocol['config']['generalist'], 'artifacts')
        probe.model.eval().requires_grad_(False)
        judge = QwenLayerProbe(str(JUDGE), layers=[-1], dtype='bfloat16', device_map='cuda')
        judge.model.requires_grad_(False)
        atomic_json(root/f'load-{shard}-{time.time_ns()}.json', {'seconds': time.perf_counter()-start, 'gpu_uuid':uuid})
        lookup = {r['id']:r for r in rows}
        completed = 0
        for index, selection in enumerate(frozen['selected']):
            if index % 2 != shard:
                continue
            key = selection['id']; path = root/'cases'/(key+'.json')
            if path.exists():
                record = json.loads(path.read_text())
                if record['identity'] != identity or record['id'] != key:
                    raise RuntimeError('invalid resumed result')
                continue
            with torch.inference_mode():
                result = real_case(probe, judge, lookup[key], old[key], protocol, sources['cases'][key], key)
            atomic_json(path, {'identity': identity, 'gpu_uuid': uuid, 'selection': selection, **result})
            print('DONE', key, 'seconds', round(result['wall_seconds'],2), 'u', sources['cases'][key]['uncertainty'], flush=True)
            completed += 1
            if max_cases and completed >= max_cases:
                print('SCHEDULING STOP; same frozen pilot', flush=True)
                return
        print('SHARD COMPLETE', shard, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare','generate'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, choices=(0,1), default=0)
    p.add_argument('--gpu-uuid', choices=GPUS, required=True)
    p.add_argument('--max-cases', type=int, default=0)
    args = p.parse_args()
    os.umask(0o002)
    if args.stage == 'prepare': prepare(args.output, args.gpu_uuid)
    else: generate(args.output, args.shard_index, args.gpu_uuid, args.max_cases)


if __name__ == '__main__': main()
