"""Replay the frozen BARD protocol from native caches without loading specialists."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from merit_feddg.bard import BARDConfig, _decision_from_residuals, _prepare
from merit_feddg.bard_protocol import (
    acquire_expert_groups,
    build_isolated_sessions,
    run_bard_bundle,
)
from merit_feddg.capability_runtime import CapabilityRuntime, NativeSession, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import (
    SharedExpertPool,
    experiment_arms,
    generation_prompt,
    load_manifest,
)
from merit_feddg.open_study import atomic_json, fingerprint


def stress_audit(session, acquisition, bard_config, tokens):
    """Shadow comparisons on the same clean BARD prefixes; never change generation."""
    config = BARDConfig(**bard_config)
    base, experts, transport = build_isolated_sessions(session, acquisition['groups'])
    names = list(experts)
    policies = [('isolated_mean', 'mean', False),
                ('isolated_geomedian', 'geometric_median', False),
                ('bard', 'geometric_median', True)]
    records = []
    for step in range(min(8, len(tokens))):
        prefix = tokens[:step]
        b, mask, residuals = _prepare(base.next_scores(prefix),
                                     [experts[n].next_scores(prefix) for n in names])
        for method, aggregation, bounded in policies:
            def decide(matrix, b=b, mask=mask, aggregation=aggregation, bounded=bounded):
                return _decision_from_residuals(b, mask, matrix, config,
                    aggregation=aggregation, bounded_commit=bounded)
            clean, clean_audit = decide(residuals)
            faults = []
            for i, name in enumerate(names):
                corrupt = residuals.copy()
                if np.linalg.norm(corrupt[i]) > 0:
                    corrupt[i] *= -config.fault_probe_scale
                else:
                    order = np.argsort(b[mask]); source = int(order[-1])
                    target = int(order[-2]) if len(order) > 1 else source
                    magnitude = config.fault_probe_scale + abs(float(b[mask][source] - b[mask][target]))
                    corrupt[i, target] = magnitude
                    corrupt[i, source] = -magnitude
                token, audit = decide(corrupt)
                faults.append({'node': name, 'same_as_clean': token == clean,
                                   'fallback_to_base': token == clean_audit['base_token'],
                                   'selected_token': token, 'reason': audit['reason']})
            without_retrieval = None
            if method == 'bard' and 'medcpt_pubmed' in names:
                keep = [i for i, n in enumerate(names) if n != 'medcpt_pubmed']
                without_retrieval, _ = decide(residuals[keep])
            records.append({'step': step, 'method': method, 'clean_token': clean,
                                'base_token': clean_audit['base_token'], 'faults': faults,
                                'without_retrieval_token': without_retrieval})
    return {'scope': 'first eight clean BARD prefixes; token-level residual stress, not corrupted free generation',
                'expert_branches': names, 'transport': transport, 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--case-id', action='append', help='Restrict scheduling only; validate the full frozen manifest')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-forwards', type=int, default=200000)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, default=4)
    args = parser.parse_args()
    cache = Path(args.cache)
    protocol = json.loads((cache / 'protocol.json').read_text())
    assert protocol['shards_complete'] and protocol['cache_only']
    config = load_experiment_yaml(args.config)
    if config['experts'] != protocol['config']['experts']:
        raise ValueError('Native expert specifications differ from frozen donor cache')
    specs = {k: v for k, v in config['experts'].items()
             if k != 'source_cases' and k not in protocol['excluded']}
    routes = json.loads((cache / 'routing.json').read_text())
    original = load_manifest(args.manifest)
    assert set(routes) == {r['id'] for r in original}
    decoder = ValueGenerationConfig(**config['capability_value']['generation'])
    arms = experiment_arms(decoder, 'bard')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    probe = load_generalist(config['generalist'])
    count = [0]
    def count_forward(_module, _args):
        count[0] += 1
        if count[0] > args.max_forwards:
            raise RuntimeError('Canary worker forward ceiling reached')
    hook = probe.model.register_forward_pre_hook(count_forward)
    try:
        for index, original_row in enumerate(original):
            if index % args.shard_count != args.shard_index:
                continue
            if args.case_id and original_row['id'] not in args.case_id:
                continue
            assert hashlib.sha256(Path(original_row['image']).read_bytes()).hexdigest() == original_row['image_sha256']
            row = {k: original_row[k] for k in ('id', 'image', 'question', 'image_sha256')}
            row.update(modality=routes[row['id']]['modality'], capability='classification',
                       task=original_row.get('task', 'open_vqa'), domain=original_row['domain'],
                       domain_kind='official_dataset_split', role='source', group_id=row['image_sha256'])
            pool = SharedExpertPool(None, cache / 'expert-cache' / fingerprint(row['id']), protocol['identity'])
            prompt = generation_prompt(original_row, config)
            session = NativeSession(probe, row['image'], prompt, row['question'], decoder)
            engine = CapabilityRuntime(session, pool, row, specs, decoder, None)
            before = count[0]
            outputs = {}
            for method in ('generalist', 'joint_all'):
                arm = arms[method]
                arm_session = NativeSession(probe, row['image'], prompt, row['question'], arm)
                runtime = CapabilityRuntime(arm_session, pool, row, specs, arm, None)
                outputs[method] = runtime.run('generalist' if method == 'generalist' else 'all_evidence')
                atomic_json(out / row['id'] / f'{method}.json', outputs[method])
            acquisition = acquire_expert_groups(engine)
            outputs.update(run_bard_bundle(session, acquisition, config.get('bard', {})))
            for method in ('isolated_mean', 'isolated_geomedian', 'bard'):
                atomic_json(out / row['id'] / f'{method}.json', outputs[method])
            atomic_json(out / row['id'] / 'stress.json', stress_audit(
                session, acquisition, config.get('bard', {}), outputs['bard']['token_ids']))
            atomic_json(out / row['id'] / 'provenance.json', {
                'cache_identity': protocol['identity'], 'row': row,
                'receiver_config': config, 'prompt': prompt,
                'routing_source': 'frozen LLaVA image-only routes shared across receivers',
                'protocol_sha256': hashlib.sha256((cache / 'protocol.json').read_bytes()).hexdigest(),
                'native_cache_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (cache / 'expert-cache' / fingerprint(row['id'])).glob('*.json')},
                'model_forwards': count[0] - before, 'torch_version': probe.torch.__version__,
                'device': probe.torch.cuda.get_device_name(0),
                'acquisition_events': acquisition['events'],
                'fault_group_members': acquisition['fault_group_members'],
                'answers_loaded': False,
            })
            print('COMPLETE', index, row['id'], 'forwards', count[0] - before,
                  {m: v['text'] for m, v in outputs.items()}, flush=True)
    finally:
        atomic_json(out / f'worker-{args.shard_index}-cost.json', {'model_forwards': count[0]})
        hook.remove()


if __name__ == '__main__':
    main()
