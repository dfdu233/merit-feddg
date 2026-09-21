"""Replay the frozen BARD protocol from native caches without loading specialists."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from merit_feddg.bard import BARDConfig, _decision_from_residuals, _prepare
from merit_feddg.bard_protocol import (
    acquire_expert_groups,
    build_isolated_sessions,
    run_bard_bundle,
    run_bard_method,
)
from merit_feddg.capability_runtime import CapabilityRuntime, NativeSession, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import (
    SharedExpertPool,
    experiment_arms,
    generation_prompt,
    load_cached,
    load_manifest,
)
from merit_feddg.open_study import atomic_json, fingerprint


def native_evidence_references(directory, cache):
    """Index immutable native items already retained in the frozen cache."""
    index = {}
    for source in sorted(directory.glob('*.json')):
        raw = source.read_bytes()
        record = json.loads(raw)
        for position, item in enumerate(record.get('output', {}).get('items', [])):
            key = item.get('evidence_id')
            if key is None:
                continue
            index.setdefault(key, []).append((item, {
                'cache_identity': record['identity'],
                'relative_path': str(source.relative_to(cache)),
                'file_sha256': hashlib.sha256(raw).hexdigest(),
                'item_index': position,
            }))
    return index


def reference_native_evidence(value, index):
    """Replace only exact native-item duplicates with content-verified references."""
    if isinstance(value, dict):
        for original, reference in index.get(value.get('evidence_id'), []):
            if value == original:
                return {'__native_evidence_ref_v1__': reference}
        return {key: reference_native_evidence(item, index) for key, item in value.items()}
    if isinstance(value, list):
        return [reference_native_evidence(item, index) for item in value]
    return value


def restore_native_evidence(value, cache, loaded=None):
    """Losslessly expand stored references, rejecting mismatched cache bytes."""
    loaded = {} if loaded is None else loaded
    if isinstance(value, dict):
        if set(value) == {'__native_evidence_ref_v1__'}:
            ref = value['__native_evidence_ref_v1__']
            path = (cache / ref['relative_path']).resolve()
            if not path.is_relative_to(cache.resolve()):
                raise ValueError('Native evidence reference escapes cache')
            if path not in loaded:
                raw = path.read_bytes()
                loaded[path] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
            digest, record = loaded[path]
            if digest != ref['file_sha256'] or record['identity'] != ref['cache_identity']:
                raise ValueError('Native evidence reference provenance mismatch')
            return record['output']['items'][ref['item_index']]
        return {key: restore_native_evidence(item, cache, loaded) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_native_evidence(item, cache, loaded) for item in value]
    return value


def write_case_artifact(path, value, *, compressed=False, native_index=None):
    """Keep complete evidence, optionally using lossless gzip on disk."""
    if native_index is not None:
        value = reference_native_evidence(value, native_index)
    if not compressed:
        atomic_json(path, value)
        return
    path = Path(str(path) + '.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


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
    parser.add_argument('--methods', nargs='+', choices=['generalist', 'joint_all', 'isolated_mean', 'isolated_geomedian', 'bard'], default=['generalist', 'joint_all', 'isolated_mean', 'isolated_geomedian', 'bard'])
    parser.add_argument('--skip-stress', action='store_true')
    parser.add_argument('--prevent-empty-eos', action='store_true', help='Separate repair protocol: reject selected EOS only while decoded output is blank')
    parser.add_argument('--compress-artifacts', action='store_true',
                        help='Losslessly gzip complete per-case JSON artifacts; no evidence is removed')
    parser.add_argument('--reference-native-evidence', action='store_true',
                        help='Store exact native evidence duplicates as SHA-verified references; retain native cache')
    parser.add_argument('--max-forwards', type=int, default=200000)
    parser.add_argument('--cached-receiver', action='store_true', help='Use separately parity-validated native receiver KV scores')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, default=4)
    args = parser.parse_args()
    if args.prevent_empty_eos and (not args.cached_receiver or args.methods != ['bard'] or not args.skip_stress):
        raise ValueError('Empty-EOS repair requires cached receiver, bard only, and skip-stress')
    cache = Path(args.cache)
    protocol = json.loads((cache / 'protocol.json').read_text())
    assert protocol['cache_only']
    partial_cache = not protocol['shards_complete'] and args.methods != ['generalist']
    if partial_cache and not (args.case_id or args.end_index is not None):
        raise ValueError('Incomplete cache requires an explicit bounded scheduling subset')
    config = load_experiment_yaml(args.config)
    if config['experts'] != protocol['config']['experts']:
        raise ValueError('Native expert specifications differ from frozen donor cache')
    specs = {k: v for k, v in config['experts'].items()
             if k != 'source_cases' and k not in protocol['excluded']}
    routes = json.loads((cache / 'routing.json').read_text())
    original = load_manifest(args.manifest)
    # Preserve the already-frozen, answer-blind benchmark prompt, including
    # multiple-choice options. Native expert requests still use the question.
    benchmark_prompts = {}
    for source in map(json.loads, Path(args.manifest).read_text().splitlines()):
        if 'benchmark_prompt' in source:
            prompt = source['benchmark_prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('Frozen benchmark prompt must be nonempty text')
            benchmark_prompts[source['id']] = prompt
    assert set(routes) == {r['id'] for r in original}
    decoder = ValueGenerationConfig(**config['capability_value']['generation'])
    arms = experiment_arms(decoder, 'bard')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    native_index = None
    def save_case(path, value):
        write_case_artifact(path, value, compressed=args.compress_artifacts, native_index=native_index)
    if partial_cache:
        from dataclasses import asdict

        from prepare_bard_expert_cache import make_request
        schedule = json.loads((cache / 'schedule.json').read_text())
        for index, source in enumerate(original):
            if index < args.start_index or (args.end_index is not None and index >= args.end_index):
                continue
            if index % args.shard_count != args.shard_index or (args.case_id and source['id'] not in args.case_id):
                continue
            check_row = dict(source, modality=routes[source['id']]['modality'], group_id=source['image_sha256'])
            for descriptor in schedule[source['id']]:
                request = make_request(check_row, descriptor, decoder)
                key = fingerprint(['infer', descriptor['expert'], asdict(request)])
                path = cache / 'expert-cache' / fingerprint(source['id']) / f'{key}.json'
                if load_cached(path, protocol['identity']) is None:
                    raise ValueError(f'Incomplete selected case {source["id"]}: {descriptor["expert"]}')
    probe = load_generalist(config['generalist'])
    if args.cached_receiver:
        from huatuo_cached_receiver import enable_cached_sessions
        enable_cached_sessions(probe)
    count = [0]
    def count_forward(_module, _args):
        count[0] += 1
        if count[0] > args.max_forwards:
            raise RuntimeError('Canary worker forward ceiling reached')
    hook = probe.model.register_forward_pre_hook(count_forward)
    try:
        for index, original_row in enumerate(original):
            if index < args.start_index or (args.end_index is not None and index >= args.end_index):
                continue
            if index % args.shard_count != args.shard_index:
                continue
            if args.case_id and original_row['id'] not in args.case_id:
                continue
            assert hashlib.sha256(Path(original_row['image']).read_bytes()).hexdigest() == original_row['image_sha256']
            row = {k: original_row[k] for k in ('id', 'image', 'question', 'image_sha256')}
            row.update(modality=routes[row['id']]['modality'], capability='classification',
                       task=original_row.get('task', 'open_vqa'), domain=original_row['domain'],
                       domain_kind='official_dataset_split', role=original_row.get('role', 'target'), group_id=row['image_sha256'])
            pool = SharedExpertPool(None, cache / 'expert-cache' / fingerprint(row['id']), protocol['identity'])
            native_index = (native_evidence_references(pool.directory, cache)
                            if args.reference_native_evidence else None)
            prompt = benchmark_prompts.get(row['id'], generation_prompt(original_row, config))
            session = NativeSession(probe, row['image'], prompt, row['question'], decoder)
            engine = CapabilityRuntime(session, pool, row, specs, decoder, None)
            before = count[0]
            outputs = {}
            for method in ('generalist', 'joint_all'):
                if method not in args.methods:
                    continue
                arm = arms[method]
                arm_session = NativeSession(probe, row['image'], prompt, row['question'], arm)
                runtime = CapabilityRuntime(arm_session, pool, row, specs, arm, None)
                outputs[method] = runtime.run('generalist' if method == 'generalist' else 'all_evidence')
                save_case(out / row['id'] / f'{method}.json', outputs[method])
            isolated = [m for m in ('bard', 'isolated_mean', 'isolated_geomedian') if m in args.methods]
            acquisition = acquire_expert_groups(engine) if isolated else {'groups': {}, 'events': [], 'fault_group_members': {}}
            if args.cached_receiver or not isolated:
                for method in isolated:
                    outputs[method] = run_bard_method(session, acquisition, config.get('bard', {}), method, fault_probe=not args.skip_stress, prevent_empty_eos=args.prevent_empty_eos)
                    save_case(out / row['id'] / f'{method}.json', outputs[method])
            else:
                outputs.update(run_bard_bundle(session, acquisition, config.get('bard', {})))
                for method in ('isolated_mean', 'isolated_geomedian', 'bard'):
                    save_case(out / row['id'] / f'{method}.json', outputs[method])
            if not args.skip_stress and 'bard' in outputs:
                save_case(out / row['id'] / 'stress.json', stress_audit(
                    session, acquisition, config.get('bard', {}), outputs['bard']['token_ids']))
            save_case(out / row['id'] / 'provenance.json', {
                'cache_identity': protocol['identity'], 'row': row,
                'receiver_config': config, 'prompt': prompt, 'methods': args.methods,
                'prompt_source': 'frozen_benchmark_prompt' if row['id'] in benchmark_prompts else 'generation_prompt',
                'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                'receiver_execution': 'parity_validated_kv' if args.cached_receiver else 'production_replay',
                'artifact_encoding': 'gzip-json' if args.compress_artifacts else 'json',
                'native_evidence_storage': ('sha256-verified-cache-references-v1'
                                            if args.reference_native_evidence else 'inline'),
                'routing_source': protocol['routing'],
                'protocol_sha256': hashlib.sha256((cache / 'protocol.json').read_bytes()).hexdigest(),
                'native_cache_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (cache / 'expert-cache' / fingerprint(row['id'])).glob('*.json')},
                'model_forwards': count[0] - before, 'torch_version': probe.torch.__version__,
                'device': probe.torch.cuda.get_device_name(0),
                'acquisition_events': acquisition['events'],
                'fault_group_members': acquisition['fault_group_members'],
                'answers_loaded': False,
                'empty_eos_repair': 'selected-eos-visible-continuation-v1' if args.prevent_empty_eos else None,
            })
            print('COMPLETE', index, row['id'], 'forwards', count[0] - before,
                  {m: v['text'] for m, v in outputs.items()}, flush=True)
    finally:
        atomic_json(out / f'worker-{args.shard_index}-cost.json', {'model_forwards': count[0]})
        hook.remove()


if __name__ == '__main__':
    main()
