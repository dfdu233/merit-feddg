"""Opt-in experiment on a completed semantic/compact run; legacy entrypoints unchanged.

Run --check-only first. Live execution requires the existing local model assets.
No trained controller, new downloads, reference labels, or test-selected thresholds.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter

import yaml

from .agent_protocol import (
    atomic_json, file_hash, load_incumbent, merge_shards, read_inputs, read_sources,
)
from .agent_runtime import augment_case
from .evidence_agent import ToolUnavailable, commit_or_keep, digest

MODES = {'graph_static': 'static', 'region_control': 'opposite_control',
         'full_image_control': 'full_image_control', 'agent_candidate': 'agent'}
METHODS = ['incumbent', 'graph_noop', *MODES, 'agent_committed']
DEFAULTS = {'region_limit': 2, 'max_steps': 6, 'observation_tokens': 48,
            'planner_tokens': 48, 'padding': 0.05, 'commit_policy': 'audit_only'}


def load_options(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('unknown evidence-agent option')
    options = {**DEFAULTS, **raw}
    if options['commit_policy'] not in {'audit_only', 'external_compare'}:
        raise ValueError('commit_policy must be audit_only or external_compare')
    for key in ('region_limit', 'max_steps', 'observation_tokens', 'planner_tokens'):
        if type(options[key]) is not int or not 1 <= options[key] <= 1024:
            raise ValueError(f'invalid budget: {key}')
    if type(options['padding']) not in (int, float) or not 0 <= options['padding'] <= .25:
        raise ValueError('padding must be a fixed value in [0,.25]')
    return options


def prepare(args):
    options = load_options(args.config)
    rows = read_inputs(args.manifest)
    protocol, incumbent = load_incumbent(args.base_run, args.incumbent, rows)
    sources, references, source_audit = read_sources(args.source_manifest, rows)
    source = Path(__file__).parent
    # Bind ALL Python source, base config/output bytes and actual image hashes.
    code = {p.relative_to(source).as_posix(): file_hash(p) for p in sorted(source.rglob('*.py'))}
    identity = digest({'schema': 'evidence-agent-v1', 'options': options,
                       'manifest': rows, 'base_protocol': protocol,
                       'base_output_sha256': file_hash(Path(args.base_run) / f'{args.incumbent}.json'),
                       'source_audit': source_audit, 'implementation': code})
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('invalid shard index/count')
    if args.shard_count > len(rows):
        raise ValueError('more shards than samples')
    cfg = protocol['config']
    if sources and 'source_cases' not in cfg.get('experts', {}):
        raise ValueError('source manifest supplied without a configured source_cases adapter')
    return options, rows, protocol, incumbent, sources, references, source_audit, identity


def live(args, prepared, root):
    # Do not import torch/transformers in --check-only or --merge-only.
    import torch
    import fcntl
    from dataclasses import asdict

    from .capabilities import CapabilityRequest, EvidenceItem
    from .capability_experts import CapabilityPool
    from .capability_runtime import CapabilityRuntime, NativeSession, NativeState, ValueGenerationConfig
    from .generalist_factory import generalist_provenance, load_generalist
    from .matched_evaluation import generation_prompt
    from .open_data import INFERENCE_FIELDS
    from .open_study import model_provenance

    options, rows, protocol, base, sources, references, _, identity = prepared
    config = protocol['config']
    spec = config['generalist']
    if spec.get('tensor_bridge_checkpoint'):
        raise ValueError('a trained bridge is outside this experiment')
    model_id = {'generalist': generalist_provenance(spec, args.artifacts)}
    if sources:
        model_id['retriever'] = model_provenance(config['experts']['source_cases'], args.artifacts)
    if options['commit_policy'] == 'external_compare':
        model_id['verifiers'] = {k: model_provenance(v, args.artifacts)
                                 for k, v in config.get('answer_verifiers', {}).items()}
    model_file = root / 'model_provenance.json'
    with (root / '.assets.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if model_file.exists() and json.loads(model_file.read_text()) != model_id:
            raise ValueError('local model/source assets changed since prior execution')
        atomic_json(model_file, model_id)
    probe = load_generalist(spec, args.artifacts)
    if hasattr(probe, 'model'):
        probe.model.eval().requires_grad_(False)
    specs = config.get('experts', {})
    pool = (CapabilityPool({'source_cases': specs['source_cases']}, args.artifacts,
                           source_records=sources, source_references=references) if sources else None)
    shard = root / 'shards' / str(args.shard_index)
    shard.mkdir(parents=True, exist_ok=True)
    cache = shard / 'case-cache'
    outputs = {name: {} for name in METHODS}
    verifier_cache = {}
    shard_rows = rows[args.shard_index::args.shard_count]
    try:
        for case_index, row in enumerate(shard_rows):
            if getattr(args, 'canary_cases', 0) and case_index >= args.canary_cases:
                break
            cache_path = cache / f"{digest(row['id'])}.json"
            if cache_path.exists():
                saved = json.loads(cache_path.read_text())
                if saved.get('identity') != identity or set(saved.get('outputs', {})) != set(METHODS):
                    raise ValueError('case cache identity/method mismatch')
                values = saved['outputs']
            else:
                if pool is not None:
                    pool.reset_case()
                original = base[row['id']]
                generation = ValueGenerationConfig(**original['generation_config'])
                prompt = generation_prompt(row, config)
                native = tuple(EvidenceItem(**v) for v in original['evidence'])
                modality = original.get('input_modality', row.get('modality'))
                if not modality or modality == 'mixed':
                    raise ValueError('base answer must record its actual routed modality')
                runtime_row = {
                    'id': row['id'], 'image': row['image'], 'question': row['question'],
                    'modality': modality, 'capability': 'classification',
                    'task': row.get('task', 'open_vqa'), 'domain': row.get('domain', 'agent-query'),
                    'domain_kind': row.get('domain_kind', 'declared_manifest'),
                    'role': row.get('role', 'target'),
                    'group_id': row.get('group_id', row['image_sha256']),
                    'image_sha256': row['image_sha256'],
                }
                assert set(runtime_row) == INFERENCE_FIELDS
                for event in original.get('trace', []):
                    req = event.get('request', {})
                    if req.get('question') is not None and req['question'] != row['question']:
                        raise ValueError('question differs from base tool request')

                def render(items):
                    session = NativeSession(probe, row['image'], prompt, row['question'], generation)
                    engine = CapabilityRuntime(session, None, runtime_row, specs, generation)
                    with torch.inference_mode():
                        answer = engine.complete(NativeState(items=items))
                    answer.update(evidence=[asdict(i) for i in items], trace=[])
                    return answer

                noop_started = perf_counter()
                noop = render(native)
                noop_seconds = perf_counter() - noop_started
                if (noop['text'] != original['text'] or noop['token_ids'] != original['token_ids']):
                    atomic_json(shard / f"parity_failure_{digest(row['id'])}.json",
                                {'id': row['id'], 'expected': original, 'actual': noop})
                    raise RuntimeError('no-op text/token parity failed; stop before running new arms')
                before = {(v['expert_id'], v['evidence_id'])
                          for v in noop.get('evidence_transport', {}).get('presented', [])}

                def synthesize(extras):
                    added = tuple(EvidenceItem(**value) for value in extras)
                    trial = NativeSession(probe, row['image'], prompt, row['question'], generation)
                    trial.context(NativeState(items=native + added))
                    after = {(v['expert_id'], v['evidence_id'])
                             for v in trial.last_transport.get('presented', [])}
                    extra_ids = {(v.expert_id, v.evidence_id) for v in added}
                    if not before.issubset(after) or not extra_ids.issubset(after):
                        return None  # no silent eviction or partial delivery
                    return render(native + added)

                def generate(path, text, tokens, allowed):
                    if not probe.context_token_budget(path, text, tokens)['fits']:
                        raise ToolUnavailable('intact context budget exceeded')
                    with torch.inference_mode():
                        return probe.generate_with_usage(path, text, max_new_tokens=tokens,
                                                         allowed_texts=allowed)

                def retrieve(path):
                    request = CapabilityRequest(
                        sample_id=row['id'], image=path, question=row['question'],
                        modality=modality, task=runtime_row['task'], domain=runtime_row['domain'],
                        group_id=runtime_row['group_id'], capability='retrieval',
                        scope=specs['source_cases']['scope'], query=row['question'],
                    )
                    with torch.inference_mode():
                        result = pool.infer('source_cases', request)
                    if not result.items:
                        raise ToolUnavailable(result.reason)
                    return {'items': [asdict(item) for item in result.items],
                            'original_query_pixel_digest': row['_pixel_sha256'],
                            'retrieval_scope': 'source_analogies_only'}

                values = {'incumbent': copy.deepcopy(original), 'graph_noop': noop}
                workflows = {}
                for arm, mode in MODES.items():
                    work = augment_case(
                        case_id=row['id'], image=row['image'], question=row['question'],
                        evidence=original['evidence'],
                        source_models={key: digest([value['id'], value.get('revision')])
                                       for key, value in specs.items()},
                        output_dir=str(shard / 'crops' / digest(row['id']) / arm),
                        generate=generate, synthesize=synthesize,
                        retrieve=retrieve if pool is not None else None, mode=mode,
                        **{k: v for k, v in options.items() if k != 'commit_policy'},
                    )
                    workflows[arm] = work
                    result = copy.deepcopy(work['candidate'] or original)
                    result['agent_workflow'] = work
                    # Report compute actually incurred AND inherited incumbent compute.
                    result['new_seconds'] = work['seconds']
                    result['seconds'] = original['seconds'] + work['seconds']
                    result['shared_setup_audit_seconds'] = noop_seconds
                    values[arm] = result
                candidate = workflows['agent_candidate']['candidate']
                decision, reason, judge = 'unknown', 'audit_only', None
                then = perf_counter()
                if candidate is None:
                    reason = 'no_usable_candidate'
                elif options['commit_policy'] == 'external_compare':
                    from .answer_arbitration import assess_revision, load_verifier
                    eligible = [(name, v) for name, v in config.get('answer_verifiers', {}).items()
                                if modality in v['modalities']]
                    if len(eligible) > 1:
                        raise ValueError('declare at most one external verifier per modality')
                    verifier = None
                    if eligible:
                        name, definition = eligible[0]
                        if name not in verifier_cache:
                            verifier_cache[name] = load_verifier(definition, args.artifacts)
                        verifier = verifier_cache[name]
                    judge = assess_revision(
                        original['text'], candidate['text'], image=row['image'],
                        question=row['question'], modality=modality, verifier=verifier,
                        generalist_id=spec['id'],
                        source_model_ids=tuple(specs[v['expert_id']]['id'] for v in original['evidence'])
                        + ((specs['source_cases']['id'],) if pool is not None else ()),
                    )
                    decision = {'accept': 'accept', 'unchanged': 'accept', 'reject': 'reject',
                                'abstain': 'unknown'}[judge['decision']]
                    reason = judge['reason']
                committed = commit_or_keep(original, candidate, decision=decision, reason=reason)
                committed['gate_audit'] = judge
                committed['gate_seconds'] = perf_counter() - then
                committed['seconds'] = values['agent_candidate']['seconds'] + committed['gate_seconds']
                committed['candidate_audit_arm'] = 'agent_candidate'
                values['agent_committed'] = committed
                atomic_json(cache_path, {'identity': identity, 'outputs': values})
            for name in METHODS:
                outputs[name][row['id']] = values[name]
            print(f"completed {row['id']}", flush=True)
        complete = len(outputs['incumbent']) == len(shard_rows)
        atomic_json(shard / 'results.json', {
            'identity': identity, 'shard_count': args.shard_count, 'shard_index': args.shard_index,
            'complete': complete, 'outputs': outputs,
            'model_provenance_sha256': digest(model_id),
        })
    finally:
        if pool is not None:
            pool.clear()
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evidence_agent.yaml')
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--incumbent', default='compact_rows',
                        choices=['semantic_all', 'compact_rows', 'compact_all'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--canary-cases', type=int, default=0,
                        help='Scheduling-only cap; partial runs NEVER publish aggregate results')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    if args.canary_cases < 0 or (args.canary_cases and args.merge_only):
        parser.error('canary-cases must be nonnegative and is incompatible with merge-only')
    if args.check_only and args.merge_only:
        parser.error('check-only and merge-only are mutually exclusive')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    prepared = prepare(args)
    options, rows, base, _, _, _, source_audit, identity = prepared
    root = Path(args.output) / identity
    preflight = {'identity': identity, 'n': len(rows), 'methods': METHODS,
                 'base_run': str(Path(args.base_run).resolve()), 'base_identity': base['identity'],
                 'incumbent_method': args.incumbent, 'options': options,
                 'source_audit': source_audit, 'output_root': str(root),
                 'weights_fitted': False, 'calibration_fitted': False,
                 'langgraph_required': False, 'gpu_validated_by_preflight': False}
    atomic_json(root / 'preflight.json', preflight)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if args.check_only:
        return
    if not args.merge_only:
        import fcntl
        with (root / f'.shard-{args.shard_index}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('this shard already has an active worker') from exc
            complete = live(args, prepared, root)
        if not complete:
            print('Partial canary saved. No aggregate result; resume without --canary-cases.')
            return
    if args.merge_only or args.shard_count == 1:
        merge_shards(root, identity, [r['id'] for r in rows], METHODS, args.shard_count)
        atomic_json(root / 'protocol.json', {**preflight, 'shards_complete': True,
                                            'shard_count': args.shard_count})


if __name__ == '__main__':
    main()
