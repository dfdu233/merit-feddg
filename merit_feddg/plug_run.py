"""Opt-in plug-observe experiment; every historical entrypoint remains unchanged.

A completed run supplies cached native evidence and routed image modality, NOT
reference answers. New arms all use the SAME free-generation prompt, including a
fresh legacy compact baseline. Historical CE/OE-conditioned scores are archived,
not compared as if they were the new baseline. No downloads or fitted policy.
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
    atomic_json,
    file_hash,
    load_incumbent,
    merge_shards,
    read_inputs,
    read_sources,
)
from .evidence_agent import ToolUnavailable, digest
from .plug_observe import ANSWER_SUFFIX, json_copy, run_case, validate_specs

METHODS = ['incumbent', 'plug_read', 'plug_static', 'plug_agent']
DEFAULTS = {'max_calls': 3, 'region_limit': 2, 'max_new_tokens': 64,
            'observation_tokens': 48, 'planner_tokens': 16, 'observation_view': 'region'}


def prepare(args):
    raw = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('unknown plug-observe configuration field')
    options = {**DEFAULTS, **raw}
    for key in ('region_limit', 'max_new_tokens', 'observation_tokens', 'planner_tokens'):
        if type(options[key]) is not int or not 1 <= options[key] <= 1024:
            raise ValueError(f'invalid integer budget: {key}')
    if type(options['max_calls']) is not int or not 0 <= options['max_calls'] <= 32:
        raise ValueError('max_calls must be 0..32')
    if options['observation_view'] not in {'region', 'whole'}:
        raise ValueError('observation_view must be region or whole')
    rows = read_inputs(args.manifest)
    base_protocol, base = load_incumbent(args.base_run, args.incumbent, rows)
    if not 0 <= args.shard_index < args.shard_count <= len(rows):
        raise ValueError('invalid sharding')
    sources, references, source_audit = read_sources(args.source_manifest, rows)
    specs = copy.deepcopy(base_protocol['config']['experts'])
    if args.expert_registry:
        extra = yaml.safe_load(Path(args.expert_registry).read_text(encoding='utf-8'))
        if not isinstance(extra, dict) or set(extra) != {'experts'}:
            raise ValueError('expert extension file must contain only experts')
        extra = validate_specs(extra['experts'])
        if set(extra) & set(specs):
            raise ValueError('extensions may add new expert IDs, not silently replace old checkpoints')
        specs.update(extra)
    excluded = {}
    for name, spec in list(specs.items()):
        if spec.get('enabled') is False:
            excluded[name] = 'explicitly_disabled'
        elif spec.get('adapter') == 'source_retrieval' and not sources:
            excluded[name] = 'no_source_manifest'
        elif (spec.get('optional') and spec.get('checkpoint_path')
              and not Path(spec['checkpoint_path']).expanduser().exists()):
            excluded[name] = 'optional_checkpoint_not_found'
        if name in excluded:
            del specs[name]
    validate_specs(specs)
    config = json_copy(base_protocol['config'])
    if config['generalist'].get('tensor_bridge_checkpoint'):
        raise ValueError('trained bridges are forbidden')
    # No target answer-type field crosses the case boundary.
    cases = []
    for row in rows:
        original = base[row['id']]
        for event in original.get('trace', []):
            request = event.get('request', {})
            if request.get('question') is not None and request['question'] != row['question']:
                raise ValueError('manifest question differs from cached expert request')
            if request.get('sample_id') is not None and request['sample_id'] != row['id']:
                raise ValueError('cached expert sample ID differs from manifest')
        modality = original.get('input_modality')
        if not modality or modality == 'mixed':
            raise ValueError('cached baseline must record a routed modality')
        cases.append({'id': row['id'], 'image': row['image'], 'question': row['question'],
                      'modality': modality, 'task': row.get('task', 'open_vqa'),
                      'domain': row.get('domain', 'plug-query'),
                      'group_id': row.get('group_id', row['image_sha256']),
                      'image_sha256': row['_pixel_sha256']})
    source = Path(__file__).parent
    identity = digest({'schema': 'plug-observe-v1', 'options': options,
        'manifest': rows, 'base_protocol': base_protocol,
        'base_output_sha256': file_hash(Path(args.base_run) / f'{args.incumbent}.json'),
        'registry': specs, 'excluded': excluded, 'source_audit': source_audit,
        'implementation': {p.relative_to(source).as_posix(): file_hash(p)
                           for p in sorted(source.rglob('*.py'))}})
    return {'identity': identity, 'options': options, 'cases': cases, 'base': base,
            'base_protocol': base_protocol, 'config': config, 'specs': specs,
            'excluded': excluded, 'sources': sources, 'references': references,
            'source_audit': source_audit}


def live(args, data, root):
    import fcntl
    from dataclasses import asdict, replace

    import torch

    from .capabilities import CapabilityRequest, EvidenceItem, validate_result
    from .capability_experts import CapabilityPool
    from .capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from .generalist_factory import generalist_provenance, load_generalist
    from .open_study import model_provenance

    cfg, opts, specs = data['config'], data['options'], data['specs']
    model_ids = json_copy({'generalist': generalist_provenance(cfg['generalist'], args.artifacts),
                          'experts': {k: model_provenance(v, args.artifacts) for k, v in specs.items()}})
    with (root / '.assets.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'model_provenance.json'
        if path.exists() and json.loads(path.read_text()) != model_ids:
            raise ValueError('model assets changed; do not reuse old caches')
        atomic_json(path, model_ids)
    load_started = perf_counter()
    probe = load_generalist(cfg['generalist'], args.artifacts)
    probe.model.eval().requires_grad_(False)
    load_seconds = perf_counter() - load_started
    pool = CapabilityPool(specs, args.artifacts, source_records=data['sources'],
                          source_references=data['references'])
    shard = root / 'shards' / str(args.shard_index)
    shard.mkdir(parents=True, exist_ok=True)
    values = {name: {} for name in METHODS}
    cases = data['cases'][args.shard_index::args.shard_count]
    try:
        for index, case in enumerate(cases):
            if args.canary_cases and index >= args.canary_cases:
                break
            cache = shard / 'case-cache' / (digest(case['id']) + '.json')
            if cache.exists():
                saved = json.loads(cache.read_text())
                if (saved.get('identity') != data['identity']
                        or set(saved.get('outputs', {})) != set(METHODS)
                        or saved.get('case_id') != case['id']):
                    raise ValueError('case cache mismatch')
                outputs = saved['outputs']
            else:
                historical = data['base'][case['id']]
                # Use original medical values, never answer/reference labels.
                native = tuple(EvidenceItem(**v) for v in historical['evidence'])
                generation = replace(ValueGenerationConfig(**historical['generation_config']),
                                     max_new_tokens=opts['max_new_tokens'],
                                     block_tokens=opts['max_new_tokens'])
                # All new methods have the same neutral prompt; no CE/OE branching.
                prompt = case['question'] + ANSWER_SUFFIX
                start = perf_counter()
                session = NativeSession(probe, case['image'], prompt, case['question'], generation)
                with torch.inference_mode():
                    block = session.propose(NativeState(items=native), opts['max_new_tokens'])
                baseline = {'text': session.decode(block.tokens).strip(),
                            'token_ids': list(block.tokens), 'finished': block.finished,
                            'evidence': copy.deepcopy(historical['evidence']), 'trace': [],
                            'seconds': perf_counter() - start,
                            'evidence_transport': getattr(session, 'last_transport', {}),
                            'generation_config': asdict(generation),
                            'input_modality': case['modality']}
                if not baseline['text'] or not baseline['token_ids']:
                    raise RuntimeError('empty fresh compact baseline')
                baseline['inherited_expert_seconds'] = historical.get('seconds')
                baseline['inherited_cost_includes_historical_answer'] = True
                baseline['replayed_experts'] = False
                baseline['agent_workflow'] = {'calls': [{'role': 'incumbent_answer',
                    'model_generation_started': True, 'output_tokens': len(block.tokens),
                    'seconds': baseline['seconds']}], 'candidate': None,
                    'region_count': 0, 'medical_verification_implemented': False}
                inherited = {o['expert_id'] for o in historical['evidence']}
                # Only the exact old observations actually delivered by the baseline
                # are protected. Do not silently turn old unpresented packets into inputs.
                delivered = {(v['expert_id'], v['evidence_id']) for v in
                             baseline['evidence_transport'].get('presented', [])}
                seeded = [v for v in historical['evidence']
                          if (v['expert_id'], v['evidence_id']) in delivered]
                baseline['seeded_evidence_count'] = len(seeded)
                baseline['inherited_expert_ids'] = sorted(inherited)
                baseline['historical_answer_used_as_prompt'] = False
                # Source-answer content policy is inherited, not secretly relaxed.
                from .evidence_need import presentation_items
                seeded = [asdict(v) for v in presentation_items(
                    tuple(EvidenceItem(**v) for v in seeded), case['question'], generation)]

                def generate(image, text, tokens, allowed):
                    with torch.inference_mode():
                        return probe.generate_with_usage(image, text, max_new_tokens=tokens,
                                                         allowed_texts=allowed)

                def invoke(action, current):
                    request = CapabilityRequest(sample_id=current['id'], image=current['image'],
                        question=current['question'], modality=current['modality'], task=current['task'],
                        domain=current['domain'], group_id=current['group_id'],
                        capability=action.capability, scope=action.scope, query=current['question'],
                        region=tuple(action.region['box']) if action.region else None)
                    # Trusted plugins still own preprocessing. Frozen mode is explicit;
                    # inference_mode alone is not described as retraining an expert.
                    model = pool._model(specs[action.expert])
                    for target in (model, getattr(model, 'model', None)):
                        if isinstance(target, torch.nn.Module):
                            target.eval().requires_grad_(False)
                    with torch.inference_mode():
                        result = validate_result(pool.infer(action.expert, request), action.expert, request)
                    if not result.items:
                        raise ToolUnavailable(result.reason)
                    # A supplied, audited source manifest explicitly enables native
                    # source reference content. The new observation retains source-only
                    # scope; legacy seed policies above remain unchanged.
                    return [asdict(v) for v in result.items]

                outputs = {'incumbent': baseline}
                for name, mode in [('plug_read', 'read'), ('plug_static', 'fixed'), ('plug_agent', 'agent')]:
                    pool.reset_case()
                    outputs[name] = run_case(case=case, specs=specs, seed_items=seeded,
                        incumbent=baseline, generate=generate, measure=probe.context_token_budget,
                        invoke=invoke, output_dir=shard / 'crops' / digest(case['id']) / name,
                        mode=mode, **opts)
                for out in outputs.values():
                    out['inherited_cost_upper_bound_seconds'] = historical.get('seconds')
                    out['historical_generation_not_rerun_for_seed_tools'] = True
                    out['generation_config'] = asdict(generation)
                    out['uniform_free_prompt'] = True
                atomic_json(cache, {'identity': data['identity'], 'case_id': case['id'],
                                    'outputs': outputs})
            for name in METHODS:
                values[name][case['id']] = outputs[name]
            print(f"completed {index + 1}/{len(cases)} {case['id']}", flush=True)
        complete = len(values['incumbent']) == len(cases)
        atomic_json(shard / 'results.json', {'identity': data['identity'],
            'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'complete': complete, 'outputs': values, 'model_provenance_sha256': digest(model_ids),
            'generalist_initialization_seconds': load_seconds})
        return complete
    except Exception as exc:
        # Preserve an honest failure snapshot, never a successful fallback row.
        atomic_json(shard / 'failure.json', {'error': type(exc).__name__, 'message': str(exc),
            'completed_cases': len(values['incumbent']), 'identity': data['identity']})
        raise
    finally:
        pool.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--incumbent', default='compact_rows', choices=['compact_rows', 'compact_all', 'semantic_all'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', default='configs/plug_observe.yaml')
    parser.add_argument('--expert-registry', help='Optional add-only experts mapping, using existing factory interface')
    parser.add_argument('--source-manifest')
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--canary-cases', type=int, default=0)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    if args.canary_cases < 0 or (args.merge_only and (args.check_only or args.canary_cases)):
        parser.error('invalid check/merge/canary combination')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    data = prepare(args)
    root = Path(args.output) / data['identity']
    root.mkdir(parents=True, exist_ok=True)
    preflight = {'identity': data['identity'], 'n': len(data['cases']), 'methods': METHODS,
        'output_root': str(root.resolve()), 'base_identity': data['base_protocol']['identity'],
        'options': data['options'], 'experts': data['specs'], 'excluded': data['excluded'],
        'source_audit': data['source_audit'], 'weights_fitted': False, 'calibration_fitted': False,
        'prompt_protocol': 'uniform-free-v1', 'answer_type_used_at_inference': False,
        'historical_scores_directly_comparable': False, 'clinical_gate_implemented': False,
        'new_retrieval_content_policy': 'native_source_scoped_with_explicit_corpus'}
    atomic_json(root / 'preflight.json', preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.check_only:
        return
    import fcntl
    if not args.merge_only:
        with (root / f'.shard-{args.shard_index}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            complete = live(args, data, root)
        if not complete:
            print('Canary is partial. Inspect delivery and real candidate counts before continuing.')
            return
    if args.merge_only or args.shard_count == 1:
        with (root / '.merge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merge_shards(root, data['identity'], [c['id'] for c in data['cases']], METHODS, args.shard_count)
            atomic_json(root / 'protocol.json', {**preflight, 'shards_complete': True,
                                                'shard_count': args.shard_count})


if __name__ == '__main__':
    main()
