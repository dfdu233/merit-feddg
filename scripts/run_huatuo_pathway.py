"""Replay frozen native Huatuo/MERIT inputs with opt-in pathway restoration.

No specialist reruns, no ground-truth reads, no new router or trained verifier.
Use an existing TRAIN run. The full queue/modes are frozen before any inference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


LABEL_KEYS = {'answer', 'answers', 'label', 'labels', 'reference', 'references', 'ground_truth'}
MODES = ('restore_mass', 'restore_vector')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    """Atomic exclusive publication; preserve existing/incomplete runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pathway-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails on existing destination instead of replacing it.
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def output_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.pathway.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_generation_kwargs(value, max_new_tokens):
    required = {'do_sample', 'num_beams', 'repetition_penalty', 'min_new_tokens', 'max_new_tokens'}
    allowed = required | {'eos_token_id', 'pad_token_id', 'use_cache'}
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - allowed:
        raise ValueError('Supply explicit native greedy kwargs only; unsupported processors are not silently dropped')
    if (value['do_sample'] is not False or type(value['num_beams']) is not int
            or value['num_beams'] != 1 or value.get('use_cache', True) is not True):
        raise ValueError('This implementation supports cached greedy generation only')
    if (type(value['max_new_tokens']) is not int or value['max_new_tokens'] < 1
            or type(value['min_new_tokens']) is not int
            or not 0 <= value['min_new_tokens'] < value['max_new_tokens']
            or not math.isfinite(value['repetition_penalty']) or value['repetition_penalty'] <= 0):
        raise ValueError('Invalid explicit greedy settings')
    if value['max_new_tokens'] != max_new_tokens:
        raise ValueError('Answer budget differs from the frozen source run')
    return dict(value)


def check_native_generation_defaults(config):
    # model.generate inherits these even when not present in explicit kwargs.
    neutral = dict(num_return_sequences=1, num_beam_groups=1, min_length=0,
        no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
        encoder_repetition_penalty=1.0, bad_words_ids=None, force_words_ids=None,
        constraints=None, suppress_tokens=None, begin_suppress_tokens=None,
        forced_bos_token_id=None, forced_eos_token_id=None, forced_decoder_ids=None,
        sequence_bias=None, watermarking_config=None, guidance_scale=None,
        remove_invalid_values=False, token_healing=False, stop_strings=None,
        max_time=None, exponential_decay_length_penalty=None, cache_implementation=None)
    for key, default in neutral.items():
        if getattr(config, key, default) != default:
            raise ValueError('Unsupported inherited generation option: ' + key)


def assert_exposed_canary(output, rows):
    canaries = [read(output / 'cases' / (row['id'] + '.json')) for row in rows]
    if not any(c.get('parity', {}).get('attention_path_exercised') for c in canaries):
        raise RuntimeError('Canary contained no expert-exposed attention pass; do not call it a passed pathway test')


def frozen_source(directory):
    source = read(directory / 'protocol.json')
    complete = read(directory / 'complete.json')
    rows = source['rows']
    if complete['identity'] != source['identity'] or complete['n'] != len(rows) or not rows:
        raise ValueError('Complete frozen source queue required')
    if 'train' not in str(source.get('selection', '')).lower():
        raise ValueError('Use an explicitly identified existing TRAIN queue, not a new target benchmark')
    if not source.get('test_image_exclusion_sha256'):
        raise ValueError('Source must record its prior test-image exclusion audit')
    seen, hashes = set(), {}
    for row in rows:
        if set(row) & LABEL_KEYS:
            raise ValueError('Ground truth must not enter the generation manifest')
        key = row['id']
        if key in seen or Path(key).name != key or key in ('.', '..') or '/' in key or '\\' in key:
            raise ValueError('Duplicate/unsafe sample ID')
        seen.add(key)
        if digest(row['image']) != row['image_sha256']:
            raise ValueError('Source image changed')
        path = directory / 'cases' / (key + '.json')
        case = read(path)
        if case.get('id') != key or case['identity'] != source['identity'] or not case['complete']:
            raise ValueError('Incomplete or incompatible source case')
        for arm in ('generalist', 'compact'):
            if not case['arms'][arm]['token_ids'] or not isinstance(case['arms'][arm]['text'], str):
                raise ValueError('Native controls must have real token/text outputs')
        hashes[key] = digest(path)
    for group in ('source', 'formal_sources'):
        for name, expected in source.get(group, {}).items():
            if not Path(name).is_file() or digest(name) != expected:
                raise ValueError('Pinned source/backend dependency changed: ' + name)
    # Source-only property is inherited from this recorded audit, not re-proved here.
    return source, hashes


def restore_context(adapter, row, case, generation_config, limit):
    from merit_feddg.huatuo_pathway import native_inputs
    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    config = ValueGenerationConfig(**generation_config)
    if config.semantic_spatial or config.evidence_style == 'tensor' or config.visual_views:
        raise ValueError('This pilot preserves the original single image; spatial-input rewriting is a separate method')

    class Budget:
        def context_token_budget(self, image, prompt, reserve_tokens):
            ids, _ = native_inputs(adapter, image, prompt)
            expanded = ids.shape[1] + adapter.model.get_vision_tower().num_patches - 1
            return dict(fits=expanded + reserve_tokens <= limit, input_tokens=expanded,
                        remaining_tokens=limit - expanded, reserved_tokens=reserve_tokens)

    items = tuple(EvidenceItem(**e) for e in case['compact_raw']['evidence'])
    session = NativeSession(Budget(), row['image'], row['benchmark_prompt'], row['question'], config)
    image, prompt = session.context(NativeState(items=items))
    if image != row['image']:
        raise ValueError('Native context unexpectedly replaced the original image')
    return prompt, session.last_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation-json', type=Path, required=True,
                        help='Exact baseline kwargs, NOT guessed defaults; five required greedy fields')
    parser.add_argument('--adapter-factory', default='anchor.corrected_sgta.models_oe:HuatuoOEAdapter')
    parser.add_argument('--adapter-kwargs', type=Path)
    parser.add_argument('--import-root', type=Path, action='append', default=[])
    parser.add_argument('--stage', choices=('check', 'canary', 'run'), default='check')
    parser.add_argument('--canary-cases', type=int, default=2)
    parser.add_argument('--layers', nargs='+', type=int, default=[-1])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or args.canary_cases < 1:
        raise ValueError('Unique modes and positive canary count required')
    source, case_hashes = frozen_source(args.source_run)
    settings = validate_generation_kwargs(read(args.generation_json), source['generation_config']['max_new_tokens'])
    code = {str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__), *sorted((ROOT / 'merit_feddg').glob('*.py'))]}
    frozen = dict(source_protocol_sha256=digest(args.source_run / 'protocol.json'),
                  source_identity=source['identity'], case_sha256=case_hashes, generation=settings,
                  modes=args.modes, layers=args.layers, canary_cases=args.canary_cases,
                  adapter_factory=args.adapter_factory,
                  adapter_kwargs=read(args.adapter_kwargs) if args.adapter_kwargs else {}, code=code,
                  no_training=True, no_reference_answers=True,
                  source_image_exclusion='inherited from source audit; not re-proved by this runner')
    frozen['identity'] = fingerprint(frozen)
    if args.stage == 'check':
        print(json.dumps(dict(static_identity=frozen['identity'], n=len(source['rows']),
                             stage='static_check_only', medical_model_loaded=False), indent=2))
        return
    # No network downloads, environment installations, or automatic GPU selection.
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[name] = '1'
    for path in args.import_root:
        sys.path.append(str(path.resolve()))
    module, function = args.adapter_factory.split(':', 1)
    factory = getattr(importlib.import_module(module), function)
    import torch
    from merit_feddg.huatuo_pathway import GreedySpec, native_inputs, paired_generate, prepare_pair
    from merit_feddg.pathway_restore import PathwayConfig

    with output_lock(args.output):
        loading_started = time.perf_counter()
        adapter = factory(**frozen['adapter_kwargs'])
        model = adapter.model.eval().requires_grad_(False)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - loading_started
        provenance = dict(merit_feddg=inspect.getfile(importlib.import_module('merit_feddg')),
                          adapter=inspect.getfile(factory), native_model=inspect.getfile(type(model)),
                          attention_classes=sorted({type(layer.self_attn).__name__ for layer in model.get_model().layers}))
        print(json.dumps(provenance), flush=True)
        check_native_generation_defaults(model.generation_config)
        frozen['adapter_source_sha256'] = digest(inspect.getfile(factory))
        frozen['torch_version'] = torch.__version__
        import transformers
        frozen['transformers_version'] = transformers.__version__
        frozen['model_config'] = model.config.to_dict()
        frozen['import_provenance'] = provenance
        frozen['native_generation_config'] = model.generation_config.to_dict()
        checkpoint = Path(str(getattr(model.config, '_name_or_path', '')))
        frozen['checkpoint_file_stats_not_content_hashes'] = {
            str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(checkpoint.glob('*')) if p.is_file()
        } if checkpoint.is_dir() else {}
        frozen['identity'] = fingerprint({k: v for k, v in frozen.items() if k != 'identity'})
        protocol_path = args.output / 'protocol.json'
        if protocol_path.exists():
            if read(protocol_path) != frozen:
                raise ValueError('Frozen model/runtime/config/input changed; use a new output directory')
        else:
            write_new(protocol_path, frozen)
        write_new(args.output / ('load-' + str(time.time_ns()) + '.json'), dict(seconds=load_seconds, stage=args.stage))
        limit = min(int(getattr(model.config, name)) for name in
                    ('tokenizer_model_max_length', 'max_position_embeddings') if getattr(model.config, name, None))
        eos = settings.get('eos_token_id', model.generation_config.eos_token_id)
        if eos is None:
            raise ValueError('EOS must be explicit in native settings or model generation config')
        eos = (eos,) if isinstance(eos, int) else tuple(eos)
        spec = GreedySpec(settings['max_new_tokens'], eos,
                          settings['repetition_penalty'], settings['min_new_tokens'],
                          tuple(model._maybe_initialize_input_ids_for_generation(
                              None, model.generation_config.bos_token_id,
                              model_kwargs={'inputs_embeds': torch.empty(1, 1, model.config.hidden_size,
                                                                        device=model.device)})[0].tolist()))
        count = min(args.canary_cases, len(source['rows']))
        planned_rows = source['rows'][:count] if args.stage == 'canary' else source['rows']
        for index, row in enumerate(planned_rows):
            if index == count:
                assert_exposed_canary(args.output, source['rows'][:count])
            target = args.output / 'cases' / (row['id'] + '.json')
            if target.exists():
                existing = read(target)
                if existing.get('identity') != frozen['identity'] or not existing.get('complete'):
                    raise ValueError('Existing incompatible/incomplete output; do not overwrite')
                continue
            original_path = args.source_run / 'cases' / (row['id'] + '.json')
            if digest(row['image']) != row['image_sha256']:
                raise ValueError('Image changed after preflight')
            if digest(original_path) != case_hashes[row['id']]:
                raise ValueError('Cached source changed after preflight')
            case = read(original_path)
            prompt, transport = restore_context(adapter, row, case, source['generation_config'], limit)
            reference, receiver = prepare_pair(adapter, row['image'], row['benchmark_prompt'], prompt)
            result = dict(id=row['id'], identity=frozen['identity'], complete=False,
                          source_case_sha256=case_hashes[row['id']], transport=transport,
                          expanded_tokens=[reference.length, receiver.length],
                          visual_spans=[vars(reference.visual_span), vars(receiver.visual_span)],
                          arms={name: dict(text=case['arms'][name]['text'],
                                           token_ids=case['arms'][name]['token_ids'], reused=True)
                                for name in ('generalist', 'compact')})
            # Necessary canary only; do not re-generate the full stored baseline.
            if index < count:
                result['parity'] = {}
                for name, prepared, text in (('generalist', reference, row['benchmark_prompt']),
                                             ('compact', receiver, prompt)):
                    ids, pixels = native_inputs(adapter, row['image'], text)
                    with torch.inference_mode(), adapter._generation_last_token_logits():
                        native_output = model.generate(ids, images=pixels, **settings,
                                                       return_dict_in_generate=True, output_scores=True)
                    native_tokens = native_output.sequences[0, -len(native_output.scores):].tolist()
                    del native_output
                    off = paired_generate(model, reference, prepared, spec,
                                          PathwayConfig('off', tuple(args.layers)), context_limit=limit)
                    expected = case['arms'][name]['token_ids']
                    ok = native_tokens == expected == off['token_ids']
                    result['parity'][name] = ok
                    if not ok:
                        write_new(args.output / ('failed-parity-' + row['id'] + '.json'),
                                  dict(arm=name, expected=expected, native=native_tokens, replay=off))
                        raise RuntimeError('Native/cached/manual greedy parity failed; do not run intervention')
                audit = paired_generate(model, reference, receiver, spec,
                                        PathwayConfig('audit', tuple(args.layers)), context_limit=limit)
                if audit['token_ids'] != case['arms']['compact']['token_ids']:
                    write_new(args.output / ('failed-audit-' + row['id'] + '.json'), audit)
                    raise RuntimeError('Attention instrumentation changed control tokens')
                result['parity']['audit'] = True
                result['parity']['attention_path_exercised'] = bool(audit['events'])
            for mode in args.modes:
                prediction = paired_generate(model, reference, receiver, spec,
                                             PathwayConfig(mode, tuple(args.layers)), context_limit=limit)
                prediction['text'] = adapter.tokenizer.decode(prediction['token_ids'], skip_special_tokens=True).strip()
                result['arms'][mode] = prediction
            result['complete'] = True
            write_new(target, result)
            print('DONE', row['id'], flush=True)
        assert_exposed_canary(args.output, source['rows'][:count])
        if args.stage == 'run':
            for row in source['rows']:
                case = read(args.output / 'cases' / (row['id'] + '.json'))
                if not case.get('complete') or case.get('identity') != frozen['identity']:
                    raise ValueError('Do not publish partial/mixed coverage as complete')
            final = args.output / 'complete.json'
            if not final.exists():
                write_new(final, dict(identity=frozen['identity'], n=len(source['rows']),
                                      clinical_efficacy_evaluated=False))


if __name__ == '__main__':
    main()
