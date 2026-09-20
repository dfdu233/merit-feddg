"""Actual Huatuo worker and isolated frozen ANCHOR scorer integration.

No new model download, training, prompt rewrite or specialist rerun. This module
requires the existing server environment. Import is CPU-safe; weights load only
inside run_job. The source-run inventory and content hashes come from freeze.
"""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

from .storage import canonical, digest, fingerprint, pin_files, read, write

ROOT = Path(__file__).resolve().parents[2]


def legacy_runner():
    path = ROOT/'scripts/run_huatuo_pathway.py'
    spec = importlib.util.spec_from_file_location('_merit_frozen_pathway_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_device(uuid, allowed_display_contexts=(), allowed_compute_contexts=()):
    if not isinstance(uuid, str) or not uuid.startswith('GPU-'):
        raise ValueError('Explicit authorized physical GPU UUID required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the authorized UUID, not a guessed index')
    available = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                                        text=True, timeout=10).splitlines()
    if uuid not in [v.strip() for v in available]:
        raise RuntimeError('Authorized device is not visible')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True, timeout=10)
    allowed = {}
    if allowed_display_contexts:
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'], text=True, timeout=10))
        for gpu in tree.findall('gpu'):
            if gpu.findtext('uuid') != uuid:
                continue
            for proc in gpu.findall('./processes/process_info'):
                for entry in allowed_display_contexts:
                    # A frozen, identified desktop context is not a blanket
                    # exemption for small compute jobs or other C+G clients.
                    if (proc.findtext('type') == 'C+G'
                            and proc.findtext('process_name') == entry['process_name'] == '/usr/bin/nautilus'
                            and int(proc.findtext('pid')) == entry['pid']
                            and 0 < int(proc.findtext('used_memory').split()[0]) <= entry['max_memory_mib'] <= 64):
                        allowed[entry['pid']] = entry
    if allowed_compute_contexts:
        # Explicit, frozen coexistence authorization; never stop another job.
        memory = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                          '--format=csv,noheader,nounits'], text=True, timeout=10)
        for line in memory.splitlines():
            fields = [v.strip() for v in line.split(',')]
            if len(fields) != 3 or fields[0] != uuid:
                continue
            for entry in allowed_compute_contexts:
                if int(fields[1]) == entry['pid']:
                    command = Path(f"/proc/{entry['pid']}/cmdline").read_bytes()
                    if (hashlib.sha256(command).hexdigest() == entry['cmdline_sha256']
                            and int(fields[2]) <= entry['max_memory_mib']):
                        allowed[entry['pid']] = entry
        free = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        if int(free.strip()) < max(e['minimum_free_mib'] for e in allowed_compute_contexts):
            raise RuntimeError('Insufficient free memory for authorized GPU sharing')
    for line in processes.splitlines():
        values = [v.strip() for v in line.split(',')]
        if len(values) == 2 and values[0] == uuid and int(values[1]) != os.getpid() and int(values[1]) not in allowed:
            raise RuntimeError('Another compute process owns the authorized GPU; no job is killed')


def build_controls(legacy, adapter, row, case, cfg, limit, central, transport):
    """Two deterministic order/layout interventions on the SAME delivered packets.

    This is a nuisance probe, not evidence filtering or new medical statements.
    No transformation of native label lists, polarities, numbers or mask values.
    """
    from ..capabilities import EvidenceItem
    from ..capability_runtime import ValueGenerationConfig, presentation_items
    from ..compact_evidence import compact_records, compact_prompt
    from ..huatuo_pathway import native_inputs
    config = ValueGenerationConfig(**cfg)
    if not config.compact_native or config.evidence_style != 'semantic':
        raise ValueError('C requires the original compact-native semantic interface')
    keys = [(v['expert_id'], v['evidence_id']) for v in transport['presented']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate delivered packet ID')
    items = [EvidenceItem(**e) for e in case['compact_raw']['evidence']]
    selected = [v for v in presentation_items(tuple(items), row['question'], config)
                if (v.expert_id,v.evidence_id) in set(keys)]
    if {(v.expert_id,v.evidence_id) for v in selected} != set(keys):
        raise ValueError('Cannot reconstruct exact delivered packet set')
    records = compact_records(selected, geometry=config.compact_geometry)
    # Preserve the selected order, no new budget-induced inclusion/exclusion.
    if compact_prompt(row['benchmark_prompt'], records, columns=config.compact_columns) != central:
        raise ValueError('Direct native renderer does not reproduce central prompt')
    views = [compact_prompt(row['benchmark_prompt'], records, columns=not config.compact_columns),
             compact_prompt(row['benchmark_prompt'], list(reversed(records)), columns=config.compact_columns)]
    for view in views:
        ids, _ = native_inputs(adapter, row['image'], view)
        tower = adapter.model.get_vision_tower()
        count = ids.shape[1]+tower.num_patches+int(getattr(tower,'select_feature','patch')=='cls_patch')-1
        if count > adapter.model.config.tokenizer_model_max_length or count+cfg['max_new_tokens'] > limit:
            raise ValueError('A fixed control exceeds budget; no packet may be silently dropped')
    return views, dict(canonical_packet_sha256=fingerprint(records),
        packet_ids=keys, transformations=['toggle_row_column_encoding','reverse_packet_order'],
        distinct_control_prompts=len(set([central,*views]))-1,
        interpretation='Same native content; format/order are assumed nuisance, not proven medically invariant')


def verify_transport(case, prompt, transport):
    """Source stores native controls in compact_raw/transport as used by old runner."""
    historical = [t['evidence_transport'] for t in case['compact_raw'].get('trace', [])
                  if t.get('event') == 'decode' and 'evidence_transport' in t]
    if not historical:
        raise ValueError('Historical decode transport missing; do not substitute answer parity')
    saved = historical[-1]
    if saved != transport:
        raise ValueError('Historical and reconstructed transport differ')
    if saved.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError('Historical prompt hash mismatch')


def _check_cached_decision(diag, baseline, compact):
    if diag['status'] != 'divergence':
        return
    prefix = diag['prefix_ids']
    position = len(prefix)
    if baseline[:position] != prefix or compact[:position] != prefix:
        raise ValueError('Diagnostic shared prefix differs from cached controls')
    if len(baseline) <= position or len(compact) <= position or baseline[position] != diag['a'] or compact[position] != diag['b']:
        raise ValueError('Diagnostic first decision differs from cached native outputs')


class ForwardBudget:
    """Hard language-model forward cap including canaries and failed attempts.

    Persist before each forward so interruption cannot erase spent attempts.
    Counter I/O is part of reported experimental latency, not deployment latency.
    """
    def __init__(self, model, limit, path):
        self.model, self.limit, self.path = model, limit, Path(path)
        if type(limit) is not int or limit < 1:
            raise ValueError('No remaining forward budget')
        self.used = read(self.path)['used'] if self.path.exists() else 0
        self.handle = None

    def __enter__(self):
        self.handle = self.model.register_forward_pre_hook(self._count)
        return self

    def _count(self, module, args):
        from ..pathway_restore import PathwayError
        if self.used >= self.limit:
            write(self.path, dict(used=self.used, limit=self.limit, exhausted=True), replace=True)
            raise PathwayError('Hard experimental forward budget exhausted')
        self.used += 1
        write(self.path, dict(used=self.used, limit=self.limit, exhausted=False), replace=True)

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


def run_job(job, output):
    """Inference child process: job deliberately contains no scorer/reference paths."""
    if any(k in job for k in ('scorer','references','scores','score_data')):
        raise ValueError('Scoring data must not enter the inference job')
    from .decoding import DecodePolicy, cut_layers, diagnose, generate
    from ..huatuo_pathway import native_inputs, prepare_native, prepare_pair, _sync
    from ..pathway_restore import PathwayError
    import torch

    runtime = job['runtime']
    pin_files(runtime['pins'])
    check_device(runtime['gpu_uuid'], runtime.get('allowed_display_contexts', ()),
                 runtime.get('allowed_compute_contexts', ()))
    for variable in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
        os.environ[variable] = '1'
    for root in runtime.get('import_roots', []):
        sys.path.append(str(Path(root).resolve()))
    module_name, factory_name = runtime['factory'].split(':',1)
    factory = getattr(importlib.import_module(module_name),factory_name)
    if str(Path(inspect.getfile(factory)).resolve()) not in runtime['pins']:
        raise ValueError('Adapter source must be content-pinned before execution')
    started = time.perf_counter()
    adapter = factory(**runtime.get('kwargs',{}))
    model = adapter.model.eval().requires_grad_(False)
    with ForwardBudget(model, job['remaining_forwards'], Path(output)/'budget.json') as budget:
        legacy = legacy_runner()
        legacy.check_native_generation_defaults(model.generation_config)
        checkpoint = Path(model.config._name_or_path).resolve()
        if checkpoint != Path(runtime['checkpoint_dir']).resolve():
            raise ValueError('Loaded checkpoint differs from frozen runtime')
        weight_files = [p for p in checkpoint.rglob('*') if p.is_file() and p.suffix in ('.safetensors','.bin')]
        if not weight_files or any(str(p.resolve()) not in runtime['pins'] for p in weight_files):
            raise ValueError('All local checkpoint weight files must have actual SHA256 pins')
        _sync(model.device)
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        job_id = fingerprint(job)
        manifest = output/'job.json'
        if manifest.exists() and read(manifest) != job:
            raise ValueError('Existing stage belongs to another immutable job')
        if not manifest.exists():
            write(manifest,job)
        import transformers
        runtime_info = dict(torch=torch.__version__, transformers=transformers.__version__,
            merit_source=str(Path(inspect.getfile(importlib.import_module('merit_feddg'))).resolve()),
            model_config=model.config.to_dict(), adapter_source=str(Path(inspect.getfile(factory)).resolve()),
            native_model_source=str(Path(inspect.getfile(type(model))).resolve()),
            attention_classes=sorted({type(x.self_attn).__name__ for x in model.get_model().layers}),
            gpu_uuid=runtime['gpu_uuid'])
        runtime_info = read_json_normalized(runtime_info)
        print(canonical(runtime_info), flush=True)
        runtime_path = output/'runtime.json'
        if runtime_path.exists() and read(runtime_path) != runtime_info:
            raise ValueError('Runtime changed on resume')
        if not runtime_path.exists():
            write(runtime_path,runtime_info)
        write(output/('load-'+str(time.time_ns())+'.json'), dict(seconds=time.perf_counter()-started))
        all_results = []
        for inv in job['inventories']:
            root = Path(inv['source_run'])
            if digest(root/'protocol.json') != inv['protocol_sha256']:
                raise ValueError('Source changed after freeze')
            source = read(root/'protocol.json')
            settings = job['generation'][inv['cohort']]
            legacy.validate_generation_kwargs(settings, source['generation_config']['max_new_tokens'])
            eos = settings.get('eos_token_id',model.generation_config.eos_token_id)
            if eos is None:
                raise ValueError('Explicit EOS required')
            bos = tuple(model._maybe_initialize_input_ids_for_generation(None,model.generation_config.bos_token_id,
                model_kwargs={'inputs_embeds':torch.empty(1,1,model.config.hidden_size,device=model.device)})[0].tolist())
            policy = DecodePolicy(settings['max_new_tokens'], (eos,) if isinstance(eos,int) else tuple(eos),
                                   settings['repetition_penalty'],settings['min_new_tokens'],bos)
            limit = int(model.config.max_position_embeddings)
            exposed_canary = False
            for index, (row, frozen_row) in enumerate(zip(source['rows'],inv['records'])):
                uid = frozen_row['id']
                target = output/'cases'/(fingerprint(uid)+'.json')
                if target.exists():
                    record = read(target)
                    if record.get('job_sha') != job_id or record.get('id') != uid or not record.get('complete'):
                        raise ValueError('Invalid resumed case')
                    exposed_canary |= record.get('canary',{}).get('expert_exposed',False)
                    all_results.append(record)
                    continue
                if index == 2 and not exposed_canary:
                    raise ValueError('First two fixed cases did not exercise expert context; no automatic favorable reselection')
                case_path = root/'cases'/(row['id']+'.json')
                if digest(case_path) != frozen_row['case_sha256'] or digest(row['image']) != frozen_row['image_sha256']:
                    raise ValueError('Frozen case/image changed')
                case = read(case_path)
                record = dict(id=uid, source_id=row['id'], cohort=inv['cohort'], dataset=inv['dataset'],
                    pixel_sha256=frozen_row['pixel_sha256'], job_sha=job_id, complete=False, arms={})
                for name in ('generalist','compact'):
                    arm = case['arms'][name]
                    record['arms'][name] = dict(status='complete',text=arm['text'],token_ids=arm['token_ids'],
                                                reused=True,seconds=None)
                prepare_start = time.perf_counter()
                rendered = case.get('rendered_context') if job.get('rendered_contexts', False) else None
                if rendered is not None:
                    if job['node'] != 'TEST':
                        raise ValueError('Rendered transport is restricted to explicit TEST jobs')
                    prompt, transport = rendered['prompt'], rendered['transport']
                    if [hashlib.sha256(t.encode()).hexdigest() for t in [prompt,*rendered['controls']]] != rendered['prompt_hashes']:
                        raise ValueError('Frozen rendered contexts changed')
                else:
                    prompt, transport = legacy.restore_context(adapter,row,case,source['generation_config'],limit)
                verify_transport(case,prompt,transport)
                if job.get('refresh_native_controls', False):
                    # A new device/runtime must use its own native controls.
                    # Generate them immediately before this case's candidate,
                    # inside the same forward ledger; labels remain scorer-only.
                    if job['node'] != 'TEST':
                        raise ValueError('Inline native refresh is restricted to explicit TEST jobs')
                    refresh_start, refresh_forwards = time.perf_counter(), budget.used
                    for name, text in [('generalist',row['benchmark_prompt']),('compact',prompt)]:
                        ids, pixels = native_inputs(adapter,row['image'],text)
                        _sync(model.device)
                        arm_start, arm_forwards = time.perf_counter(), budget.used
                        torch.cuda.reset_peak_memory_stats(model.device)
                        with torch.inference_mode(), adapter._generation_last_token_logits():
                            native = model.generate(ids,images=pixels,return_dict_in_generate=True,
                                                    output_scores=True,**settings)
                        count = len(native.scores)
                        tokens = native.sequences[0,-count:].tolist() if count else []
                        answer = adapter.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                        if not tokens or not answer:
                            raise ValueError('Empty inline native control')
                        _sync(model.device)
                        arm = dict(status='complete',text=answer,token_ids=tokens,reused=False,
                            seconds=time.perf_counter()-arm_start,measured_forwards=budget.used-arm_forwards,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
                            historical_token_match=tokens==case['arms'][name]['token_ids'])
                        record['arms'][name] = arm
                        case['arms'][name] = arm
                        del native, ids, pixels
                    record['native_refresh_seconds'] = time.perf_counter()-refresh_start
                    record['native_refresh_forwards'] = budget.used-refresh_forwards
                ref, rec = prepare_pair(adapter,row['image'],row['benchmark_prompt'],prompt)
                record['transport_sha256'] = fingerprint(transport)
                record['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
                _sync(model.device)
                record['preparation_seconds'] = time.perf_counter()-prepare_start
                canary_start, canary_forwards = time.perf_counter(), budget.used
                if index < 2:
                    checks = {}
                    for name, prepared, text in [('generalist',ref,row['benchmark_prompt']),('compact',rec,prompt)]:
                        ids,pixels = native_inputs(adapter,row['image'],text)
                        with torch.inference_mode(), adapter._generation_last_token_logits():
                            native = model.generate(ids,images=pixels,return_dict_in_generate=True,
                                                    output_scores=True,**settings)
                        count = len(native.scores)
                        actual = native.sequences[0,-count:].tolist() if count else []
                        off = generate(model,ref,prepared,policy,algorithm='off',context_limit=limit)
                        checks[name] = actual == off['token_ids'] == case['arms'][name]['token_ids']
                        if not checks[name]:
                            write(output/('parity-failed-'+fingerprint(uid)+'.json'),
                                  dict(arm=name,native=actual,replay=off,expected=case['arms'][name]['token_ids']))
                            raise ValueError('Native/historical/off canary mismatch')
                    audit = generate(model,ref,rec,policy,algorithm='audit',layer=cut_layers(model)[0],context_limit=limit)
                    if audit['token_ids'] != case['arms']['compact']['token_ids']:
                        raise ValueError('Residual audit changed native tokens')
                    checks.update(audit=True,expert_exposed=not audit['bypass'])
                    record['canary'] = checks
                    exposed_canary |= checks['expert_exposed']
                _sync(model.device)
                record['canary_seconds'] = time.perf_counter()-canary_start if index < 2 else 0.0
                record['canary_forwards'] = budget.used-canary_forwards
                record['preparation_and_canary_seconds'] = time.perf_counter()-prepare_start
                stage = job['node']
                if stage == 'A':
                    _sync(model.device)
                    torch.cuda.reset_peak_memory_stats(model.device)
                    diagnostic_start, diagnostic_forwards = time.perf_counter(), budget.used
                    try:
                        record['diagnostic'] = diagnose(model,ref,rec,policy,context_limit=limit,
                                                        max_prefix=job['policy']['diagnostic_prefix_tokens'])
                        _check_cached_decision(record['diagnostic'],case['arms']['generalist']['token_ids'],
                                                case['arms']['compact']['token_ids'])
                    except (PathwayError,ValueError) as error:
                        record['diagnostic'] = dict(status='failed',error=str(error))
                        if hasattr(error, 'parity_details'):
                            record['diagnostic']['parity_details'] = error.parity_details
                    _sync(model.device)
                    record['diagnostic']['measured_seconds'] = time.perf_counter()-diagnostic_start
                    record['diagnostic']['measured_forwards'] = budget.used-diagnostic_forwards
                    record['diagnostic']['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(model.device)
                else:
                    candidate = job['candidate']
                    controls = []
                    control_error = None
                    if candidate['algorithm'] == 'project':
                        try:
                            if rendered is not None:
                                texts, certificate = rendered['controls'], rendered['certificate']
                            else:
                                texts, certificate = build_controls(legacy,adapter,row,case,
                                    source['generation_config'],limit,prompt,transport)
                            controls = [prepare_native(adapter,row['image'],t)[0] for t in texts]
                            record['nuisance_certificate'] = certificate
                        except (ValueError,PathwayError) as error:
                            control_error = str(error)
                    for name, algorithm in [('candidate',candidate['algorithm']),('control',candidate['control'])]:
                        _sync(model.device)
                        arm_start, arm_forwards = time.perf_counter(), budget.used
                        try:
                            if control_error:
                                raise PathwayError(control_error)
                            prediction = generate(model,ref,rec,policy,algorithm=algorithm,
                                layer=candidate['layer'],controls=controls,context_limit=limit)
                            prediction.update(status='complete',text=adapter.tokenizer.decode(
                                prediction['token_ids'],skip_special_tokens=True).strip())
                            if not prediction['text']:
                                raise PathwayError('Empty decoded response is not a completed candidate')
                            record['arms'][name] = prediction
                        except (PathwayError,ValueError) as error:
                            record['arms'][name] = dict(status='failed',error=str(error),token_ids=[],text=None,
                                seconds=None, cost_unknown=True, hooks_removed=not getattr(model,'_merit_chain_active',False))
                        _sync(model.device)
                        record['arms'][name]['measured_seconds'] = time.perf_counter()-arm_start
                        record['arms'][name]['measured_forwards'] = budget.used-arm_forwards
                record['complete'] = True
                write(target,record)
                all_results.append(record)
                print('DONE',stage,uid,flush=True)
            if not exposed_canary:
                raise ValueError('No actual expert-exposed technical canary')
        result = dict(job_sha=job_id,node=job['node'],candidate=job.get('candidate'),
                      complete=True,planned_n=len(all_results),records=all_results)
        path = output/'predictions.json'
        if path.exists():
            if read(path) != result:
                raise ValueError('Existing predictions changed')
        else:
            write(path,result)


def read_json_normalized(value):
    import json
    return json.loads(canonical(value))


def score_job(job, predictions_path, scorer, output):
    """Separate subprocess only: exactly the frozen CLOSED + OPEN source metric."""
    pin_files(scorer['pins'])
    for p in scorer.get('import_roots',[]):
        sys.path.append(p)
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION,evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    for function in (evaluate_rows,answer_token_recall):
        path = str(Path(inspect.getfile(function)).resolve())
        if path not in scorer['pins'] or digest(path) != scorer['pins'][path]:
            raise ValueError('Actual imported scorer differs from frozen code')
    predictions = read(predictions_path)
    if predictions['job_sha'] != fingerprint(job) or not predictions['complete']:
        raise ValueError('Wrong or incomplete inference artifact')
    records = {r['id']:r for r in predictions['records']}
    result = []
    for inv in job['inventories']:
        rows = read(Path(inv['source_run'])/'protocol.json')['rows']
        ref_path = scorer['references'][inv['cohort']]
        if ref_path not in scorer['pins']:
            raise ValueError('Reference file is not content-pinned')
        references = read(ref_path)
        for row,frozen_row in zip(rows,inv['records']):
            pred = records[frozen_row['id']]
            refs = references[row['id']]
            if not isinstance(refs,list) or not refs:
                raise ValueError('First-reference protocol needs a nonempty list')
            values = {}
            for name,arm in pred['arms'].items():
                if arm.get('status') == 'failed':
                    values[name] = None
                    continue
                if row['answer_type'] == 'closed':
                    detail = evaluate_rows([dict(qid=frozen_row['id'],question=row['question'],
                        answer_type=row['answer_type'],answer=refs[0],text=arm['text'])])['details']
                    if len(detail)!=1 or detail[0]['question_id']!=frozen_row['id']:
                        raise ValueError('Scorer ID mismatch')
                    values[name] = float(detail[0]['correct'])
                else:
                    values[name] = float(answer_token_recall(arm['text'],refs[0]))
            result.append(dict(id=frozen_row['id'],scores=values))
    write(output,dict(predictions_sha256=digest(predictions_path),job_sha=fingerprint(job),
                      scorer_pins=scorer['pins'],scorer_version=PROTOCOL_VERSION,records=result))
