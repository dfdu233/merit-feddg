"""Verify and adapt complete existing official TEST caches; never generate answers.

Explicit TEST evaluation only. Does not relax the TRAIN chain's source contract.
Reference answers are exported separately for the isolated offline scorer.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/dbw/ANCHOR')
from anchor.corrected_sgta.protocol_v2 import build_prompt
from merit_feddg.algorithm_chain.storage import (read, write, digest, fingerprint,
                                               pin_files, pixel_hash, safe_name)


def prepare(output):
    if output.exists():
        raise FileExistsError('Preserve existing source adapters')
    anchor = Path('/home/dbw/ANCHOR')
    prior = read(ROOT/'runs/chain-v1-expansion151-gpu0-v2/plan.json')
    inventories, references, audit = [], {}, {}
    pixel_cache = {}
    for dataset, expected in [('vqa_rad', 451), ('slake', 2094)]:
        native = Path('/home/dbw/merit-feddg-expert-coverage/runs/huatuo-merit-full-v1')/dataset
        baseline = anchor/'corrected_runs/paper_baselines_v1/huatuo_final_v1/full'/dataset/'greedy'
        manifest = anchor/'corrected_runs/paper_baselines_v1/merit_common_protocol_v1'/(dataset+'.jsonl')
        canonical = (anchor/'data/vqa_rad/official_test_full_v1.json' if dataset == 'vqa_rad'
                     else Path('/home/dbw/data/SLAKE/test.json'))
        import json
        rows = [json.loads(s) for s in manifest.read_text().splitlines()]
        gold = {str(r['id'] if dataset == 'vqa_rad' else r['qid']): r for r in read(canonical)}
        answers = [json.loads(s) for s in (baseline/'answers.jsonl').read_text().splitlines()]
        bs = {str(r['question_id']): r for r in answers}
        ids = {r['id'] for r in rows}
        case_ids = {p.stem for p in (native/'cases').glob('*.json')}
        if not (len(rows) == len(ids) == len(answers) == len(bs) == len(gold) == expected
                and ids == set(gold) == set(bs) == case_ids):
            raise ValueError('Require the complete official TEST denominator: '+dataset)
        mp, bp = read(native/'protocol.json'), read(baseline/'generation_config.json')
        done = read(native/'actor-complete.json')
        if done['identity'] != mp['identity'] or done['n'] != expected:
            raise ValueError('Incomplete original native run')
        if bp['manifest_sha256'] != digest(canonical) or mp['manifest_sha256'] != digest(manifest):
            raise ValueError('Official manifest changed')
        if (bp['max_new_tokens'] != 1024 or mp['generation_config']['max_new_tokens'] != 1024
                or bp['native_repetition_penalty'] != 1.2 or bp['native_min_new_tokens'] != 1):
            raise ValueError('Historical generation settings differ')
        for pins in [mp['entry_sources'], mp['formal_sources'], mp['base']['huatuo_canary_sources']]:
            pin_files(pins)
        adapter = '/home/dbw/ANCHOR/anchor/corrected_sgta/models_oe.py'
        protocol_code = '/home/dbw/ANCHOR/anchor/corrected_sgta/protocol_v2.py'
        if (digest(adapter) != bp['native_adapter_sha256']
                or digest(protocol_code) != bp['prompt_protocol_sha256']
                or digest('/home/dbw/models/HuatuoGPT-Vision-7B/config.json') != bp['model_config_sha256']):
            raise ValueError('Historical baseline adapter/prompt/model configuration changed')
        source = output/dataset
        cohort = dataset+'-official-test'
        adapted, records = [], []
        originals = {str(p): digest(p) for p in [manifest, native/'protocol.json',
                     native/'actor-complete.json', baseline/'answers.jsonl', baseline/'generation_config.json']}
        for row in rows:
            key = safe_name(row['id']); g = gold[key]; b = bs[key]
            mpath = native/'cases'/(key+'.json'); m = read(mpath)
            if set(row) & {'answer','answers','label','labels','reference','references','ground_truth'}:
                raise ValueError('Labels in inference manifest')
            if row['question'] != g['question'] or row['benchmark_prompt'] != build_prompt(g):
                raise ValueError('Historical official question/prompt differs')
            if digest(row['image']) != row['image_sha256']:
                raise ValueError('Image changed')
            meta = b['metadata']
            if (meta['img_name'] != g['img_name'] or meta['effective_img_name'] != g['img_name']
                    or meta['fingerprint'] != bp['fingerprint'] or meta['effective_max_new_tokens'] != 1024
                    or meta.get('input_error')):
                raise ValueError('Invalid cached baseline')
            if m['identity'] != mp['identity'] or m['generation_config'] != mp['generation_config']:
                raise ValueError('Invalid cached compact identity')
            traces = m['output']['trace']
            if any('runtime_error' in str(t.get('reason','')) for t in traces):
                raise ValueError('Historical expert failure')
            if not any(t.get('event') == 'decode' and 'evidence_transport' in t for t in traces):
                raise ValueError('Missing historical native transport')
            for arm in [dict(text=b['text'], token_ids=meta['generated_token_ids']), m['output']]:
                if not arm['token_ids'] or not isinstance(arm['text'], str) or not arm['text'].strip():
                    raise ValueError('Incomplete cached answer')
            originals[str(mpath)] = digest(mpath)
            if row['image'] not in pixel_cache:
                pixel_cache[row['image']] = pixel_hash(row['image'])
            pixel, shape = pixel_cache[row['image']]
            adapted.append(dict(id=key, complete=True, compact_raw=m['output'],
                arms=dict(generalist=dict(text=b['text'],token_ids=meta['generated_token_ids']),
                          compact=m['output']), reused_only=True))
            records.append(dict(id=cohort+':'+key, source_id=key, pixel_sha256=pixel,
                                image_size=shape, image_sha256=row['image_sha256']))
        # Identity explicitly says TEST and verified adaptation, never a fake new native execution.
        payload = dict(rows=rows, generation_config=mp['generation_config'], dataset=dataset,
            split='official_test', selection='complete official TEST manifest order; no score selection',
            reused_only=True, original_pins=originals, original_native_identity=mp['identity'],
            original_baseline_fingerprint=bp['fingerprint'],
            adapter_source_sha256=digest(__file__), new_model_forwards=0)
        identity = fingerprint(payload)
        write(source/'protocol.json', dict(identity=identity, **payload))
        for case, record in zip(adapted, records):
            path = source/'cases'/(case['id']+'.json')
            write(path, dict(identity=identity, **case))
            record['case_sha256'] = digest(path)
        write(source/'complete.json', dict(identity=identity,n=expected,reused_only=True,
            meaning='All original complete outputs validated and adapted; zero new answer generation'))
        refs = output/(dataset+'-references.json')
        write(refs, {key:[str(gold[key]['answer'])] for key in ids})
        references[cohort] = str(refs.resolve())
        inventories.append(dict(cohort=cohort, dataset=dataset, split='official_test',
            source_run=str(source.resolve()), protocol_sha256=digest(source/'protocol.json'),
            source_identity=identity, records=records))
        audit[dataset] = dict(planned=expected, unique_images=len({r['pixel_sha256'] for r in records}),
            canonical_source_sha256=digest(canonical), native_identity=mp['identity'],
            reused_cases=expected, score_selection=False, historical_test_outputs_previously_available=True,
            languages={lang:sum(g.get('q_lang','unspecified')==lang for g in gold.values())
                       for lang in sorted({g.get('q_lang','unspecified') for g in gold.values()})})
        print('VERIFIED OFFICIAL TEST',dataset,expected,flush=True)
    write(output/'inventory.json',inventories)
    write(output/'references-map.json',references)
    write(output/'audit.json',dict(datasets=audit, inherited_forwards=44791,
        native_candidate_source=prior['identity'], no_independent_confirmation_claim=True,
        no_new_forward_calls=True, full_scope=2545))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.output.resolve())
