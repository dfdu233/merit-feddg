"""Frozen, test-image-disjoint TRAIN diagnostic; no labels enter GPU inference."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml
from torchxrayvision.models import model_urls

from run_uncertainty_train import (
    GPUS, MANIFEST, XRV, XRV_SHA, resources, gpu_check, load_generalist,
    evidence, sha,
)
from evaluate_soft_guidance_full import scorer_hashes
from merit_feddg.authority_projection import (
    normalized_pool_distribution, project_binary_authority, xrv_operating_coordinate,
)
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.evidence_admission import EvidenceRequest
from merit_feddg.experts.native_xrv import XrvCapabilityAdapter
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_data import pixel_digest
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.uncertainty_experiment import applicability

WORDS = {'Effusion': 'pleural effusion', 'Cardiomegaly': 'cardiomegaly',
         'Pneumothorax': 'pneumothorax'}
GROUPS = ['positive', 'positive', 'negative', 'negative']


def candidates(label, style):
    target = WORDS[label]
    residual = WORDS['Cardiomegaly' if label == 'Effusion' else 'Effusion']
    if style == 0:
        first = [f'Yes, {target} is present.', f'No, {target} is absent.']
        second = [f'{residual.capitalize()} is present.', f'{residual.capitalize()} is absent.']
    else:
        first = [f'Yes, the image shows {target}.', f'No, the image does not show {target}.']
        second = [f'The image shows {residual}.', f'The image does not show {residual}.']
    return first, [a+' '+b for a in first for b in second]


def fixed_marginal_projection(p, target):
    """Standard iterative proportional fitting, a baseline, not a novel method."""
    q = p.reshape(2, 2).copy()
    rows = np.array([target, 1-target])
    cols = q.sum(axis=0).copy()
    for step in range(10000):
        q *= (rows/q.sum(axis=1))[:, None]
        q *= (cols/q.sum(axis=0))[None, :]
        if max(np.max(abs(q.sum(axis=1)-rows)), np.max(abs(q.sum(axis=0)-cols))) < 1e-10:
            return q.ravel(), step+1
    raise RuntimeError('IPF did not converge; do not silently continue')


def prepare(root):
    if root.exists():
        raise RuntimeError('prepare requires a fresh directory')
    protocol, rows, old = resources()
    test = MANIFEST.parent.parent/'test/manifest.jsonl'
    test_rows = [json.loads(x) for x in test.read_text().splitlines()]
    forbidden = {r['image_sha256'] for r in test_rows}
    selected, seen = [], set()
    eligible = 0
    for row in rows:
        audit = applicability(row, old[row['id']])
        if not audit['allowed'] or row['image_sha256'] in forbidden:
            continue
        eligible += 1
        if row['image_sha256'] in seen or len(selected) >= 12:
            continue
        assert row['official_split'] == 'train'
        assert not {'answer', 'answers', 'label', 'report', 'mask'} & set(row)
        seen.add(row['image_sha256'])
        selected.append({'id':row['id'], 'image_sha256':row['image_sha256'], 'label':audit['label']})
    if not selected:
        raise RuntimeError('no eligible independent TRAIN images')
    specs = yaml.safe_load(Path('configs/llava_med_capabilities.yaml').read_text())['experts']
    cfg = {
        'experiment':'paired-support-gpu1-v1',
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'selected':selected, 'eligible_questions':eligible,
        'selection':'first occurrence in frozen TRAIN order, globally unique RGB hash, exclude all test RGB hashes, at most 12',
        'patient_independence':'patient/study identifiers unavailable; not established',
        'manifest':str(MANIFEST), 'manifest_sha256':sha(MANIFEST), 'test_manifest_sha256':sha(test),
        'generalist':protocol['config']['generalist'], 'prompt_config':protocol['config'],
        'generation':asdict(replace(ValueGenerationConfig(**old[selected[0]['id']]['generation_config']), admission_mode='enforce')),
        'specs':{'cxr_findings':specs['cxr_findings']}, 'xrv_sha256':XRV_SHA,
        'candidate_styles':{label:[candidates(label,i) for i in range(2)] for label in WORDS},
        'scorer':scorer_hashes(),
        'code':{p:sha(p) for p in [__file__, 'merit_feddg/authority_projection.py',
                 'merit_feddg/evidence_admission.py', 'merit_feddg/llava_generalist.py']},
        'H1':'real sequence preferences permit target updates with measurable residual propagation; compare direct replacement',
        'H0':'effects reduce to direct expert choice, candidate wording or scoring normalization',
        'primary':'target mass/selection and residual mass/selection under fixed candidate support',
        'secondary':'offline target-only correctness; no residual medical correctness claims',
        'controls':['two fixed paraphrases','sum versus mean log-likelihood','nonrequested score mutation delivery invariance',
                    'single-pair direct-expert equivalence','base versus focused free generation prefix/full comparison'],
        'stop':'delivery failure, nonfinite score, IPF failure or no usable support; no tuning or expanded run',
        'no_training':True, 'no_calibration':True, 'inference_reference_access':False,
        'gpu_uuid':GPUS[0],
    }
    atomic_json(root/'frozen.json',cfg)
    print('PREPARED', fingerprint(cfg), len(selected), dict(Counter(s['label'] for s in selected)),flush=True)


def run(root, max_cases):
    cfg=json.loads((root/'frozen.json').read_text()); identity=fingerprint(cfg)
    assert all(sha(p)==h for p,h in cfg['code'].items())
    assert sha(MANIFEST)==cfg['manifest_sha256']
    rows={r['id']:r for r in map(json.loads,MANIFEST.read_text().splitlines())}
    gpu_check(GPUS[0]); started=time.perf_counter()
    source_path=root/'sources.json'
    if source_path.exists():
        sources=json.loads(source_path.read_text()); assert sources['identity']==identity
    else:
        expert=XrvCapabilityAdapter(XRV,'classification',device='cuda',sha256=XRV_SHA)
        cases={}
        for s in cfg['selected']:
            row=rows[s['id']]; assert pixel_digest(row['image'])==s['image_sha256']
            t=time.perf_counter(); labels,scores,transform=expert.classify(row['image'])
            cases[s['id']]={'label':s['label'],'probability':float(scores[list(labels).index(s['label'])]),
                'labels':list(labels),'scores':scores.tolist(),'transform':transform,'seconds':time.perf_counter()-t}
        sources={'identity':identity,'cases':cases,'expert_calls':len(cases),'load_and_inference_seconds':time.perf_counter()-started}
        atomic_json(source_path,sources); del expert; gc.collect(); torch.cuda.empty_cache()
    t=time.perf_counter(); probe=load_generalist(cfg['generalist'],'artifacts')
    probe.model.eval().requires_grad_(False)
    atomic_json(root/f'load-{time.time_ns()}.json',{'seconds':time.perf_counter()-t,'gpu_uuid':GPUS[0]})
    meta=model_urls['densenet121-res224-all']; ops=dict(zip(meta['labels'],meta['op_threshs']))
    done=0
    for s in cfg['selected']:
        key=s['id']; path=root/'cases'/f'{key}.json'
        if path.exists():
            assert json.loads(path.read_text())['identity']==identity
            continue
        t=time.perf_counter(); row=rows[key]; source=sources['cases'][key]
        assert pixel_digest(row['image'])==s['image_sha256']
        request=EvidenceRequest(row['question'],(s['label'],),frozenset({'finding_presence'}),'cxr','open_vqa')
        packet=replace(evidence(key,source,False),expert_id='cxr_findings')
        generation=ValueGenerationConfig(**cfg['generation'])
        prompt=generation_prompt(row,cfg['prompt_config'])
        def session(items):
            native=NativeSession(probe,row['image'],prompt,row['question'],generation,
                                 authority_specs=cfg['specs'],evidence_request=request)
            image,text=native.context(NativeState(items=items))
            return native,probe.new_answer_session(image,text),text
        base_native,base,_=session(())
        focused_native,focused,focused_prompt=session((packet,))
        delivered=focused_native.delivery_items((packet,))
        assert len(delivered)==1
        assert [v['finding'] for v in delivered[0].payload['findings']]==[s['label']]
        assert focused_native.last_transport['presented']
        changed=asdict(packet)
        for entry in changed['payload']['findings']:
            if entry['finding']!=s['label']: entry['score']=1-entry['score']
        from merit_feddg.capabilities import EvidenceItem
        _,_,control_prompt=session((EvidenceItem(**changed),))
        assert focused_prompt==control_prompt
        free={}
        with torch.inference_mode():
            for name,ctx in [('base',base),('focused',focused)]:
                block=ctx.propose((),count=1,length=64)[0]
                free[name]={'text':block.text,'token_ids':list(block.tokens)}
            styles=[]; calls=0
            for style in range(2):
                pair,pool=candidates(s['label'],style)
                tokens=[probe.tokenizer.encode(text,add_special_tokens=False)+[probe.tokenizer.eos_token_id] for text in pool+pair]
                scores={name:[float(ctx.sequence_mean_logp((),tok)) for tok in tokens]
                        for name,ctx in [('base',base),('focused',focused)]}
                calls+=2*len(tokens)
                assert all(np.isfinite(values).all() for values in scores.values())
                results={}
                for norm in ['mean','sum']:
                    bs=np.array(scores['base'][:4]);fs=np.array(scores['focused'][:4])
                    if norm=='sum':
                        lengths=np.array([len(tok) for tok in tokens[:4]]);bs*=lengths;fs*=lengths
                    p=normalized_pool_distribution(bs); base_index=int(np.argmax(p))
                    methods={'base':p.tolist(),'focused':normalized_pool_distribution(fs).tolist(),
                             'fixed_cad':normalized_pool_distribution(1.5*fs-.5*bs).tolist()}
                    targets={'raw':source['probability'],'op':xrv_operating_coordinate(source['probability'],ops[s['label']])}
                    selected={'base':base_index,'focused':int(np.argmax(methods['focused'])),
                              'fixed_cad':int(np.argmax(methods['fixed_cad']))}
                    for coordinate,value in targets.items():
                        q=project_binary_authority(bs,GROUPS,value)['projected_probabilities']
                        bounded,steps=fixed_marginal_projection(p,value)
                        methods['authority_'+coordinate]=q.tolist();methods['bounded_'+coordinate]=bounded.tolist()
                        selected['authority_'+coordinate]=int(np.argmax(q));selected['bounded_'+coordinate]=int(np.argmax(bounded))
                        selected['direct_'+coordinate]=(0 if value>=.5 else 2)+base_index%2
                        pair_q=project_binary_authority(scores['base'][4:],['positive','negative'],value)['projected_probabilities']
                        assert np.allclose(pair_q,[value,1-value],atol=1e-12)
                    results[norm]={'distributions':methods,'indices':selected,
                        'residual_mass':{name:float(np.array(pv)[[0,2]].sum()) for name,pv in methods.items()},
                        'target_mass':{name:float(sum(pv[:2])) for name,pv in methods.items()}}
                styles.append({'style':style,'candidates':pool,'pair':pair,'token_lengths':[len(tok) for tok in tokens],
                               'sequence_scores':scores,'results':results})
        record={'identity':identity,'id':key,'label':s['label'],'image_sha256':s['image_sha256'],
            'free':free,'styles':styles,'source':source,'admission':focused_native.last_admission,
            'transport':focused_native.last_transport,'unrequested_mutation_prompt_invariant':True,
            'pair_equivalence_verified':True,'prefix8_same':free['base']['token_ids'][:8]==free['focused']['token_ids'][:8],
            'full_tokens_same':free['base']['token_ids']==free['focused']['token_ids'],
            'score_calls':calls,'generation_calls':2,'wall_seconds':time.perf_counter()-t}
        atomic_json(path,record);done+=1
        print('DONE',key,'score_calls',calls,'seconds',round(record['wall_seconds'],2),flush=True)
        if max_cases and done>=max_cases: return
    print('COMPLETE',identity,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-cases',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.output)
    else: run(args.output,args.max_cases)
