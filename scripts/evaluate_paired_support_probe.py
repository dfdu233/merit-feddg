"""Offline target-only and locality diagnostics; refuses incomplete GPU runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from evaluate_soft_guidance_full import scorer_hashes, paired
from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main(root):
    cfg=json.loads((root/'frozen.json').read_text()); identity=fingerprint(cfg)
    ids=[s['id'] for s in cfg['selected']]
    paths={p.stem:p for p in (root/'cases').glob('*.json')}
    assert set(paths)==set(ids), 'all frozen cases must finish before reference access'
    records={k:json.loads(paths[k].read_text()) for k in ids}
    assert all(r['identity']==identity for r in records.values())
    assert scorer_hashes()==cfg['scorer']
    if (root/'evaluation.json').exists(): raise RuntimeError('do not overwrite evaluation')
    manifest=Path(cfg['manifest'])
    rows={r['id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    refs=json.loads((manifest.parent/'references.json').read_text())
    sys.path.insert(0,'/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows, PROTOCOL_VERSION

    def score(texts):
        result=evaluate_rows([{'qid':k,'question':rows[k]['question'],'answer_type':rows[k]['answer_type'],
                              'answer':refs[k][0],'text':texts[k]} for k in ids])
        return {r['question_id']:float(r['correct']) for r in result['details']}

    def metrics(values,baseline):
        return {'correct':sum(values.values()),'n':len(ids),'vs_base':paired(values,baseline,ids),
                'image_bootstrap':cluster_bootstrap([values[k]-baseline[k] for k in ids],
                                                   [records[k]['image_sha256'] for k in ids])}

    free_scores={arm:score({k:records[k]['free'][arm]['text'] for k in ids}) for arm in ['base','focused']}
    report={'identity':identity,'complete':True,'n':len(ids),'image_clusters':len({r['image_sha256'] for r in records.values()}),
        'label_counts':dict(Counter(r['label'] for r in records.values())), 'scorer':PROTOCOL_VERSION,
        'target_only_task_metric':True,'residual_correctness_evaluated':False,
        'full_benchmark':False,'new_gate_trained':False,
        'free':{arm:metrics(s,free_scores['base']) for arm,s in free_scores.items()},
        'admission_positive_count':sum(bool(r['transport']['presented']) for r in records.values()),
        'nonrequested_mutation_invariant_count':sum(r['unrequested_mutation_prompt_invariant'] for r in records.values()),
        'pair_equivalence_count':sum(r['pair_equivalence_verified'] for r in records.values()),
        'prefix8_same':sum(r['prefix8_same'] for r in records.values()),
        'full_tokens_same':sum(r['full_tokens_same'] for r in records.values()),
        'prefix_hidden_changes':sum(r['prefix8_same'] and not r['full_tokens_same'] for r in records.values()),
        'cells':{},'cost':{'actor_generation_calls':sum(r['generation_calls'] for r in records.values()),
                         'sequence_score_calls':sum(r['score_calls'] for r in records.values()),
                         'case_wall_seconds':sum(r['wall_seconds'] for r in records.values()),
                         'actor_load_seconds':sum(json.loads(p.read_text())['seconds'] for p in root.glob('load-*.json'))},
        'limitations':['Four-state residual is an unqueried logically compatible finding, not statistically independent.',
            'Candidate residual truth is not annotated; marginal drift is not a clinical harm metric.',
            'All candidates are authored templates; this does not establish open-ended semantic locality.',
            'Raw and operating-point scores are not calibrated target-domain probabilities.',
            'First eligible image order, narrow presence grammar, 12 images; no population-level gain claim.',
            'No exact test RGB overlap; patient/study and pretrained model overlap unknown.',
            'Offline target-only scoring of the candidate first clause excludes the unverified residual.',
            'Fixed-CAD here is sequence-score reranking, not token-level CAD generation.']}
    for style in range(2):
        for norm in ['mean','sum']:
            states={k:records[k]['styles'][style]['results'][norm] for k in ids}
            arm_names=list(states[ids[0]]['indices'])
            scores={arm:score({k:records[k]['styles'][style]['pair'][states[k]['indices'][arm]//2]
                              for k in ids}) for arm in arm_names}
            cell={}
            for arm in arm_names:
                info=metrics(scores[arm],scores['base'])
                info['vs_free_base']=paired(scores[arm],free_scores['base'],ids)
                info['target_changed_vs_pool_base']=sum(states[k]['indices'][arm]//2!=states[k]['indices']['base']//2 for k in ids)
                info['residual_selected_changed_vs_pool_base']=sum(states[k]['indices'][arm]%2!=states[k]['indices']['base']%2 for k in ids)
                if arm in states[ids[0]]['residual_mass']:
                    drift=[abs(states[k]['residual_mass'][arm]-states[k]['residual_mass']['base']) for k in ids]
                    info['residual_mass_drift_mean']=statistics.mean(drift)
                    info['residual_mass_drift_max']=max(drift)
                    info['target_mass_change_mean']=statistics.mean(abs(states[k]['target_mass'][arm]-states[k]['target_mass']['base']) for k in ids)
                if arm.startswith(('authority_','bounded_')):
                    coordinate=arm.rsplit('_',1)[1]
                    direct='direct_'+coordinate
                    info['vs_direct']=paired(scores[arm],scores[direct],ids)
                    info['target_disagreements_with_direct']=sum(states[k]['indices'][arm]//2!=states[k]['indices'][direct]//2 for k in ids)
                cell[arm]=info
            report['cells'][f'style{style}_{norm}']=cell
    report['sensitivity']={}
    for arm in report['cells']['style0_mean']:
        report['sensitivity'][arm]={
            'target_flips_style_mean':sum(records[k]['styles'][0]['results']['mean']['indices'][arm]//2 !=
                                          records[k]['styles'][1]['results']['mean']['indices'][arm]//2 for k in ids),
            'target_flips_norm_style0':sum(records[k]['styles'][0]['results']['mean']['indices'][arm]//2 !=
                                           records[k]['styles'][0]['results']['sum']['indices'][arm]//2 for k in ids)}
    sources=json.loads((root/'sources.json').read_text())
    report['cost'].update({k:sources[k] for k in ['expert_calls','load_and_inference_seconds']})
    atomic_json(root/'evaluation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
