"""Offline evaluation of the exact frozen TRAIN pilot, not full-test scores."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics
import sys

from evaluate_soft_guidance_full import scorer_hashes, paired
from run_soft_guidance_full import sha
from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); root = args.run
    if (root/'evaluation.json').exists():
        raise RuntimeError('refuse to overwrite prior evaluation')
    cfg = json.loads((root/'frozen.json').read_text()); identity = fingerprint(cfg)
    if scorer_hashes() != cfg['scorer']:
        raise RuntimeError('frozen scorer changed')
    ids = [r['id'] for r in cfg['selected']]
    files = {p.stem:p for p in (root/'cases').glob('*.json')}
    if set(files) != set(ids):
        raise RuntimeError('all frozen pilot cases must complete before scoring')
    records = {k:json.loads(files[k].read_text()) for k in ids}
    if any(v['identity'] != identity or v['id'] != k for k,v in records.items()):
        raise RuntimeError('record identity mismatch')
    manifest = Path(cfg['manifest'])
    if sha(manifest) != cfg['manifest_sha256']:
        raise RuntimeError('manifest changed')
    rows = {r['id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    refs = json.loads((manifest.parent/'references.json').read_text())
    if set(rows) != set(refs):
        raise RuntimeError('full TRAIN reference alignment failure')
    sys.path.insert(0,'/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows, PROTOCOL_VERSION
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    outputs = {a:{k:records[k]['arms'][a]['text'] for k in ids} for a in cfg['arms']}
    for judge in ('cross_visual','cross_blank','self_visual'):
        outputs['verified_'+judge] = {k:records[k]['arms'][records[k]['verifiers'][judge]['selected_arm']]['text'] for k in ids}
    scores, details = {}, {}
    for arm, texts in outputs.items():
        evaluated = evaluate_rows([{'qid':k,'question':rows[k]['question'], 'answer_type':rows[k]['answer_type'],
            'answer':refs[k][0],'text':texts[k]} for k in ids])
        detail = {v['question_id']:v for v in evaluated['details']}; details[arm]=detail
        scores[arm] = {k:float(detail[k]['correct']) if rows[k]['answer_type']=='closed'
                      else answer_token_recall(texts[k],refs[k][0]) for k in ids}
    report = {'identity': identity, 'pilot_complete': True, 'full_manifest_complete': False,
        'n':len(ids), 'full_train_n':len(rows), 'scorer':PROTOCOL_VERSION,
        'evaluator_sha256':sha(__file__), 'scorer_hashes':scorer_hashes(),
        'scope':'fixed source-attribute TRAIN pilot; not a test score or clinical accuracy',
        'arms':{}, 'verifiers':{}, 'per_case_scores':scores,
        'historical_token_parity_cases':sum(r['historical_token_parity'] for r in records.values()),
        'source_uncertainty': {k:records[k]['source']['uncertainty'] for k in ids},
        'image_clusters':len({rows[k]['image_sha256'] for k in ids})}
    for arm,s in scores.items():
        leading={k:(m[1].lower() if (m:=re.match(r'^\s*(yes|no)\b',outputs[arm][k],re.I)) else None) for k in ids}
        report['arms'][arm]={'mixed_score':statistics.mean(s.values()),
            'vs_incumbent':paired(s,scores['incumbent'],ids),
            'vs_scope_text':paired(s,scores['scope_text'],ids),
            'changed_text':sum(outputs[arm][k].strip()!=outputs['incumbent'][k].strip() for k in ids),
            'leading_binary_coverage':sum(v is not None for v in leading.values()),
            'leading_binary_correct_count':sum(leading[k]==refs[k][0].strip().lower() for k in ids),
            'anchor_parsers':dict(collections.Counter(v['parser'] for v in details[arm].values())),
            'mean_weight':statistics.mean([st['effective_weight'] for k in ids
                for st in records[k]['arms'].get(arm,{}).get('steps',[])]) if any(records[k]['arms'].get(arm,{}).get('steps') for k in ids) else None,
            'image_bootstrap':cluster_bootstrap([s[k]-scores['incumbent'][k] for k in ids],[rows[k]['image_sha256'] for k in ids])}
    for judge in ('cross_visual','cross_blank','self_visual'):
        selected=[k for k in ids if records[k]['verifiers'][judge]['replace']]
        helpful=[k for k in ids if scores['source_acd'][k]>scores['incumbent'][k]]
        harmful=[k for k in ids if scores['source_acd'][k]<scores['incumbent'][k]]
        report['verifiers'][judge]={'replacements':len(selected),'helpful_offered':len(helpful),
            'harmful_offered':len(harmful),'helpful_accepted':len(set(selected)&set(helpful)),
            'harmful_accepted':len(set(selected)&set(harmful)),
            'reasons':dict(collections.Counter(records[k]['verifiers'][judge]['reason'] for k in ids)),
            'calls':sum(len(records[k]['verifiers'][judge]['calls']) for k in ids),
            'seconds':sum(c['seconds'] for k in ids for c in records[k]['verifiers'][judge]['calls'])}
    report['cost']={'case_wall_seconds':sum(r['wall_seconds'] for r in records.values()),
        'source_inference_calls':len(ids),'source_seconds':sum(r['source']['seconds'] for r in records.values()),
        'source_load_seconds':json.loads((root/'sources.json').read_text())['load_seconds'],
        'actor_judge_load_seconds':sum(json.loads(p.read_text())['seconds'] for p in root.glob('load-*.json')),
        'old_incumbent_seconds_inherited':sum(r['old_incumbent_seconds_inherited'] for r in records.values()),
        'arm_seconds':{a:sum(records[k]['arms'][a]['seconds'] for k in ids) for a in cfg['arms']},
        'parity_verification_seconds':sum(r['verification_seconds'] for r in records.values())}
    atomic_json(root/'evaluation.json', report)
    atomic_json(root/'pilot-complete.json', {'identity':identity,'pilot_complete':True,'full_manifest_complete':False,'n':len(ids)})
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_case_scores','source_uncertainty')},indent=2))


if __name__=='__main__': main()
