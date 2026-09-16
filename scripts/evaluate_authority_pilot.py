"""Offline frozen TRAIN pilot evaluation, never a full-test claim."""
import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

from evaluate_soft_guidance_full import scorer_hashes, paired
from run_soft_guidance_full import sha
from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    cfg = json.loads((root/'frozen.json').read_text())
    assert scorer_hashes() == cfg['scorer']
    assert sha(cfg['manifest']) == cfg['manifest_sha256']
    ids = [s['id'] for s in cfg['selected']]
    files = {p.stem:p for p in (root/'cases').glob('*.json')}
    assert set(files) == set(ids), 'entire frozen pilot must complete'
    records = {k:json.loads(files[k].read_text()) for k in ids}
    assert all(r['identity'] == fingerprint(cfg) for r in records.values())
    manifest = Path(cfg['manifest'])
    rows = {r['id']: r for r in map(json.loads, manifest.read_text().splitlines())}
    refs = json.loads((manifest.parent/'references.json').read_text())
    assert set(refs) == set(rows)
    sys.path.insert(0, '/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows, PROTOCOL_VERSION
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    scores = {}
    for arm in ('incumbent', 'pool_base', 'pool_text', 'authority'):
        evaluated = evaluate_rows([{'qid':k, 'question':rows[k]['question'],
            'answer_type':rows[k]['answer_type'], 'answer':refs[k][0],
            'text':records[k]['arms'][arm]} for k in ids])
        detail = {r['question_id']:r for r in evaluated['details']}
        scores[arm] = {k:float(detail[k]['correct']) if rows[k]['answer_type']=='closed'
                       else answer_token_recall(records[k]['arms'][arm], refs[k][0]) for k in ids}
    report = {'identity':fingerprint(cfg), 'n':len(ids), 'full_train_n':len(rows),
              'pilot_complete':True, 'full_manifest_complete':False,
              'scorer':PROTOCOL_VERSION, 'image_clusters':len({rows[k]['image_sha256'] for k in ids}),
              'arms':{}, 'projection_status':dict(collections.Counter(r['projection']['status'] for r in records.values())),
              'candidate_counts':[len(records[k]['candidates']) for k in ids],
              'historical_token_parity':sum(r['historical_token_parity'] for r in records.values()),
              'cost':{k:sum(r['cost'][k] for r in records.values()) for k in next(iter(records.values()))['cost']},
              'transport_toward_source':dict(collections.Counter(str(r['transport_diagnostic']['toward_source']) for r in records.values()))}
    for arm, values in scores.items():
        report['arms'][arm] = {'score':statistics.mean(values.values()),
            'vs_incumbent':paired(values, scores['incumbent'], ids),
            'vs_pool_base':paired(values, scores['pool_base'], ids),
            'changed_from_incumbent':sum(records[k]['arms'][arm]!=records[k]['arms']['incumbent'] for k in ids),
            'image_bootstrap':cluster_bootstrap([values[k]-scores['incumbent'][k] for k in ids], [rows[k]['image_sha256'] for k in ids])}
    if (root/'evaluation.json').exists():
        raise RuntimeError('refusing to overwrite evaluation')
    atomic_json(root/'evaluation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
