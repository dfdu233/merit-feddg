"""Offline-only historical harm audit; never imported by the inference runner."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

from run_soft_guidance_pilot import SOURCES

from merit_feddg.open_study import fingerprint

sys.path.insert(0, '/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import (
    PROTOCOL_VERSION,
    evaluate_rows,
)
from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall


def main():
    root=Path('runs/soft-guidance-history-audit')
    root.mkdir(parents=True,exist_ok=True)
    for dataset,(source,data) in SOURCES.items():
        destination=root/f'{dataset}.json'
        if destination.exists():
            raise RuntimeError('do not overwrite historical audit')
        base=Path('/home/dbw/merit-feddg/runs')/source
        inputs=Path('/home/dbw/merit-feddg/runs')/data
        rows=[json.loads(s) for s in (inputs/'manifest.jsonl').read_text().splitlines()]
        refs=json.loads((inputs/'references.json').read_text())
        generalist=json.loads((base/'generalist.json').read_text())
        compact,segmentation={},{}
        for index,row in enumerate(rows):
            o=json.loads((base/'case-cache/compact_rows'/f"{fingerprint(row['id'])}.json").read_text())['output']
            compact[row['id']]={'text':o['text']}
            segmentation[row['id']]=any(i['capability']=='segmentation' for i in o['evidence'])
            if index%250==0: print(dataset,'read',index,flush=True)
        scores={}
        for name,outputs in [('generalist',generalist),('compact_rows',compact)]:
            records=[{'qid':r['id'],'question':r['question'],'answer':refs[r['id']][0],
                      'answer_type':r['answer_type'],'text':outputs[r['id']]['text']} for r in rows]
            evaluated=evaluate_rows(records)
            detail={d['question_id']:d for d in evaluated['details']}
            scores[name]={r['id']:(float(detail[r['id']]['correct']) if r['answer_type']=='closed'
                            else answer_token_recall(outputs[r['id']]['text'],refs[r['id']][0])) for r in rows}
        bad=[r['id'] for r in rows if scores['generalist'][r['id']]>scores['compact_rows'][r['id']]]
        result={'dataset':dataset,'n':len(rows),'scorer':PROTOCOL_VERSION,
                'metric':'decoded strict CLOSED + unchanged ANCHOR OPEN answer-token-recall',
                'scorer_sha256':hashlib.sha256(Path('/home/dbw/ANCHOR/anchor/corrected_sgta/evaluate_medheval_answers.py').read_bytes()).hexdigest(),
                'means':{k:statistics.mean(v.values()) for k,v in scores.items()},
                'scores':scores,'historical_harm_ids':bad,
                'first_two_harm_with_segmentation':[k for k in bad if segmentation[k]][:2],
                'improvements':sum(scores['compact_rows'][r['id']]>scores['generalist'][r['id']] for r in rows),
                'harms':len(bad),'labels_used_only_offline':True}
        with destination.open('x') as f: json.dump(result,f,indent=2)
        print(dataset,'means',result['means'],'harm',len(bad),'selected',result['first_two_harm_with_segmentation'],flush=True)


if __name__=='__main__':
    main()
