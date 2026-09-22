#!/usr/bin/env python3
"""Strict full-TEST export and official scoring for the overnight BARD queue."""

import gzip
import json
import os
import subprocess
from pathlib import Path


ROOT = Path('/home/dbw/merit-feddg-huatuo-spatial')
ANCHOR = Path('/home/dbw/ANCHOR')
RUN = ROOT / 'runs/mimic-gpu1-v2'
MANIFEST = ANCHOR / 'corrected_runs/paper_baselines_v1/llava_med_final_v1/mimic_cxr_full_v1/manifest.json'
SCORER = ANCHOR / 'anchor/medeval/evaluate_report_generation_final.py'
PYTHON = '/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python'


def score(model: str) -> None:
    rows = json.loads(MANIFEST.read_text())
    assert len(rows) == len({row['id'] for row in rows}) == 5159
    if model == 'llava':
        ranges = [(0, 1), (1, 64)]
        base = ANCHOR / 'corrected_runs/paper_baselines_v1/llava_med_final_v1/mimic_cxr_full_v1/greedy/evaluation_report_three_metrics.json'
    else:
        ranges = [(0, 64)]
        base = ANCHOR / 'corrected_runs/paper_baselines_v1/huatuo_final_v1/full/mimic_cxr/greedy/evaluation_report_three_metrics.json'
    ranges += [(64, 320), (320, 832), (832, 1344), (1344, 1856),
               (1856, 2368), (2368, 2880), (2880, 3392), (3392, 3904),
               (3904, 4416), (4416, 4928), (4928, 5159)]
    where = {row['id']: (start, end) for start, end in ranges for row in rows[start:end]}
    assert len(where) == 5159
    answers = []
    expert_cases = 0
    for row in rows:
        start, end = where[row['id']]
        name = f'mimic-gpu1-bard-v2-{start}-{end}'
        if model == 'huatuo' and start == 0:
            name = 'mimic-gpu1-bard-0-64'
        case = ROOT / 'runs' / f'{model}-test-v1' / name / row['id']
        with gzip.open(case / 'bard.json.gz', 'rt') as handle:
            answer = json.load(handle)
        with gzip.open(case / 'provenance.json.gz', 'rt') as handle:
            provenance = json.load(handle)
        assert provenance['row']['image_sha256'] == row['image_sha256']
        assert provenance['methods'] == ['bard']
        expert_cases += bool(answer['expert_branches'])
        answers.append({'question_id': row['id'], 'text': answer['text'], 'gt_ans': row['answer']})
    assert expert_cases > 0, (model, 'no expert-branch cases')
    answer_path = RUN / f'{model}-bard-v2-full-answers.jsonl'
    answer_path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in answers))
    result_path = RUN / f'{model}-bard-v2-full-evaluation.json'
    env = dict(os.environ, PYTHONPATH=str(ANCHOR))
    subprocess.run([PYTHON, str(SCORER), '--manifest', str(MANIFEST),
                    '--answers', str(answer_path), '--output', str(result_path)],
                   cwd='/tmp', env=env, check=True)
    result = json.loads(result_path.read_text())
    baseline = json.loads(base.read_text())
    summary = {'model': model, 'n': len(answers), 'expert_branch_cases': expert_cases,
               'candidate': result['primary_metrics'], 'historical_greedy': baseline['primary_metrics'],
               'candidate_scoring': str(result_path), 'historical_scoring': str(base)}
    (RUN / f'{model}-bard-v2-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    score('llava')
    score('huatuo')
