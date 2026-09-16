"""Read-only workload structure audit. Never use answers or explanations."""
import collections
import hashlib
import json
from pathlib import Path
import sys


def audit(path):
    rows = []
    for line in Path(path).open():
        x = json.loads(line)
        rows.append(dict(case_id=x['case_id'], images=tuple(x['images']),
                         question_hash=hashlib.sha256(x['question'].encode()).hexdigest(),
                         question_id=x['full_question_id']))
    cases = collections.defaultdict(list)
    for row in rows:
        cases[row['case_id']].append(row)
    counts = collections.Counter(len(v) for v in cases.values())
    occurrences = sum(len(r['images']) for r in rows)
    images = {i for r in rows for i in r['images']}
    sets = {r['images'] for r in rows}
    scoped = {(r['case_id'], i) for r in rows for i in r['images']}
    query_keys = {(r['case_id'], r['images'], r['question_hash']) for r in rows}
    return dict(
        audit='medrax-reuse-structure-20260916-v1', rows=len(rows),
        unique_question_ids=len({r['question_id'] for r in rows}), cases=len(cases),
        questions_per_case_histogram=dict(sorted(counts.items())),
        multi_question_cases=sum(len(v) > 1 for v in cases.values()),
        questions_in_multi_question_cases=sum(len(v) for v in cases.values() if len(v) > 1),
        image_reference_occurrences=occurrences, unique_image_references=len(images),
        case_scoped_unique_image_references=len(scoped), distinct_ordered_image_sets=len(sets),
        unique_case_image_set_exact_question_keys=len(query_keys),
        no_question_conditioning_hypothetical_hits=occurrences-len(scoped),
        exact_question_key_hypothetical_hits=len(rows)-len(query_keys),
        cases_with_single_ordered_image_set=sum(len({r['images'] for r in v}) == 1 for v in cases.values()),
        cases_with_multiple_ordered_image_sets=sum(len({r['images'] for r in v}) > 1 for v in cases.values()),
        labels_used=False, real_tool_calls=0, image_bytes_verified=False)


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1]), indent=2))
