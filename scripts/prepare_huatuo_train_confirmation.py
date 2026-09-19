"""Offline label separation and image-disjoint scheduling; no score-based selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/dbw/ANCHOR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor.corrected_sgta.protocol_v2 import build_prompt

from merit_feddg.open_data import pixel_digest
from merit_feddg.open_study import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['vqa_rad', 'slake'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-schedule', type=Path,
                        help='Exclude all pixels of an already inspected TRAIN schedule')
    parser.add_argument('--max-images', type=int,
                        help='Fixed scheduling count, never selected by scores')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError('Positive fixed image count required')
    if args.output.exists():
        raise FileExistsError('Preserve frozen scheduling inputs')
    cache = {}
    def image_info(path):
        path = str(path)
        if path not in cache:
            cache[path] = (digest(path), pixel_digest(path))
        return cache[path]
    rows, refs, excluded = [], {}, set()
    if args.dataset == 'vqa_rad':
        data = Path('/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data')
        train, test = data/'train/manifest.jsonl', data/'test/manifest.jsonl'
        raw = [json.loads(s) for s in train.read_text().splitlines()]
        tests = [json.loads(s) for s in test.read_text().splitlines()]
        refs = json.loads((data/'train/references.json').read_text())
        used = set(json.loads(Path('/home/dbw/merit-feddg-anchored-revision/runs/pilot-v4/frozen.json').read_text())['selected'])
        excluded = {image_info(r['image'])[1] for r in tests}
        excluded |= {image_info(r['image'])[1] for r in raw if r['id'] in used}
        for r in raw:
            byte, pixels = image_info(r['image'])
            rows.append({'id':r['id'],'image':r['image'],'image_sha256':byte,'pixel_sha256':pixels,
                'question':r['question'],'answer_type':r['answer_type'],'task':'open_vqa',
                'benchmark_prompt':build_prompt({'question':r['question'],
                    'question_type':'binary' if r['answer_type']=='closed' else 'open'})})
        limit = None
    else:
        train, test = Path('/home/dbw/data/SLAKE/train.json'), Path('/home/dbw/data/SLAKE/test.json')
        images = Path('/home/dbw/ANCHOR/data/medheval/images/Slake')
        raw, tests = json.loads(train.read_text()), json.loads(test.read_text())
        excluded = {image_info(images/r['img_name'])[1] for r in tests}
        for r in raw:
            key = 'slake-train-' + str(r['qid'])
            byte, pixels = image_info(images/r['img_name'])
            # Match the existing formal SLAKE builder: source has no question_type.
            # Do not turn every CLOSED label into a binary generation instruction.
            prompt = build_prompt({'question':r['question']})
            rows.append({'id':key,'image':str(images/r['img_name']), 'image_sha256':byte,
                'pixel_sha256':pixels,'question':r['question'],'answer_type':r['answer_type'].lower(),
                'q_lang':r['q_lang'],'task':'open_vqa','benchmark_prompt':prompt})
            refs[key] = [str(r['answer'])]
        limit = 64
    if args.exclude_schedule:
        previous = json.loads(args.exclude_schedule.read_text())
        if previous['dataset'] != args.dataset:
            raise ValueError('Excluded schedule dataset mismatch')
        used_ids = set(previous['ids'])
        if not used_ids <= {r['id'] for r in rows}:
            raise ValueError('Excluded schedule contains unknown TRAIN IDs')
        excluded |= {r['pixel_sha256'] for r in rows if r['id'] in used_ids}
    if args.max_images is not None:
        limit = args.max_images
    groups = {}
    for row in rows:
        if row['pixel_sha256'] not in excluded:
            groups.setdefault(row['pixel_sha256'], []).append(row)
    selected = [min(group,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())['id']
                for _,group in sorted(groups.items())]
    if args.max_images is not None and len(selected) < args.max_images:
        raise ValueError('Insufficient unused TRAIN images; do not silently shrink')
    if limit:
        selected = selected[:limit]
    if len(refs) != len(rows) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Full source ID/reference alignment failed')
    args.output.mkdir(parents=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic_json(args.output/'references.json', refs)
    atomic_json(args.output/'schedule.json', {'ids':selected,'dataset':args.dataset,
        'full_train_count':len(rows),'eligible_unique_images':len(groups),
        'selection':'one minimum SHA256(id) question per eligible pixel image; image-hash order; no scores',
        'excluded_test_and_used_image_count':len(excluded),'patient_disjoint':'unknown',
        'source_train_sha256':digest(train),'source_test_sha256':digest(test)})
    if args.exclude_schedule:
        atomic_json(args.output/'confirmation-exclusion.json', {
            'excluded_schedule_sha256': digest(args.exclude_schedule),
            'excluded_schedule': str(args.exclude_schedule.resolve()),
            'frozen_count': limit, 'method': 'pixel exclusion before hash-order scheduling',
            'labels_used_for_selection': False})
    print(args.dataset, 'full_manifest',len(rows),'scheduled',len(selected),'images',len(groups))


if __name__ == '__main__':
    main()
