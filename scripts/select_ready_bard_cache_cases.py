"""Select only full-manifest rows whose frozen native requests are already cached.

This is an answer-blind scheduling aid for an incomplete native cache. It does
not change the benchmark denominator or publish a partial score.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import load_cached, load_manifest
from merit_feddg.open_study import atomic_json, fingerprint
from prepare_bard_expert_cache import make_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    cache = Path(args.cache)
    protocol = json.loads((cache / 'protocol.json').read_text())
    config = load_experiment_yaml(args.config)
    rows = load_manifest(args.manifest)
    if not protocol['cache_only'] or len(rows) != protocol['n']:
        raise ValueError('Frozen cache/manifest identity or size differs')
    if config['experts'] != protocol['config']['experts']:
        raise ValueError('Native expert specifications differ')
    routes = json.loads((cache / 'routing.json').read_text())
    schedule = json.loads((cache / 'schedule.json').read_text())
    if set(routes) != {row['id'] for row in rows} or set(schedule) != set(routes):
        raise ValueError('Frozen routing/schedule does not cover the exact manifest')
    decoder = ValueGenerationConfig(**config['capability_value']['generation'])

    ready = []
    missing_cache = 0
    missing_image = 0
    for index, row in enumerate(rows):
        if not Path(row['image']).is_file():
            missing_image += 1
            continue
        check_row = dict(row, modality=routes[row['id']]['modality'], group_id=row['image_sha256'])
        directory = cache / 'expert-cache' / fingerprint(row['id'])
        for descriptor in schedule[row['id']]:
            request = make_request(check_row, descriptor, decoder)
            key = fingerprint(['infer', descriptor['expert'], asdict(request)])
            if load_cached(directory / f'{key}.json', protocol['identity']) is None:
                missing_cache += 1
                break
        else:
            ready.append({'index': index, 'id': row['id']})

    atomic_json(args.output, {
        'schema': 'bard-ready-native-cases-v1',
        'cache_identity': protocol['identity'],
        'cache_shards_complete': protocol['shards_complete'],
        'manifest': args.manifest,
        'denominator': len(rows),
        'ready_count': len(ready),
        'missing_native_rows': missing_cache,
        'missing_image_rows': missing_image,
        'ready': ready,
        'partial_outputs_are_not_full_scores': True,
    })
    print(args.output, len(ready), '/', len(rows), 'missing_native', missing_cache,
          'missing_image', missing_image, flush=True)


if __name__ == '__main__':
    main()
