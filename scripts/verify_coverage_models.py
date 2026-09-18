"""Real native inference smoke; no reference answers and no efficacy scoring."""
import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

from merit_feddg.capabilities import CapabilityRequest, validate_result
from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.expert_coverage import load_config, new_specs
from merit_feddg.pathology_pilot import file_sha, write_new


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/expert_coverage_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    from run_quilt_worker import assert_single_gpu
    assert_single_gpu(torch, args.gpu_uuid)
    torch.set_num_threads(4)
    config = load_config(args.config)
    specs = new_specs(config, 'cuda:0')
    source = Path(config['unimed_source']) / 'docs/sample_images'
    # Author-provided demo images: engineering only, not held-out evidence.
    images = {'unimed_ct_anatomy': source / 'ct_scan_right_kidney.tiff',
              'unimed_lung_histology': source / 'tumor_histo_pathology.jpg',
              'flair_retina': Path(config['flair_source']) / 'documents/sample_macular_hole.png'}
    # First pre-existing inferred dermatology image; no answer/score-based selection.
    base = Path('/home/dbw/ANCHOR/corrected_runs/paper_baselines_v1/merit_common_protocol_v1')
    donor = base / 'pathvqa/a095b918c9a484530e90b3d1c47f7590ed5295c1ac01ac40f8c0eaba8744ac70'
    routes = json.loads((donor / 'routing.json').read_text())
    manifest = [json.loads(line) for line in (base / 'pathvqa.jsonl').read_text().splitlines()]
    images['monet_skin'] = Path(next(r['image'] for r in manifest
                                    if routes[r['id']]['modality'] == 'dermatology'))
    records = []
    for name, image in images.items():
        pool = CapabilityPool({name: specs[name]}, 'artifacts')
        spec = specs[name]
        request = CapabilityRequest('engineering-smoke', str(image),
                                    'Return the configured native catalog similarities.',
                                    spec['modalities'][0], 'open_vqa', 'engineering',
                                    file_sha(image), 'classification', scope=spec['scope'])
        start = time.perf_counter()
        result = validate_result(pool.infer(name, request), name, request)
        assert result.items and all(item.confidence is None for item in result.items)
        model = next(iter(pool.models.values()))
        record = {'expert': name, 'request': asdict(request), 'result': asdict(result),
                  'wall_seconds': time.perf_counter() - start, 'cost': model.last_cost,
                  'all_parameters_frozen': all(not p.requires_grad for p in model.backend.model.parameters()),
                  'held_out_accuracy_evaluated': False, 'gpu_uuid': args.gpu_uuid,
                  'torch': torch.__version__}
        write_new(args.output / (name + '.json'), record)
        records.append({k: v for k, v in record.items() if k not in ('request', 'result')})
        print('NATIVE_OK', name, record['cost'], flush=True)
        del pool, model
        gc.collect()
        torch.cuda.empty_cache()
    write_new(args.output / 'summary.json', {'complete': True, 'models': records,
                                          'medical_benefit_established': False})


if __name__ == '__main__':
    main()
