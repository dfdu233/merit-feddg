"""Strict, training-free adapter smoke on an explicitly supplied TRAIN image.

Emits engineering metadata only, not the image, patient text or class scores.
This does not establish accuracy, split independence or downstream VQA benefit.
"""
import argparse
import json
import time

import torch

from merit_feddg.capabilities import CapabilityRequest, validate_result
from merit_feddg.experts.native_small_medical import MedMNISTExpert, UKANExpert


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['breastmnist', 'ukan'], required=True)
    p.add_argument('--bundle', required=True)
    p.add_argument('--image', required=True, help='Preselected TRAIN image only')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    start = time.perf_counter()
    if args.kind == 'breastmnist':
        cls, capability, scope = MedMNISTExpert, 'classification', 'breast_ultrasound_binary'
    else:
        cls, capability, scope = UKANExpert, 'segmentation', 'breast_ultrasound_mask'
    expert = cls(args.bundle, expert_id=args.kind, scope=scope, device=args.device)
    initialized = time.perf_counter()
    request = CapabilityRequest('train-engineering-smoke', args.image, 'breast',
        'ultrasound', 'open_vqa', 'BUSI-derived', 'unverified-patient', capability, scope=scope)
    result = validate_result(expert.infer(request), args.kind, request)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    finished = time.perf_counter()
    if not result.items:
        raise RuntimeError('no native evidence: ' + result.reason)
    if any(p.requires_grad for p in expert.model.parameters()):
        raise RuntimeError('unfrozen expert parameter')
    mask = result.items[0].payload.get('mask')
    mask_audit = None if mask is None else {
        'size_hw': mask['size'], 'foreground_pixels': sum(mask['counts'][1::2]),
        'region_available': sum(mask['counts'][1::2]) > 0}
    print(json.dumps({'kind': args.kind, 'strict_load': True, 'all_frozen': True,
        'native_items': len(result.items), 'capability': capability,
        'checkpoint_sha256': expert.bundle['checkpoint_sha256'],
        'initialization_seconds': initialized-start, 'inference_seconds': finished-initialized,
        'device': args.device, 'mask_audit': mask_audit, 'medical_accuracy_assessed': False,
        'patient_split_independence_verified': False}, indent=2))


if __name__ == '__main__':
    main()
