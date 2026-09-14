"""Local-only official-architecture adapters. Missing audited bundles fail before init.

The bundle is a reviewed JSON manifest, not a checkpoint from generated text.
No download, training or permissive/partial state loading is supported.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..capabilities import CapabilityResult, EvidenceItem
from ..capability_experts import encode_binary_mask


def load_bundle(path, kind):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('audited bundle/official weights missing; no random initialization')
    bundle = json.loads(path.read_text())
    kinds = (kind,) if isinstance(kind, str) else kind
    if bundle.get('kind') not in kinds or not bundle.get('source_url', '').startswith('https://github.com/'):
        raise ValueError('bundle kind and official source required')
    for key in ('checkpoint', 'source_file'):
        file = Path(bundle[key])
        if not file.is_file():
            raise FileNotFoundError(f'{key} unavailable')
        if hashlib.sha256(file.read_bytes()).hexdigest() != bundle[key + '_sha256']:
            raise ValueError(f'{key} digest mismatch')
    if not bundle.get('training_split') or not bundle.get('source_revision'):
        raise ValueError('training split and source revision required')
    for name, expected in bundle.get('related_source_sha256', {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('related source digest mismatch')
    return bundle


def source_module(path):
    path = Path(path).resolve()
    name = 'merit_audited_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class MedMNISTExpert:
    """Minimal official BreastMNIST ResNet18 28px variant, not arbitrary ResNet weights."""
    def __init__(self, model_id, expert_id='breastmnist', scope='breast_ultrasound_binary', device='cpu'):
        self.bundle = b = load_bundle(model_id, 'medmnist-breast-resnet18-28')
        if b.get('labels') != ['malignant', 'normal, benign'] or b.get('training_split') != 'train':
            raise ValueError('official BreastMNIST label order and TRAIN checkpoint required')
        if b['source_url'] != 'https://github.com/MedMNIST/experiments':
            raise ValueError('official MedMNIST experiments source required')
        import torch
        self.torch, self.device = torch, device
        if type(b.get('as_rgb')) is not bool:
            raise ValueError('official checkpoint as_rgb setting must be explicit')
        self.model = source_module(b['source_file']).ResNet18(in_channels=3 if b['as_rgb'] else 1, num_classes=2)
        self.model.load_state_dict(torch.load(b['checkpoint'], map_location='cpu', weights_only=True)['net'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.expert_id, self.scope = expert_id, scope

    def infer(self, request):
        if request.capability != 'classification' or request.scope != self.scope or request.modality != 'ultrasound' or request.region:
            return CapabilityResult(self.expert_id, request.capability, (), 'unsupported_request')
        with Image.open(request.image) as im:
            # Official benchmark input is already a 28px grayscale dataset image.
            # Never silently resize an arbitrary clinical image into that benchmark.
            if im.size != (28, 28):
                return CapabilityResult(self.expert_id, request.capability, (), 'requires_native_28px_input')
            x = np.asarray(im.convert('RGB' if self.bundle['as_rgb'] else 'L'), dtype=np.float32) / 255
        x = x.transpose(2,0,1) if x.ndim == 3 else x[None]
        t = self.torch.from_numpy((x - .5) / .5)[None].to(self.device)
        with self.torch.inference_mode():
            logits = self.model(t)
            if logits.shape != (1, 2) or not self.torch.isfinite(logits).all():
                raise ValueError('invalid classification output')
            scores = logits.softmax(-1)[0].cpu().tolist()
        item = EvidenceItem(request.sample_id + ':breastmnist', self.expert_id, 'classification',
            self.scope, {'catalog': [{'concept': name, 'score': score} for name, score in
                                    zip(self.bundle['labels'], scores)],
                        'score_semantics': 'uncalibrated_softmax'}, confidence=None,
            provenance={'checkpoint_sha256': self.bundle['checkpoint_sha256'],
                        'source_family': 'BUSI', 'split': self.bundle['training_split'],
                        'preprocessing': 'native_28px; /255; normalize(.5,.5)',
                        'as_rgb':self.bundle['as_rgb']})
        return CapabilityResult(self.expert_id, request.capability, (item,))


class UKANExpert:
    """Official Seg_UKAN BUSI binary-mask variant; geometry is not a diagnosis."""
    def __init__(self, model_id, expert_id='ukan_busi', scope='breast_ultrasound_mask', device='cpu'):
        self.bundle = b = load_bundle(model_id, ('ukan-busi', 'ukan-busi-ubench'))
        source = ('https://github.com/FengheTan9/U-Bench' if b['kind'] == 'ukan-busi-ubench'
                  else 'https://github.com/CUHK-AIM-Group/U-KAN')
        if b['source_url'] != source or b.get('preprocessing') != 'official_bgr_normalize_then_div255':
            raise ValueError('audited official U-KAN preprocessing required')
        if b['kind'] == 'ukan-busi-ubench' and not b.get('related_source_sha256'):
            raise ValueError('U-Bench relative KAN source must be pinned')
        import torch
        self.torch, self.device = torch, device
        self.model = source_module(b['source_file']).UKAN(1, 3, False, embed_dims=b['embed_dims'])
        state = torch.load(b['checkpoint'], map_location='cpu', weights_only=True)
        if b['kind'] == 'ukan-busi-ubench':
            state = state['state_dict']
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.expert_id, self.scope = expert_id, scope

    def infer(self, request):
        if request.capability != 'segmentation' or request.scope != self.scope or request.modality != 'ultrasound' or request.region:
            return CapabilityResult(self.expert_id, request.capability, (), 'unsupported_request')
        import cv2
        image = cv2.imread(str(request.image))  # Official loader uses BGR, not RGB.
        if image is None:
            raise ValueError('unreadable image')
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.bundle['input_w'], self.bundle['input_h']))
        # Reproduce val.py Normalize AND dataset.py's subsequent /255, even though
        # unusual. A different preprocessing requires a differently audited bundle.
        x = ((resized.astype(np.float32) / 255 - np.array([.485,.456,.406], dtype=np.float32))
             / np.array([.229,.224,.225], dtype=np.float32)) / 255
        t = self.torch.from_numpy(x.transpose(2,0,1).copy())[None].to(self.device)
        with self.torch.inference_mode():
            logits = self.model(t)
            if tuple(logits.shape) != (1,1,self.bundle['input_h'],self.bundle['input_w']) or not self.torch.isfinite(logits).all():
                raise ValueError('invalid segmentation output')
            mask = (logits.sigmoid()[0,0].cpu().numpy() >= .5).astype(np.uint8)
        mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
        item = EvidenceItem(request.sample_id + ':ukan', self.expert_id, 'segmentation', self.scope,
            {'mask': encode_binary_mask(mask), 'semantic_class': 'breast_foreground',
             'mask_coordinate_system': 'original_image', 'label': 'breast_foreground',
             'original_size_hw': [h,w]}, confidence=None,
            provenance={'source_family':'BUSI', 'checkpoint_sha256':self.bundle['checkpoint_sha256'],
                        'training_split':self.bundle['training_split'], 'outside_scope':'unknown',
                        'transform':{'resize_from_hw':[h,w], 'model_size_hw':[self.bundle['input_h'],self.bundle['input_w']]}})
        return CapabilityResult(self.expert_id, request.capability, (item,))
