"""Frozen, local-only UniMed-CLIP / FLAIR / MONET catalog observations.

These are relative image/text matches, not diagnoses or calibrated probabilities.
The fixed catalog is configuration-owned, never constructed from answer choices.
Existing CapabilityPool factory and EvidenceItem interfaces are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from weakref import WeakValueDictionary

import numpy as np
from PIL import Image

from ..capabilities import CapabilityResult, EvidenceItem
from .native_small_medical import source_module

_BACKBONES = WeakValueDictionary()
PINS = {
    'monet': ('chanwkim/monet', '1d1efd0b8d61bde82cde22d2e93c28d73441f29e'),
    'flair': ('jusiro2/FLAIR', '5f6bdd0a068353dc41a896ba3abdd7c0f6d35938'),
    'unimed': ('UzairK/unimed-clip-vit-b16', '84a67cb0331b511b630136e4878beaa5a3fdc657'),
    'clinicalbert': ('emilyalsentzer/Bio_ClinicalBERT', 'd5892b39a4adaed74b92212a44081509db72f87b'),
    'biomedbert': ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                   'd673b8835373c6fa116d6d8006b33d48734e305d'),
}
SOURCE_PINS = {
    'flair': 'd6652d53389ff49e5f73efaccf4246e9de88d1a3',
    'unimed': '536d1651ab28a68a85af07b0ad1a3c56de34e14c',
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_resource(root, kind):
    root = Path(root).resolve()
    audit = json.loads((root / 'resource.json').read_text())
    if (audit.get('repo_id'), audit.get('revision')) != PINS[kind]:
        raise ValueError('unreviewed resource identity')
    if not audit.get('publisher_sha256'):
        raise ValueError('publisher weight hashes required')
    for filename, digest in audit['files_sha256'].items():
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError('local resource hash mismatch')
    for filename, digest in audit['publisher_sha256'].items():
        if audit['files_sha256'].get(filename) != digest:
            raise ValueError('publisher weight identity mismatch')
    return audit


def verify_source(path, kind):
    import subprocess
    path = Path(path).resolve()
    sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != SOURCE_PINS[kind]:
        raise ValueError('upstream source revision mismatch')
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain',
                                     '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError('modified upstream source rejected')


def strict_state(model, state):
    """Only verify/remove legacy deterministic position buffers, never weights."""
    import torch
    expected = model.state_dict()
    state = dict(state)
    compatibility = []
    for name in set(state) - set(expected):
        if not name.endswith('.embeddings.position_ids'):
            continue
        embedding = model.get_submodule(name.rsplit('.', 1)[0])
        native = embedding.position_ids.detach().cpu()
        if not torch.equal(state[name].cpu(), native):
            raise ValueError('legacy position buffer differs from runtime positions')
        del state[name]
        compatibility.append(name)
    model.load_state_dict(state, strict=True)
    return compatibility


class _Backbone:
    def __init__(self, root, kind, device, source_path, text_path):
        import torch
        self.torch, self.kind, self.device = torch, kind, torch.device(device)
        self.audit = verify_resource(root, kind)
        self.text_cache = {}
        start = perf_counter()
        if kind == 'monet':
            from transformers import AutoProcessor, CLIPModel
            self.processor = AutoProcessor.from_pretrained(root, local_files_only=True,
                                                          use_fast=False)
            self.model, info = CLIPModel.from_pretrained(
                root, local_files_only=True, output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
                raise ValueError(f'incomplete MONET checkpoint: {info}')
        elif kind == 'flair':
            from safetensors.torch import load_file
            verify_resource(text_path, 'clinicalbert')
            verify_source(source_path, kind)
            module = source_module(Path(source_path) / 'flair/modeling/model.py')
            # Upstream uses a module device, not an explicit forward argument.
            module.device = str(self.device)
            config = json.loads((Path(root) / 'config.json').read_text())
            # Final checkpoint contains ALL trained towers. No ImageNet re-download.
            config.update(vision_pretrained=False, bert_type=str(Path(text_path).resolve()),
                          from_checkpoint=False)
            self.model = module.FLAIRModel(**config)
            self.compatibility = strict_state(
                self.model, load_file(str(Path(root) / 'model.safetensors')))
        elif kind == 'unimed':
            verify_resource(text_path, 'biomedbert')
            verify_source(source_path, kind)
            module = source_module(Path(source_path) / 'src/open_clip/__init__.py')
            mean, std = module.get_mean_std()
            self.model, _, self.processor = module.create_model_and_transforms(
                'ViT-B-16-quickgelu', pretrained='', precision='fp32', device='cpu',
                force_quick_gelu=True, pretrained_image=False, mean=mean, std=std,
                inmem=True, text_encoder_name=str(Path(text_path).resolve()))
            checkpoint = torch.load(Path(root) / 'unimed-clip-vit-b16.pt',
                                    map_location='cpu', weights_only=False)
            state = checkpoint.get('state_dict', checkpoint)
            if all(k.startswith('module.') for k in state):
                state = {k[7:]: v for k, v in state.items()}
            self.compatibility = strict_state(self.model, state)
            self.tokenizer = module.HFTokenizer(str(Path(text_path).resolve()), context_length=256)
            del checkpoint, state
        else:
            raise ValueError('unsupported backbone')
        self.model.to(self.device).eval().requires_grad_(False)
        self.load_seconds = perf_counter() - start
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('expert must be frozen')

    def scores(self, image, prompts):
        torch = self.torch
        with torch.inference_mode():
            if self.kind == 'flair':
                # Full prompt supplied once; official preprocessing/forward preserved.
                _, logits = self.model(np.array(image), list(prompts))
                scale = float(self.model.logit_scale.exp())
                return np.asarray(logits[0]) / scale, np.asarray(logits[0])
            if self.kind == 'monet':
                inputs = self.processor(images=image, text=list(prompts), padding=True,
                                        truncation=False, return_tensors='pt').to(self.device)
                if inputs['input_ids'].shape[-1] > 77:
                    raise ValueError('fixed MONET prompt exceeds native context')
                output = self.model(**inputs)
                cosine = (output.image_embeds @ output.text_embeds.T)[0]
                logits = output.logits_per_image[0]
            else:
                key = tuple(prompts)
                if key not in self.text_cache:
                    tokens = self.tokenizer(list(prompts)).to(self.device)
                    features = self.model.encode_text(tokens)
                    self.text_cache[key] = torch.nn.functional.normalize(features, dim=-1)
                pixels = self.processor(image)[None].to(self.device)
                features = torch.nn.functional.normalize(self.model.encode_image(pixels), dim=-1)
                cosine = (features @ self.text_cache[key].T)[0]
                logits = cosine * self.model.logit_scale.exp()
            return cosine.float().cpu().numpy(), logits.float().cpu().numpy()


class CatalogExpert:
    def __init__(self, model_id, *, kind, expert_id, scope, modalities, catalog,
                 device='cpu', source_path=None, text_path=None, source_family='unknown'):
        if kind not in ('flair', 'unimed', 'monet') or not catalog or not modalities:
            raise ValueError('reviewed kind, modality and fixed catalog required')
        self.catalog = tuple((x['name'], x['prompt']) for x in catalog)
        if len({x[0] for x in self.catalog}) != len(self.catalog):
            raise ValueError('duplicate catalog concepts')
        if any(not n.strip() or not p.strip() for n, p in self.catalog):
            raise ValueError('empty concept or prompt')
        self.expert_id, self.scope, self.modalities = expert_id, scope, tuple(modalities)
        self.kind, self.source_family = kind, source_family
        key = (str(Path(model_id).resolve()), kind, device, source_path, text_path)
        backend = _BACKBONES.get(key)
        self.new_load = backend is None
        if backend is None:
            backend = _Backbone(*key)
            _BACKBONES[key] = backend
        self.backend = backend
        self.last_cost = {}

    def infer(self, request):
        if (request.capability != 'classification' or request.scope != self.scope
                or request.modality not in self.modalities or request.region is not None):
            return CapabilityResult(self.expert_id, request.capability, (), 'unsupported_request')
        torch = self.backend.torch
        if self.backend.device.type == 'cuda':
            torch.cuda.synchronize(self.backend.device)
        start = perf_counter()
        with Image.open(request.image) as image:
            if getattr(image, 'n_frames', 1) != 1:
                return CapabilityResult(self.expert_id, request.capability, (), 'requires_2d_image')
            scores, logits = self.backend.scores(image.convert('RGB'), [p for _, p in self.catalog])
        if self.backend.device.type == 'cuda':
            torch.cuda.synchronize(self.backend.device)
        if (np.shape(scores) != (len(self.catalog),) or np.shape(logits) != np.shape(scores)
                or not np.isfinite(scores).all() or not np.isfinite(logits).all()):
            raise ValueError('invalid native catalog output')
        self.last_cost = {'inference_seconds': perf_counter() - start, 'model_calls': 1,
                          'load_seconds': self.backend.load_seconds if self.new_load else 0.0}
        self.new_load = False
        entries = [{'concept': name, 'similarity': float(score), 'native_logit': float(logit)}
                   for (name, _), score, logit in zip(self.catalog, scores, logits)]
        item = EvidenceItem(
            f'{self.expert_id}:{request.sample_id}:catalog', self.expert_id, 'classification',
            self.scope, {'catalog': entries, 'score_semantics': 'relative_image_text_similarity',
                         'catalog_exhaustive': False, 'unlisted_concepts': 'unknown'},
            summary='Fixed-catalog visual matches, not calibrated diagnosis probabilities.',
            confidence=None, provenance={'adapter': f'native_{self.kind}',
                'revision': self.backend.audit['revision'], 'source_family': self.source_family,
                'query_used': False, 'target_candidates_used': False, 'spatial_scope': 'whole_image'})
        return CapabilityResult(self.expert_id, request.capability, (item,))
