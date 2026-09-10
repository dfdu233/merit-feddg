"""Optional local BiomedParse v1 soft masks, with frozen released weights.

Official reference: microsoft/BiomedParse, db5c10782dab2377db4f68bbc03f71c54572e51b,
example_prediction.py and inference_utils/inference.py. No target masks, target
statistics, source fitting, automatic downloads or generated reports are used.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..capabilities import CapabilityResult, EvidenceItem
from ..native_evidence import _tokens
from ..spatial_evidence import encode_soft_mask
from .base import load_rgb

V1_COMMIT = "db5c10782dab2377db4f68bbc03f71c54572e51b"


class BiomedParseCapabilityExpert:
    def __init__(self, model_id, expert_id, source_path, groups_by_modality,
                 scope="biomedical_objects", max_prompts=8, group_requirements=None):
        if type(max_prompts) is not int or max_prompts < 1:
            raise ValueError("max_prompts must be a positive integer")
        self.checkpoint = Path(model_id).expanduser().resolve()
        self.source = Path(source_path).expanduser().resolve()
        self.expert_id, self.scope = expert_id, scope
        self.groups = groups_by_modality
        self.group_requirements = group_requirements or {}
        self.max_prompts, self.model = max_prompts, None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from PIL import Image

        # Detectron2 at the BiomedParse v1 revision still imports the historical
        # Pillow interpolation alias, which was removed in newer Pillow builds.
        # Keep the compatibility shim local to this optional expert instead of
        # downgrading the shared LLaVA-Med environment.
        if not hasattr(Image, "LINEAR"):
            Image.LINEAR = Image.BILINEAR

        if not self.checkpoint.is_file():
            raise FileNotFoundError("BiomedParse requires an existing local v1 checkpoint file")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise ValueError("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before starting inference")
        if not torch.cuda.is_available():
            raise RuntimeError("Official BiomedParse v1 inference requires CUDA")
        revision = subprocess.check_output(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True, timeout=10).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(self.source), "status", "--porcelain", "--untracked-files=no"],
            text=True, timeout=10).strip()
        if revision != V1_COMMIT or dirty:
            raise ValueError("BiomedParse source must be the clean pinned v1 revision")
        # Official v1 modules use top-level imports and relative config paths.
        # Refuse module collisions rather than silently load a different package.
        for name in ("modeling", "utilities", "inference_utils"):
            module = sys.modules.get(name)
            if module is not None and not Path(module.__file__).resolve().is_relative_to(self.source):
                raise RuntimeError(f"BiomedParse import collision: {name}")
        sys.path.insert(0, str(self.source))
        previous = Path.cwd()
        try:
            os.chdir(self.source)
            from inference_utils.inference import interactive_infer_image
            from modeling import build_model
            from modeling.BaseModel import BaseModel
            from utilities.arguments import load_opt_from_config_files
            from utilities.constants import BIOMED_CLASSES, BIOMED_OBJECTS
            from utilities.distributed import init_distributed

            opt = init_distributed(load_opt_from_config_files(["configs/biomedparse_inference.yaml"]))
            model = BaseModel(opt, build_model(opt)).from_pretrained(str(self.checkpoint))
            model = model.eval().cuda().requires_grad_(False)
            with torch.inference_mode():
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    BIOMED_CLASSES + ["background"], is_eval=True)
            self.model, self.objects, self.predict = model, BIOMED_OBJECTS, interactive_infer_image
        finally:
            os.chdir(previous)
            sys.path.remove(str(self.source))

    def infer(self, request):
        if request.capability != "segmentation" or request.scope != self.scope:
            return CapabilityResult(self.expert_id, request.capability, (), "wrong_capability_or_scope")
        if request.modality not in self.groups or request.region is not None:
            return CapabilityResult(self.expert_id, request.capability, (), "unsupported_modality_or_region")
        query = _tokens(request.question)
        groups = [group for group in self.groups[request.modality]
                  if all(query.intersection(alternatives)
                         for alternatives in self.group_requirements.get(group, []))]
        if not groups:
            return CapabilityResult(self.expert_id, request.capability, (), "unknown_anatomy_or_sequence")
        self._load()
        # Group names are explicit protocol configuration, not inferred MRI sequences.
        prompts = sorted({target for group in groups
                          for target in self.objects[group]})
        query = _tokens(request.question)
        prompts.sort(key=lambda label: (-len(_tokens(label) & query), label))
        omitted = prompts[self.max_prompts:]
        prompts = prompts[:self.max_prompts]
        image = load_rgb(request.image)
        import torch

        with torch.inference_mode():
            masks = np.asarray(self.predict(self.model, image, prompts))
        if masks.shape != (len(prompts), image.height, image.width):
            raise ValueError("BiomedParse returned unexpected mask geometry")
        structures = [{"label": label, "soft_mask": encode_soft_mask(mask),
                       "mask_coordinate_system": "original_image",
                       "object_presence_confirmed": False}
                      for label, mask in zip(prompts, masks)]
        item = EvidenceItem(f"{self.expert_id}:{request.sample_id}:soft-masks", self.expert_id,
                            "segmentation", self.scope, {"structures": structures,
                            "prompt_budget_omitted": omitted, "selected_groups": groups,
                            "mask_probabilities_calibrated": False},
                            provenance={"adapter": "biomedparse_v1", "source_commit": V1_COMMIT,
                                        "target_masks_used": False, "target_statistics_used": False,
                                        "weights_frozen": True, "query_used": True})
        return CapabilityResult(self.expert_id, request.capability, (item,))
