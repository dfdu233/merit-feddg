"""Image-type inference and conservative question-only applicability metadata.

This is modality/task matching, NOT OOD or a domain generalization guarantee.
Predicted image types must be audited separately from oracle image-type metadata.
"""

from __future__ import annotations

import re
from time import perf_counter

IMAGE_TYPES = {
    # Natural labels are deliberate. A real LLaVA-Med can name these modalities,
    # while the old arbitrary 0..13 codebook collapsed to the final option even
    # for an unmistakable chest radiograph. Decoding remains a finite whitelist.
    "unknown": ("unknown", "Unknown, ambiguous or outside this list"),
    "histopathology": ("pathology", "Microscopic histopathology tissue section"),
    "gross specimen": ("gross_pathology", "Macroscopic gross specimen or dissected organ"),
    "chest X-ray": ("cxr", "Chest X-ray radiograph"),
    "non-chest X-ray": ("xray", "Non-chest X-ray radiograph, for example a bone radiograph"),
    "CT": ("ct", "Computed tomography (CT)"),
    "MRI": ("mri", "Magnetic resonance imaging (MRI)"),
    "ultrasound": ("ultrasound", "Ultrasound sonography"),
    "OCT": ("oct", "Optical coherence tomography (OCT)"),
    "fundus photograph": ("fundus", "Retinal fundus photograph"),
    "skin photograph": ("dermatology", "Clinical close-up skin or dermoscopy image"),
    "endoscopy": ("endoscopy", "Endoscopic image"),
    "clinical photograph": ("clinical_photo", "Other clinical photograph of a person or body part"),
    "diagram": ("diagram", "Teaching illustration, schematic or chart"),
}


def infer_image_type(probe, row):
    prompt = (
        "Identify only the visual medical image type. Do not diagnose. Select exactly one "
        "matching image-type phrase."
    )
    started = perf_counter()
    output = probe.generate_with_usage(
        row["image"], prompt, max_new_tokens=8, allowed_texts=list(IMAGE_TYPES)
    )
    selected = output["text"].strip()
    modality = IMAGE_TYPES.get(selected, IMAGE_TYPES["unknown"])[0]
    return {
        "modality": modality, "selected_type": modality, "raw_action": selected,
        "method": "model_inferred", "seconds": perf_counter() - started,
        "input_tokens": output["input_tokens"], "output_tokens": output["output_tokens"],
        "warning": "Predicted applicability metadata, not a validated OOD detector.",
    }


def question_type(question):
    """Transparent English pilot rules; unknown does not imply an incompatible tool."""
    text = re.sub(r"\s+", " ", question.lower()).strip()
    patterns = (
        ("anatomy", r"\b(what|which) (organ|body part|anatomical structure|system)\b"),
        ("location", r"\b(where|location|located|which side)\b"),
        ("count", r"\b(how many|number of|count)\b"),
        ("measurement", r"\b(size|diameter|volume|area|how large|how big|measure)\b"),
        ("tissue", r"\b(type of tissue|tissue type|what tissue|which tissue)\b"),
        ("diagnosis", r"\b(diagnosis|disease|disorder|syndrome|carcinoma)\b"),
        ("appearance", r"\b(color|colour|appearance|morphology|stain|shape|pattern)\b"),
        ("explanation", r"\b(explain|describe|why|report)\b"),
    )
    return next((name for name, pattern in patterns if re.search(pattern, text)), "unknown")
