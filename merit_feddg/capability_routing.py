"""Image-type inference and conservative question-only applicability metadata.

This is modality/task matching, NOT OOD or a domain generalization guarantee.
Predicted image types must be audited separately from oracle image-type metadata.
"""

from __future__ import annotations

import re
from time import perf_counter

IMAGE_TYPES = {
    "0": ("unknown", "Unknown, ambiguous or outside this list"),
    "1": ("pathology", "Microscopic histopathology tissue section"),
    "2": ("gross_pathology", "Macroscopic gross specimen or dissected organ"),
    "3": ("cxr", "Chest X-ray radiograph"),
    "4": ("xray", "Non-chest X-ray radiograph, for example a bone radiograph"),
    "5": ("ct", "Computed tomography (CT)"),
    "6": ("mri", "Magnetic resonance imaging (MRI)"),
    "7": ("ultrasound", "Ultrasound sonography"),
    "8": ("oct", "Optical coherence tomography (OCT)"),
    "9": ("fundus", "Retinal fundus photograph"),
    "10": ("dermatology", "Clinical close-up skin or dermoscopy image"),
    "11": ("endoscopy", "Endoscopic image"),
    "12": ("clinical_photo", "Other clinical photograph of a person or body part"),
    "13": ("diagram", "Teaching illustration, schematic or chart"),
}


def infer_image_type(probe, row):
    prompt = (
        "Identify only the visual image type. Do not diagnose. Select exactly one numeric ID; "
        "choose 0 if unsure. A gross specimen is not a microscopic tissue section.\n"
        + "\n".join(f"{key}: {value[1]}" for key, value in IMAGE_TYPES.items())
    )
    started = perf_counter()
    output = probe.generate_with_usage(
        row["image"], prompt, max_new_tokens=8, allowed_texts=list(IMAGE_TYPES)
    )
    selected = output["text"].strip()
    modality = IMAGE_TYPES.get(selected, IMAGE_TYPES["0"])[0]
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
