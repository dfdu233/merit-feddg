"""Budgeted native observations and predicted overlays, never diagnostic labels.

The text compiler is a lossy *presentation* of immutable native evidence. Masks
remain in the original evidence store; a bounded, explicitly predicted view can
show their geometry to the VLM without interpreting foreground as disease.
Only the supplied query image may be opened, never a path from a tool payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .capabilities import CAPABILITIES, EvidenceItem
from .experts.base import load_rgb

_DROP_KEYS = {
    "usage", "source_image", "image", "image_path", "local_path", "path", "url",
    "checkpoint", "checkpoint_path", "model_path", "model_id", "model_fingerprint",
    "pixel_digest", "sha256", "prompt", "target_label", "target_reference", "ground_truth",
}
_LIST_KEYS = {"catalog", "findings", "references", "structures", "detections", "boxes", "objects"}
_CONTENT_KEYS = _LIST_KEYS | {"mask", "generated_text"}
_STOP_WORDS = {"a", "an", "the", "is", "are", "of", "in", "on", "to", "this", "image", "what"}
_MAX_MASK_PIXELS = 16_777_216


def _item_dict(item):
    if isinstance(item, EvidenceItem):
        return {
            "expert_id": item.expert_id, "evidence_id": item.evidence_id,
            "capability": item.capability, "scope": item.scope, "payload": item.payload,
            "provenance": item.provenance,
        }
    return dict(item) if isinstance(item, Mapping) else {}


def _valid_identity(item):
    return all(isinstance(item.get(key), str) and item[key] for key in (
        "expert_id", "evidence_id", "scope", "capability",
    )) and item["capability"] in CAPABILITIES


def _tokens(text):
    return set(re.findall(r"[\w]+", str(text).lower())) - _STOP_WORDS


def _relevance(value, question_tokens):
    if isinstance(value, Mapping):
        # Reference answers are deliberately not used to rank retrieved cases.
        text = " ".join(str(value.get(key, "")) for key in (
            "concept", "finding", "label", "anatomical_structure", "semantic_class",
            "source_question", "generated_text",
        ))
    else:
        text = str(value)
    return len(_tokens(text) & question_tokens)


def _clean(value, question_tokens):
    if isinstance(value, Mapping):
        if value.get("encoding") == "rle-row-major-zero-first":
            if _rle_components(value) is None:
                return None
            return {
                "encoding": value["encoding"], "size": _clean(value.get("size"), set()),
                "mask_data_omitted_from_text": True,
            }
        out = {}
        for key, entry in value.items():
            if not isinstance(key, str) or key in _DROP_KEYS or key.endswith("_path"):
                continue
            cleaned = _clean(entry, question_tokens)
            if cleaned is not None or entry is None:
                out[key] = cleaned
        return out
    if isinstance(value, (list, tuple)):
        entries = [_clean(entry, question_tokens) for entry in value]
        entries = [entry for entry in entries if entry is not None]
        if entries and all(isinstance(entry, dict) for entry in entries):
            entries.sort(key=lambda entry: -_relevance(entry, question_tokens))
        return entries
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return None


def _json_size(value):
    # The ordinary json.dumps rendering, not a truncated serialized substring.
    return len(json.dumps(value, allow_nan=False))


def _prunable_lists(value):
    lists = []
    if isinstance(value, dict):
        for key, entry in value.items():
            if key in _LIST_KEYS and isinstance(entry, list) and len(entry) > 1:
                lists.append(entry)
            lists.extend(_prunable_lists(entry))
    elif isinstance(value, list):
        for entry in value:
            lists.extend(_prunable_lists(entry))
    return lists


def _record_relevance(record, query):
    payload = record["payload"]
    scores = [_relevance(payload, query)]
    for key in _LIST_KEYS:
        if isinstance(payload.get(key), list):
            scores.extend(_relevance(entry, query) for entry in payload[key])
    return max(scores)


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def _mask_description(value):
    return (isinstance(value, dict) and value.get("encoding") == "rle-row-major-zero-first"
            and value.get("mask_data_omitted_from_text") is True
            and isinstance(value.get("size"), list) and len(value["size"]) == 2
            and all(type(size) is int and size > 0 for size in value["size"]))


def _has_native_content(payload, capability):
    """Metadata, empty containers, and invalid scores do not count as adoption."""
    if capability == "generation":
        return _nonempty_text(payload.get("generated_text"))
    if capability == "segmentation" and _mask_description(payload.get("mask")):
        return True
    for key in _LIST_KEYS:
        entries = payload.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if capability == "classification" and key in {"catalog", "findings"}:
                score = entry.get("similarity" if key == "catalog" else "score")
                name = entry.get("concept" if key == "catalog" else "finding")
                if (_nonempty_text(name) and type(score) in (float, int) and math.isfinite(score)):
                    return True
            elif capability == "retrieval" and key == "references":
                if (_nonempty_text(entry.get("source_question"))
                        or _nonempty_text(entry.get("source_reference"))):
                    return True
            elif capability in {"segmentation", "detection"} and key in {
                "structures", "detections", "boxes", "objects",
            }:
                if _mask_description(entry.get("mask")):
                    return True
                if (_bbox(entry.get("bbox_xyxy_normalized_original_image")) is not None
                        or _bbox(entry.get("bbox_xyxy_normalized")) is not None):
                    return True
    return False


def compile_evidence(items, question, max_chars=1800) -> list[dict]:
    """Return complete JSON records within ``max_chars`` of ``json.dumps``.

    Question-matching observations precede other entries, preserving native order
    on ties. Whole low-priority entries/records are removed to meet the budget;
    score thresholds, negative diagnoses, or new medical findings are not added.
    An individually oversized record may be omitted rather than truncated.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 2:
        raise ValueError("max_chars must be an integer of at least 2")
    query = _tokens(question)
    records = []
    for source in items:
        item = _item_dict(source)
        payload = item.get("payload")
        if not _valid_identity(item) or not isinstance(payload, Mapping):
            continue
        if item["capability"] == "generation":
            generated = payload.get("generated_text")
            if not isinstance(generated, str) or not generated.strip():
                continue
        if not any(payload.get(key) for key in _CONTENT_KEYS):
            continue
        record = {key: item[key] for key in ("expert_id", "evidence_id", "capability", "scope")}
        record["payload"] = _clean(payload, query)
        if _has_native_content(record["payload"], record["capability"]):
            records.append(record)
    records.sort(key=lambda record: -_record_relevance(record, query))
    accepted = []
    for record in records:
        while _json_size(accepted + [record]) > max_chars:
            candidates = _prunable_lists(record["payload"])
            if not candidates:
                break
            max(candidates, key=_json_size).pop()
        if (_json_size(accepted + [record]) <= max_chars
                and _has_native_content(record["payload"], record["capability"])):
            accepted.append(record)
    return accepted


def _bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in value):
        return None
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    return [float(v) for v in value]


def _rle_components(value):
    if not isinstance(value, Mapping) or value.get("encoding") != "rle-row-major-zero-first":
        return None
    size, counts = value.get("size"), value.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    if any(type(x) is not int or x <= 0 for x in size):
        return None
    pixels = size[0] * size[1]
    if pixels > _MAX_MASK_PIXELS or not isinstance(counts, (list, tuple)) or not counts:
        return None
    if len(counts) > pixels + 1 or any(type(x) is not int or x < 0 for x in counts):
        return None
    if sum(counts) != pixels or any(x == 0 for x in counts[1:]):
        return None
    return size, counts


def _decode_mask(value):
    components = _rle_components(value)
    if components is None:
        return None
    size, counts = components
    pixels = size[0] * size[1]
    flat = np.zeros(pixels, dtype=np.uint8)
    position = 0
    for index, length in enumerate(counts):
        if index % 2:
            flat[position:position + length] = 255
        position += length
    return Image.fromarray(flat.reshape(size))


def _mapped_mask(entry, payload, provenance, image_size):
    mask = _decode_mask(entry.get("mask"))
    if mask is None or mask.getbbox() is None:
        return None
    width, height = image_size
    coordinates = entry.get("mask_coordinate_system", provenance.get("mask_resolution"))
    if coordinates == "original_image":
        return mask if mask.size == image_size else None
    if coordinates != "model_grid_of_center_crop":
        return None
    transform = payload.get("image_transform")
    if not isinstance(transform, Mapping) or transform.get("original_size_hw") != [height, width]:
        return None
    crop = _bbox(transform.get("crop_box_xyxy_normalized"))
    if crop is None or transform.get("model_size_hw") != [mask.height, mask.width]:
        return None
    left, top, right, bottom = [round(v * n) for v, n in zip(crop, (width, height) * 2)]
    if right <= left or bottom <= top:
        return None
    canvas = Image.new("L", image_size)
    canvas.paste(mask.resize((right - left, bottom - top), Image.Resampling.NEAREST), (left, top))
    return canvas if canvas.getbbox() is not None else None


def _label(entry):
    for key in ("anatomical_structure", "label", "finding", "concept", "semantic_class"):
        if isinstance(entry.get(key), str) and entry[key].strip():
            return entry[key].strip()
    return "unlabelled prediction"


def _pixel_digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def _annotations(image, items, question):
    annotations = []
    for source in items:
        item = _item_dict(source)
        if not _valid_identity(item) or item["capability"] not in {"segmentation", "detection"}:
            continue
        payload, provenance = item.get("payload"), item.get("provenance", {})
        if not isinstance(payload, Mapping) or not isinstance(provenance, Mapping):
            continue
        if any(provenance.get(key) for key in ("target_mask_used", "target_masks_used", "target_annotations_used")):
            continue
        entries = [payload] if "mask" in payload else []
        for key in ("structures", "detections", "boxes", "objects"):
            if isinstance(payload.get(key), (list, tuple)):
                entries.extend(payload[key])
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            mask = _mapped_mask(entry, payload, provenance, image.size) if "mask" in entry else None
            if "mask" in entry:
                if mask is None:
                    continue  # Empty/malformed masks must not fall back to a prompt box.
                box = [v / n for v, n in zip(mask.getbbox(), image.size * 2)]
                kind = "predicted_mask"
            else:
                box = _bbox(entry.get("bbox_xyxy_normalized_original_image"))
                if box is None:
                    # This explicit field means normalized coordinates of the input image.
                    box = _bbox(entry.get("bbox_xyxy_normalized"))
                if box is None:
                    continue
                kind = "predicted_bbox"
            metadata = {key: item.get(key, "") for key in ("expert_id", "evidence_id", "capability", "scope")}
            metadata.update(label=_label(entry), coordinates=box,
                            coordinate_system="normalized_xyxy_original_image", representation=kind)
            annotations.append((metadata, mask))
    annotations.sort(key=lambda pair: -_relevance(pair[0], _tokens(question)))
    return annotations[:8]


def make_visual_evidence(image, items, question, max_views=1) -> tuple[list[Image.Image], list[dict]]:
    """Make at most one predicted-mask/bbox overlay of the current query image.

    ``items`` must belong to this query; arbitrary payload paths are never opened.
    Only explicitly supported coordinate systems are rendered. Geometry is not
    ground truth, and an empty mask produces no view and no anatomical claim.
    """
    if isinstance(max_views, bool) or not isinstance(max_views, int) or max_views < 0:
        raise ValueError("max_views must be a nonnegative integer")
    if not max_views:
        return [], []
    original = load_rgb(image)
    annotations = _annotations(original, items, question)
    if not annotations:
        return [], []
    view = original.copy()
    colors = ((255, 180, 0), (0, 210, 255), (235, 80, 235), (80, 240, 120))
    for index, (metadata, mask) in enumerate(annotations):
        color = colors[index % len(colors)]
        if mask is not None:
            colored = Image.new("RGB", view.size, color)
            view = Image.composite(Image.blend(view, colored, 0.26), view, mask)
            edge = Image.fromarray(
                np.asarray(mask.filter(ImageFilter.MaxFilter(3)))
                - np.asarray(mask.filter(ImageFilter.MinFilter(3)))
            )
            view.paste(color, (0, 0, view.width, view.height), edge)
        draw = ImageDraw.Draw(view)
        x0, y0, x1, y1 = [v * n for v, n in zip(metadata["coordinates"], view.size * 2)]
        box = (int(x0), int(y0), min(view.width - 1, math.ceil(x1) - 1),
               min(view.height - 1, math.ceil(y1) - 1))
        draw.rectangle(box, outline=color, width=2)
        caption = f"{index + 1}: {metadata['label'][:36]}".encode("ascii", "replace").decode()
        draw.text((box[0], box[1]), caption, fill=color, stroke_width=1, stroke_fill="black")
    draw = ImageDraw.Draw(view)
    draw.rectangle((0, 0, view.width - 1, view.height - 1), outline="yellow", width=2)
    draw.text((3, max(0, view.height - 13)), "PREDICTED TOOL EVIDENCE", fill="yellow",
              stroke_width=1, stroke_fill="black")
    metadata = {
        "view_kind": "predicted_evidence_overlay",
        "sources": [entry for entry, _ in annotations],
        "pixel_digest": _pixel_digest(view), "original_pixel_digest": _pixel_digest(original),
        "ground_truth": False, "outside_predicted_region": "unknown",
    }
    return [view], [metadata]
