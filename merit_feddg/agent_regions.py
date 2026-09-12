"""Native predicted masks -> bounded ROIs; no invented whole-image prompt box."""
from __future__ import annotations

import base64
import binascii
import math
import zlib

import numpy as np

MAX_PIXELS = 16_777_216


def decode_mask(value: dict) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError('encoded mask dictionary required')
    size = value.get('size', [])
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(x) is not int or x < 1 for x in size)
            or math.prod(size) > MAX_PIXELS):
        raise ValueError('invalid mask size')
    count = math.prod(size)
    if value.get('encoding') == 'rle-row-major-zero-first':
        runs = value.get('counts', [])
        if (not isinstance(runs, list) or not runs or len(runs) > count + 1
                or any(type(x) is not int or x < 0 for x in runs) or sum(runs) != count):
            raise ValueError('invalid row-major RLE')
        return np.repeat(np.arange(len(runs)) % 2, runs).reshape(size).astype(np.float32)
    if value.get('encoding') == 'float32-zlib-base64':
        data = value.get('data')
        if not isinstance(data, str) or len(data) > count * 8 + 256:
            raise ValueError('missing or excessive compressed mask data')
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(base64.b64decode(data, validate=True), count * 4 + 1)
        except (ValueError, zlib.error, binascii.Error) as exc:
            raise ValueError('invalid compressed mask') from exc
        if len(raw) != count * 4 or not decoder.eof or decoder.unused_data:
            raise ValueError('mask byte count mismatch')
        mask = np.frombuffer(raw, dtype='<f4').reshape(size)
        if not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError('nonfinite or out-of-range mask')
        return mask
    raise ValueError('unsupported native mask encoding')


def _box(box):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)):
        raise ValueError('finite normalized xyxy required')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('empty or out-of-range region')
    return tuple(float(x) for x in box)


def region_from_entry(entry: dict, payload: dict, image_size: tuple[int, int]) -> dict:
    mask = decode_mask(entry.get('soft_mask', entry.get('mask', {})))
    coordinate = entry.get('mask_coordinate_system')
    if coordinate == 'original_image':
        if mask.shape != (image_size[1], image_size[0]):
            raise ValueError('original-resolution mask mismatch')
        crop = (0., 0., 1., 1.)
    elif coordinate == 'model_grid_of_center_crop':
        transform = payload.get('image_transform', {})
        if (transform.get('original_size_hw') != [image_size[1], image_size[0]]
                or transform.get('model_size_hw') != list(mask.shape)):
            raise ValueError('missing or mismatched crop transform')
        crop = _box(transform.get('crop_box_xyxy_normalized'))
    else:
        raise ValueError('explicit supported coordinate system required')
    # Fixed geometric extraction convention, NOT a disease/acceptance threshold.
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        raise ValueError('empty predicted mask; absence cannot be inferred')
    h, w = mask.shape
    a, b, c, d = crop
    box = (a + xs.min()/w*(c-a), b + ys.min()/h*(d-b),
           a + (xs.max()+1)/w*(c-a), b + (ys.max()+1)/h*(d-b))
    box = _box([float(v) for v in box])
    return {
        'box': list(box), 'coordinate_system': 'original_image_normalized_xyxy',
        'label': str(entry.get('anatomical_structure', entry.get('label', 'unknown'))),
        'mask_area_fraction_in_own_grid': float((mask > 0.5).mean()),
        'object_presence_confirmed': False,
        'allowed_use': ['crop', 'local_observation', 'analogy'],
        'not_supported': ['disease_presence', 'disease_absence', 'physical_measurement'],
    }


def extract_regions(evidence: list[dict], image_size: tuple[int, int], *, limit=2):
    if type(limit) is not int or limit < 1:
        raise ValueError('positive region limit required')
    regions, audit, seen = [], [], set()
    for source_index, item in enumerate(evidence):
        if item.get('capability') != 'segmentation':
            continue  # CAM is not silently promoted to lesion segmentation.
        provenance = item.get('provenance', {})
        if provenance.get('target_masks_used') or provenance.get('target_mask_used'):
            raise ValueError('target reference mask is forbidden')
        payload = item.get('payload', {})
        entries = payload.get('structures', [])
        if not entries and ('mask' in payload or 'soft_mask' in payload):
            # Native MedSAM adapter explicitly declares original-image resolution.
            if provenance.get('mask_resolution') == 'original_image':
                entries = [{**payload, 'mask_coordinate_system': 'original_image'}]
            else:
                entries = [payload]
        for index, entry in enumerate(entries):
            origin = {'expert_id': item.get('expert_id'), 'evidence_id': item.get('evidence_id'),
                      'source_index': source_index, 'entry_index': index}
            try:
                region = region_from_entry(entry, payload, image_size)
                key = (origin['expert_id'], tuple(region['box']))
                if key in seen:
                    raise ValueError('duplicate region from same expert')
                seen.add(key)
                if len(regions) >= limit:
                    audit.append({**origin, 'reason': 'region_budget'})
                    continue
                regions.append({**region, **origin})
            except (ValueError, TypeError, KeyError) as exc:
                audit.append({**origin, 'reason': str(exc)})
    return regions, audit


def crop_box_pixels(box, image_size, *, mode='expert', padding=0.05):
    x0, y0, x1, y1 = _box(box)
    if not math.isfinite(padding) or not 0 <= padding <= 0.25:
        raise ValueError('invalid fixed padding')
    if mode not in {'expert', 'opposite_control', 'full_image'}:
        raise ValueError('unknown crop mode')
    w, h = image_size
    if type(w) is not int or type(h) is not int or w < 2 or h < 2:
        raise ValueError('invalid image dimensions')
    if mode == 'full_image':
        return 0, 0, w, h
    dx, dy = (x1-x0)*padding, (y1-y0)*padding
    left, top = max(0, math.floor((x0-dx)*w)), max(0, math.floor((y0-dy)*h))
    right, bottom = min(w, math.ceil((x1+dx)*w)), min(h, math.ceil((y1+dy)*h))
    if mode == 'opposite_control':
        left, right, top, bottom = w-right, w-left, h-bottom, h-top
    if right <= left or bottom <= top:
        raise ValueError('empty crop')
    # Shift the window; do NOT flip patient anatomy.
    return left, top, right, bottom
