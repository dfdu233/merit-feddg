"""Content-addressed manifests and exclusive resumable artifacts."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def read(path):
    def invalid(x):
        raise ValueError('Non-standard JSON constant: '+x)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value, *, replace=False):
    """Only state pointers may replace; reports, jobs and predictions are exclusive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.chain-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(canonical(value)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', name):
        raise ValueError('Unsafe cohort/sample name')
    return name


def pixel_hash(path):
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        # Chain registry uses raw RGB bytes. The older source pipeline also
        # has a size-prefixed variant, validated separately in source_inventory.
        return hashlib.sha256(rgb.tobytes()).hexdigest(), list(rgb.size)


def pin_files(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError('Explicit nonempty file-content pins required')
    for path, sha in pins.items():
        if not re.fullmatch('[0-9a-f]{64}', sha) or digest(path) != sha:
            raise ValueError('Changed pinned file: '+str(path))


def source_inventory(cohort):
    name = safe_name(cohort['name'])
    if cohort.get('split') != 'train':
        raise ValueError('Only explicitly declared official TRAIN input is allowed')
    root = Path(cohort['source_run']).resolve()
    protocol, complete = read(root/'protocol.json'), read(root/'complete.json')
    rows = protocol['rows']
    if not rows or complete['identity'] != protocol['identity'] or complete['n'] != len(rows):
        raise ValueError('A complete source queue is required')
    if 'train' not in str(protocol.get('selection', '')).lower():
        raise ValueError('Source protocol must explicitly identify TRAIN selection')
    if not protocol.get('test_image_exclusion_sha256'):
        raise ValueError('Missing historical test-image audit')
    forbidden = {'answer','answers','reference','references','label','labels','ground_truth'}
    records, seen = [], set()
    for row in rows:
        key = safe_name(row['id'])
        if key in seen or forbidden.intersection(row):
            raise ValueError('Duplicate ID or reference-answer field in generation manifest')
        seen.add(key)
        case_path = root/'cases'/(key+'.json')
        case = read(case_path)
        if not case.get('complete') or case.get('identity') != protocol['identity'] or case.get('id') != key:
            raise ValueError('Invalid cached source identity')
        image = Path(row['image'])
        if digest(image) != row['image_sha256']:
            raise ValueError('Changed source image')
        pixels, shape = pixel_hash(image)
        source_pixel_scheme = 'raw_rgb'
        if row.get('pixel_sha256', pixels) != pixels:
            # Existing native TRAIN caches use open_data.pixel_digest:
            # SHA256(str(rgb.size).encode() + rgb.tobytes()). Do not reinterpret
            # that digest as raw bytes or mutate the source/registry semantics.
            from ..open_data import pixel_digest
            if row['pixel_sha256'] != pixel_digest(image):
                raise ValueError('Changed decoded image pixels')
            source_pixel_scheme = 'size_prefixed_rgb'
        for arm in ('generalist','compact'):
            prediction = case['arms'][arm]
            if not prediction['token_ids'] or not isinstance(prediction['text'], str):
                raise ValueError('Missing actual cached baseline/compact output')
        records.append(dict(id=name+':'+key, source_id=key, pixel_sha256=pixels, image_size=shape,
                            case_sha256=digest(case_path), image_sha256=digest(image),
                            source_pixel_scheme=source_pixel_scheme))
    for group in ('source','formal_sources'):
        if protocol.get(group):
            pin_files(protocol[group])
    return dict(cohort=name, source_run=str(root), dataset=cohort['dataset'], split='train',
                protocol_sha256=digest(root/'protocol.json'), source_identity=protocol['identity'],
                records=records)


def freeze_plan(plan, root):
    from .policy import validate_policy
    from . import SCHEMA
    if plan.get('schema') != SCHEMA:
        raise ValueError('Wrong plan schema')
    validate_policy(plan['policy'])
    if not plan.get('development'):
        raise ValueError('Development cohort required')
    # Prior exposure registry is user-audited, not inferable from file names.
    exposed = set(read(plan['exposed_pixels']))
    tests = set(read(plan['test_pixels']))
    if not tests or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in exposed|tests):
        raise ValueError('Explicit test and exposure pixel registries required')
    pin_files(plan['runtime']['pins'])
    pin_files(plan['scorer']['pins'])
    frozen = json.loads(canonical(plan))
    frozen['inventory'] = {}
    frozen['generation'] = {}
    frozen['generation_pins'] = {}
    names, all_ids = set(), set()
    role_pixels = {}
    for role in ('development','confirmation','extension'):
        inventories = []
        role_pixels[role] = set()
        for cohort in plan.get(role, []):
            inv = source_inventory(cohort)
            generation_path = str(Path(cohort['generation_json']).resolve())
            frozen['generation'][inv['cohort']] = read(generation_path)
            frozen['generation_pins'][generation_path] = digest(generation_path)
            if inv['cohort'] in names:
                raise ValueError('Cohort names must be globally unique')
            names.add(inv['cohort'])
            ids = {r['id'] for r in inv['records']}
            pixels = {r['pixel_sha256'] for r in inv['records']}
            if pixels & tests or ids & all_ids:
                raise ValueError('TEST-image leakage or duplicate identity')
            if pixels & role_pixels[role]:
                raise ValueError('Repeated image across cohorts')
            role_pixels[role] |= pixels
            all_ids |= ids
            inventories.append(inv)
        frozen['inventory'][role] = inventories
    if role_pixels['confirmation'] & (role_pixels['development']|exposed):
        raise ValueError('Confirmation includes previously observed/development images')
    if role_pixels['extension'] & (role_pixels['development']|role_pixels['confirmation']|exposed):
        raise ValueError('Extension is not fresh relative to prior stages')
    frozen['registry_pins'] = {str(Path(plan[k]).resolve()): digest(plan[k])
                                for k in ('test_pixels','exposed_pixels')}
    root = Path(root).resolve()
    files = [*sorted((root/'merit_feddg').rglob('*.py')), root/'scripts/run_algorithm_chain.py',
             root/'scripts/run_huatuo_pathway.py']
    frozen['code_pins'] = {str(p):digest(p) for p in files if p.is_file()}
    frozen['identity'] = fingerprint(frozen)
    return frozen


def verify_plan(plan):
    if fingerprint({k:v for k,v in plan.items() if k != 'identity'}) != plan['identity']:
        raise ValueError('Frozen plan identity changed')
    for group in ('code_pins','registry_pins','generation_pins'):
        pin_files(plan[group])
    pin_files(plan['runtime']['pins'])
    pin_files(plan['scorer']['pins'])
    for role in ('development','confirmation','extension'):
        if len(plan.get(role, [])) != len(plan['inventory'][role]):
            raise ValueError('Frozen cohort inventory length changed')
        for cohort, frozen in zip(plan.get(role,[]),plan['inventory'][role]):
            if source_inventory(cohort) != frozen:
                raise ValueError('Frozen source queue changed')
