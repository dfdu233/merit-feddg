"""Fixed TRAIN plumbing probe: native Huatuo multiview candidate likelihoods.

No references at inference, no changed candidates, no confidence calibration.
Full-answer scoring is a diagnostic before any claim-level selector is built.
"""
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit_feddg.agent_regions import decode_mask

BASE = Path('/home/dbw/merit-feddg-huatuo-critic/runs/native-slake128-confirm-v2')
FULL_QUEUE = os.environ.get('OBSERVATION_FULL_QUEUE') == '1'
OUT = ROOT / ('runs/observation-blind-slake128-v1' if FULL_QUEUE else 'runs/observation-blind-slake8-v1')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive write: never replace a prior experiment output.
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def observation(question, evidence, size):
    """Exact native-name mention only; NOT general medical question parsing.

    The same pure view is used for normal and inherited packets. Never treats
    prompted structures as present or anatomy masks as lesion masks.
    """
    words = set(re.findall(r'[a-z]+', question.lower()))
    width, height = size
    entries = []
    union = np.zeros((height, width), dtype=bool)
    for item in evidence:
        if item.get('expert_id') != 'biomedparse_objects':
            continue  # Pinned native-coordinate adapter, not generic segmentation coverage.
        for row in item['payload'].get('structures', []):
            name = row.get('label', '')
            terms = set(re.findall(r'[a-z]+', name.lower()))
            core = terms - {'left', 'right'}
            if not core or not core.issubset(words):
                continue
            if words & {'left', 'right'} and not (terms & words & {'left', 'right'}):
                continue
            if row.get('mask_coordinate_system') != 'original_image':
                raise ValueError('Unsupported mask coordinate mapping')
            mask = decode_mask(row['soft_mask'])
            if mask.shape != (height, width):
                raise ValueError('Image/mask coordinate mismatch')
            binary = mask > .5  # Frozen original segmentation postprocessing, NOT gate threshold.
            if not binary.any():
                entries.append({'label': name, 'status': 'empty_mask'})
                continue
            ys, xs = np.nonzero(binary)
            union |= binary
            entries.append({'label': name, 'status': 'predicted_region_not_confirmed',
                'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                'area_pixels': int(binary.sum()), 'existence_confirmed': False,
                'evidence_id': item['evidence_id'], 'role': 'operation_only'})
    if not union.any():
        return {'available': False, 'entries': entries, 'reason': 'no_matching_nonempty_native_mask'}
    ys, xs = np.nonzero(union)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    dx, dy = math.ceil((x1-x0)*.1), math.ceil((y1-y0)*.1)
    box = [max(0,x0-dx), max(0,y0-dy), min(width,x1+dx), min(height,y1+dy)]
    if box == [0,0,width,height]:
        return {'available': False, 'entries': entries, 'reason': 'crop_equals_full_image'}
    return {'available': True, 'bbox_xyxy': box, 'entries': entries,
            'padding_fraction': .1, 'native_mask_threshold': .5}


def prepare():
    source = read(BASE/'protocol.json')
    assert read(BASE/'complete.json')['identity'] == source['identity']
    selected, audit = [], []
    for row in source['rows']:
        if row.get('q_lang') != 'en' and not FULL_QUEUE:
            continue
        if any(k in row for k in ('answer','answers','label','reference','references')):
            raise ValueError('Labels in generation manifest')
        path = BASE/'cases'/(row['id']+'.json')
        case = read(path)
        assert case['complete'] and case['identity'] == source['identity']
        with Image.open(row['image']) as im:
            view = observation(row['question'], case['compact_raw']['evidence'], im.size)
        if row.get('q_lang') != 'en':
            view = {'available':False, 'entries':[], 'reason':'unsupported_language'}
        audit.append({'id':row['id'], 'available':view['available'], 'reason':view.get('reason')})
        if view['available'] or FULL_QUEUE:
            selected.append(dict(row, observation=view, case_sha256=digest(path)))
        if len(selected) == 8 and not FULL_QUEUE:
            break
    if len(selected) != (len(source['rows']) if FULL_QUEUE else 8):
        raise ValueError('Fewer than eight eligible TRAIN cases; do not change sampling silently')
    config = {'base':str(BASE), 'base_identity':source['identity'], 'rows':selected,
        'selection':'first8 existing TRAIN schedule, English, exact native anatomy name mentioned and nonempty non-full crop; no outcome selection',
        'selection_audit':audit, 'source_sha256':digest(__file__),
        'stage':'full-answer likelihood engineering diagnostic, not claim-level gate or formal benchmark',
        'arms':['image_only','segmentation_text','native_crop','full_view_control'],
        'scoring':'mean normalized log probability of original candidate tokens including EOS; sum diagnostic',
        'new_models':False, 'training':False, 'gate_threshold':None,
        'scope_limit':'BiomedParse original-coordinate operation-only crops; no disease interpretation',
        'candidate_names':['generalist','compact'], 'inference_reads_references':False}
    if FULL_QUEUE:
        config['selection'] = 'Entire existing 128 TRAIN schedule unchanged; no outcome selection'
        config['unavailable_policy'] = 'Image-only likelihood remains evaluated; observation arms retain baseline when native observation unavailable. Report fallback separately, never as evidence efficacy.'
    config['identity'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write(OUT/'protocol.json', config)
    print('Prepared', config['identity'], [r['id'] for r in selected], flush=True)


def serialize(adapter, images, prompt):
    import torch
    bot = adapter.bot
    text = bot.insert_image_placeholder(bot.input_moderation(prompt), len(images))
    ids = bot.preprocess(bot.get_conv_without_history(text), return_tensors='pt').unsqueeze(0).to(bot.device)
    pixels = torch.stack(bot.get_image_tensors(images)).to(device=bot.device,dtype=torch.bfloat16)
    from llava.constants import IMAGE_TOKEN_INDEX
    assert int((ids == IMAGE_TOKEN_INDEX).sum()) == len(images) == pixels.shape[0]
    return ids, pixels


def score(adapter, ids, pixels, candidate):
    import torch
    model = adapter.model
    tok = torch.tensor(candidate,device=ids.device,dtype=ids.dtype).unsqueeze(0)
    full = torch.cat([ids,tok],dim=1)
    labels = torch.cat([torch.full_like(ids,-100),tok],dim=1)
    expanded = full.shape[1] + pixels.shape[0]*(model.get_vision_tower().num_patches-1)
    if expanded > min(model.config.tokenizer_model_max_length,model.config.max_position_embeddings):
        raise ValueError('No silent multimodal/answer truncation')
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        _, pos, attn, _, embeds, aligned = model.prepare_inputs_labels_for_multimodal_new(
            full,None,None,None,labels,pixels)
        assert embeds.shape[1] == expanded
        assert aligned[aligned != -100].tolist() == candidate
        output = model(inputs_embeds=embeds, position_ids=pos, attention_mask=attn,
                       labels=aligned, use_cache=False, return_dict=True)
        valid = aligned[:,1:] != -100
        logits = output.logits[:,:-1][valid].float()
        targets = aligned[:,1:][valid]
        values = logits.log_softmax(-1).gather(1,targets[:,None]).squeeze(1)
        if not torch.isfinite(values).all() or abs(float(values.mean()+output.loss)) > 1e-4:
            raise ValueError('Teacher-forcing alignment/CE parity failure')
        result = {'mean':float(values.mean()), 'sum':float(values.sum()),
                  'tokens':len(candidate), 'token_logprobs':values.cpu().tolist(),
                  'expanded_tokens':int(expanded), 'ce_parity':True}
    torch.cuda.synchronize()
    result['seconds'] = time.monotonic()-start
    return result


def run(canary=False):
    import subprocess
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],timeout=10,text=True).strip()
    if gpu != 'GPU-3846413a-4238-d307-b1f3-10c2dfbe002c':
        raise ValueError('Expected authorized host GPU1 only')
    config = read(OUT/'protocol.json')
    if config['source_sha256'] != digest(__file__):
        raise ValueError('Frozen code changed')
    sys.path.insert(0,'/home/dbw/ANCHOR')
    import torch
    from anchor.corrected_sgta.models_oe import HuatuoOEAdapter
    start = time.monotonic()
    adapter = HuatuoOEAdapter()
    adapter.model.eval().requires_grad_(False)
    print('MODEL_READY',time.monotonic()-start,flush=True)
    write(OUT/('load-canary.json' if canary else 'load-run.json'),
          {'seconds':time.monotonic()-start,'gpu':gpu,'torch':torch.__version__})
    for row in config['rows'][:1] if canary else config['rows']:
        target = OUT/'cases'/(row['id']+'.json')
        if target.exists():
            assert read(target)['identity'] == config['identity']
            continue
        if digest(row['image']) != row['image_sha256'] or digest(BASE/'cases'/(row['id']+'.json')) != row['case_sha256']:
            raise ValueError('Frozen input changed')
        case = read(BASE/'cases'/(row['id']+'.json'))
        image = Image.open(row['image']).convert('RGB')
        view = observation(row['question'],case['compact_raw']['evidence'],image.size)
        assert view == observation(row['question'],copy.deepcopy(case['compact_raw']['evidence']),image.size)
        if row.get('q_lang') != 'en':
            view = {'available':False, 'entries':[], 'reason':'unsupported_language'}
        assert view == row['observation']
        crop = image.crop(tuple(view['bbox_xyxy'])) if view['available'] else None
        prompt = row['benchmark_prompt']
        multiview = ('Image 1 is the original image. Image 2 is a view from the same image. '
                     'The view is an observation aid, not a confirmed finding.\n')
        conditions = {'image_only':([image],prompt)}
        if view['available']:
            conditions.update({
            'segmentation_text':([image],prompt+'\nUnconfirmed native segmentation output; query names do not establish presence:\n'+json.dumps(view['entries'])),
            'native_crop':([image,crop],multiview+'Image 2 pixel bbox in image 1: '+str(view['bbox_xyxy'])+'.\n'+prompt),
            'full_view_control':([image,image],multiview+'Image 2 pixel bbox in image 1: '+str([0,0,*image.size])+'.\n'+prompt)})
        result = {'id':row['id'],'identity':config['identity'],'complete':False,'conditions':{},
            'normal_inherited_view_parity':True,'observation':view,'calls':[],
            'original_pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
            'crop_pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest() if crop else None}
        for name in set(config['arms']) - set(conditions):
            result['conditions'][name] = {'selected':'generalist','text':case['arms']['generalist']['text'],
                'unavailable':view['reason'], 'scores':{}, 'new_answer_generation':False}
        for name,(images,text) in conditions.items():
            ids,pixels = serialize(adapter,images,text)
            if name == 'image_only':
                old_ids,old_pixels = adapter._inputs(image,prompt)
                assert torch.equal(ids,old_ids) and torch.equal(pixels,old_pixels)
                result['native_single_image_parity'] = True
            if name == 'native_crop' and torch.equal(pixels[0],pixels[1]):
                raise ValueError('Crop indistinguishable after actual preprocessing')
            values = {}
            for candidate in config['candidate_names']:
                tokens = case['arms'][candidate]['token_ids']
                if adapter.tokenizer.decode(tokens,skip_special_tokens=True).strip() != case['arms'][candidate]['text'].strip():
                    raise ValueError('Candidate tokens/text mismatch')
                values[candidate] = score(adapter,ids,pixels,tokens)
                result['calls'].append({'stage':name,'candidate':candidate,'seconds':values[candidate]['seconds']})
            winner = 'compact' if values['compact']['mean'] > values['generalist']['mean'] else 'generalist'
            result['conditions'][name] = {'scores':values,'selected':winner,
                'text':case['arms'][winner]['text'],'delivered_tensor_shape':list(pixels.shape),
                'prompt':text,'image_count':len(images), 'new_answer_generation':False}
            print(row['id'],name,winner,flush=True)
        result['complete'] = True
        result['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        write(target,result)
    if not canary:
        assert {p.stem for p in (OUT/'cases').glob('*.json')} == {r['id'] for r in config['rows']}
        write(OUT/'complete.json',{'identity':config['identity'],'n':len(config['rows'])})


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'prepare':
        prepare()
    elif stage in ('canary','run'):
        run(canary=stage == 'canary')
    else:
        raise ValueError('Expected prepare / canary / run')
