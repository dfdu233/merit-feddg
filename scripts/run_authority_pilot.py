"""Bounded real TRAIN candidate-pool pilot; no references enter inference."""
import argparse
import json
import time
from pathlib import Path

import torch
from torchxrayvision.models import model_urls

from run_uncertainty_train import (
    resources, gpu_check, GPUS, context, evidence, visible, EvidenceItem,
    ValueGenerationConfig, load_generalist, atomic_json, fingerprint, sha,
    source_fingerprint, MANIFEST, BASE,
)
from merit_feddg.authority_projection import (
    binary_finding_group, project_free_text_candidates, transport_diagnostic,
)
from merit_feddg.authority_projection import xrv_operating_coordinate
from merit_feddg.open_data import pixel_digest

OLD = Path('/home/dbw/merit-feddg-uncertainty/runs/uncertainty-train-v1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-cases', type=int, default=2)
    args = parser.parse_args()
    protocol, rows, incumbent = resources()
    frozen = json.loads((OLD/'frozen.json').read_text())
    sources = json.loads((OLD/'sources.json').read_text())
    assert sources['identity'] == fingerprint(frozen)
    assert source_fingerprint() == frozen['source']
    assert sha(MANIFEST) == frozen['manifest_sha256']
    assert sha(BASE/'compact_rows.json') == frozen['base_sha256']
    meta = model_urls['densenet121-res224-all']
    from evaluate_soft_guidance_full import scorer_hashes
    assert scorer_hashes() == frozen['scorer']
    cfg = {'schema': 'authority-train-pool-v1', 'selected': frozen['selected'],
           'manifest': str(MANIFEST), 'manifest_sha256': sha(MANIFEST),
           'scorer': scorer_hashes(), 'source_identity': fingerprint(frozen),
           'operating_points': dict(zip(meta['labels'], meta['op_threshs'])),
           'pool': 'deduplicated existing ten TRAIN arm texts in frozen order; no fabricated answers',
           'scores': 'sequence mean log likelihood, canonical tokens including EOS',
           'code': {p: sha(p) for p in [__file__, 'merit_feddg/authority_projection.py']},
           'case_hashes': {s['id']: sha(OLD/'cases'/(s['id']+'.json')) for s in frozen['selected']}}
    identity = fingerprint(cfg)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root/'frozen.json').exists():
        assert json.loads((root/'frozen.json').read_text()) == cfg
    else:
        atomic_json(root/'frozen.json', cfg)
    print('PREFLIGHT', identity, root, flush=True)
    if args.check_only:
        return
    gpu_check(GPUS[0])
    started = time.perf_counter()
    probe = load_generalist(protocol['config']['generalist'], 'artifacts')
    probe.model.eval().requires_grad_(False)
    atomic_json(root/f'load-{time.time_ns()}.json', {'seconds': time.perf_counter()-started})
    lookup = {r['id']: r for r in rows}
    done = 0
    for selection in cfg['selected']:
        key = selection['id']
        path = root/'cases'/(key+'.json')
        if path.exists():
            assert json.loads(path.read_text())['identity'] == identity
            continue
        started = time.perf_counter()
        row, old = lookup[key], incumbent[key]
        assert pixel_digest(row['image']) == row['image_sha256']
        cached = json.loads((OLD/'cases'/(key+'.json')).read_text())
        assert cached['identity'] == fingerprint(frozen)
        candidates = list(dict.fromkeys(cached['arms'][a]['text'] for a in frozen['arms']))
        native = tuple(EvidenceItem(**e) for e in old['evidence'])
        gen = ValueGenerationConfig(**old['generation_config'])
        initial, transport = context(probe, row, protocol, gen, native)
        delivered = visible(transport)
        kept = tuple(e for e in native if (e.expert_id, e.evidence_id) in delivered)
        base, base_transport = context(probe, row, protocol, gen, kept)
        source = sources['cases'][key]
        conditioned, text_transport = context(probe, row, protocol, gen, kept+(evidence(key, source, True),))
        assert visible(base_transport) == delivered
        assert visible(text_transport) == delivered | {('xrv_native_uncertainty', 'xrv-native:'+key)}
        with torch.inference_mode():
            direct = initial.propose((), count=1, length=64)[0]
            if list(direct.tokens) != old['token_ids']:
                raise RuntimeError('historical incumbent token parity failed')
            parity_seconds = time.perf_counter()-started
            tokens = [probe.tokenizer.encode(t, add_special_tokens=False)+[probe.tokenizer.eos_token_id] for t in candidates]
            score_start = time.perf_counter()
            base_scores = [base.sequence_mean_logp((), t) for t in tokens]
            text_scores = [conditioned.sequence_mean_logp((), t) for t in tokens]
            score_seconds = time.perf_counter()-score_start
        aliases = {'Effusion': ['effusion'], 'Cardiomegaly': ['cardiomegaly', 'enlarged heart'],
                   'Pneumothorax': ['pneumothorax']}[source['label']]
        groups = [binary_finding_group(t, aliases) for t in candidates]
        target = xrv_operating_coordinate(source['probability'], cfg['operating_points'][source['label']])
        projection = project_free_text_candidates(candidates, base_scores, groups, target)
        result = {'id': key, 'identity': identity, 'candidates': candidates, 'groups': groups,
                  'base_scores': base_scores, 'text_scores': text_scores,
                  'projection': {k: v.tolist() if hasattr(v, 'tolist') else v for k,v in projection.items()},
                  'transport_diagnostic': transport_diagnostic(base_scores, text_scores, groups, target),
                  'arms': {'incumbent': old['text'], 'pool_base': projection['base_text'],
                           'pool_text': candidates[max(range(len(candidates)), key=lambda i:text_scores[i])],
                           'authority': projection['selected_text']},
                  'historical_token_parity': True, 'transport': text_transport,
                  'cost': {'wall_seconds': time.perf_counter()-started, 'score_seconds': score_seconds,
                           'parity_seconds': parity_seconds, 'score_calls': 2*len(candidates),
                           'parity_calls': 1, 'inherited_candidate_seconds': sum(a['seconds'] for a in cached['arms'].values()),
                           'inherited_source_seconds': source['seconds'], 'inherited_incumbent_seconds': old['seconds']}}
        atomic_json(path, result)
        print('DONE', key, projection['status'], 'candidates', len(candidates), 'seconds', result['cost']['wall_seconds'], flush=True)
        done += 1
        if done >= args.max_cases:
            break


if __name__ == '__main__':
    main()
