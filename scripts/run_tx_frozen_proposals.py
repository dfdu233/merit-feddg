"""Generate source-only immutable Generalist and BARD proposals, without labels.

Uses the existing native-cache and BARD implementations. Sharding schedules cases;
it does not alter the model, prompts, experts or decoding policy.
"""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from merit_feddg.bard_protocol import acquire_expert_groups, run_bard_method
from merit_feddg.capability_runtime import CapabilityRuntime, NativeSession, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import SharedExpertPool, experiment_arms, generation_prompt, load_manifest, load_cached
from merit_feddg.open_study import fingerprint
from merit_feddg.open_data import INFERENCE_FIELDS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with gzip.open(tmp, 'wt') as f:
        json.dump(value, f)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--cache')
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=['generalist', 'bard'], required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--cached-receiver', action='store_true')
    p.add_argument('--wait-for-cache', action='store_true', help='Allow an in-progress lossless cache transfer')
    args = p.parse_args()
    config = load_experiment_yaml(args.config)
    rows = load_manifest(args.manifest)
    metadata = {r['id']:r for r in map(json.loads, Path(args.manifest).read_text().splitlines())}
    assert 0 <= args.shard_index < args.shard_count
    assert all(r.get('split') in {'source','development','train'} for r in metadata.values())
    cache = Path(args.cache) if args.cache else None
    protocol = json.loads((cache/'protocol.json').read_text()) if cache else None
    if args.method == 'bard':
        assert protocol and protocol['shards_complete']
        assert protocol['config']['capability_value'] == config['capability_value']
        assert protocol['config']['experts'] == config['experts']
    routes = json.loads((cache/'routing.json').read_text()) if cache else {}
    probe = load_generalist(config['generalist'])
    if args.cached_receiver:
        from huatuo_cached_receiver import enable_cached_sessions
        enable_cached_sessions(probe)
    arm = experiment_arms(ValueGenerationConfig(**config['capability_value']['generation']), 'bard')[args.method]
    output = Path(args.output)
    for row in rows[args.shard_index::args.shard_count]:
        row.update(metadata[row['id']])
        path = output/row['id']/(args.method+'.json.gz')
        if path.exists():
            continue
        assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest() == row['image_sha256']
        row['modality'] = routes.get(row['id'], {}).get('modality', row['modality'])
        row['capability'] = 'classification'
        prompt = generation_prompt(row,config)
        session = NativeSession(probe,row['image'],prompt,row['question'],arm)
        pool = SharedExpertPool(None, cache/'expert-cache'/fingerprint(row['id']),protocol['identity']) if cache else None
        specs = {k:v for k,v in protocol['config']['experts'].items() if k not in protocol.get('excluded', {}) and k != 'source_cases'} if protocol else config['experts']
        public_row = {k:row[k] for k in INFERENCE_FIELDS}
        if args.method == 'bard' and args.wait_for_cache:
            from dataclasses import asdict
            from prepare_bard_expert_cache import make_request
            schedule = json.loads((cache/'schedule.json').read_text())[row['id']]
            for descriptor in schedule:
                request = make_request(public_row,descriptor,arm)
                key = fingerprint(['infer',descriptor['expert'],asdict(request)])
                cache_path = cache/'expert-cache'/fingerprint(row['id'])/(key+'.json')
                deadline = time.monotonic()+3600
                while True:
                    try:
                        if load_cached(cache_path,protocol['identity']) is not None:
                            break
                    except json.JSONDecodeError:
                        pass  # The transfer has not written the complete JSON yet.
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'incomplete native cache transfer: {cache_path}')
                    time.sleep(2)
        runtime = CapabilityRuntime(session,pool,public_row,specs,arm,None)
        if args.method == 'generalist':
            value = runtime.run('generalist')
            members = {}
        else:
            acquisition = acquire_expert_groups(runtime)
            value = run_bard_method(session,acquisition,config.get('bard',{}),'bard')
            members = acquisition['fault_group_members']
            value.pop('evidence', None)
            value['native_evidence_cache'] = {'root':str(cache),'identity':protocol['identity'],'case':fingerprint(row['id'])}
        value['proposer_expert_ids'] = sorted({e for entries in members.values() for e in entries})
        value['provenance'] = {'row':row,'config':config,'prompt':prompt,'answers_loaded':False,'cached_receiver':args.cached_receiver}
        save(path,value)
        print('COMPLETE',args.method,row['id'],value['text'],flush=True)


if __name__ == '__main__':
    main()
