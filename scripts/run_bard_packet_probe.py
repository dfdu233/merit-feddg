"""Real receiver replay of pinned source packets; no references or specialist reruns."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merit_feddg.bard import BARDConfig, _prepare, decode_bard
from merit_feddg.bard_protocol import build_isolated_sessions
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.open_study import atomic_json


def run_bundle(args, probe=None):
    bundle = json.loads(Path(args.bundle).read_text())
    row = bundle['row']
    if any(k in row for k in ('answer', 'answers', 'reference', 'label')):
        raise ValueError('No references in receiver input')
    if hashlib.sha256(Path(row['image']).read_bytes()).hexdigest() != bundle['image_file_sha256']:
        raise ValueError('Wrong image bytes')
    if probe is None:
        probe = load_probe(bundle['generalist'])
    cfg = ValueGenerationConfig(**bundle['generation'])
    bcfg = BARDConfig(**bundle['bard'])
    groups = {name: tuple(EvidenceItem(**v) for v in values)
              for name, values in bundle['groups'].items()}
    cost_path = Path(args.output) / f'cost-shard-{args.shard_index}.json'
    previous = json.loads(cost_path.read_text())['model_forwards'] if args.resume and cost_path.exists() else 0
    counter = [previous]
    def count(_module, _args):
        used = getattr(probe, '_bard_worker_forwards', 0)
        if used >= args.max_forwards:
            atomic_json(cost_path, {'model_forwards': counter[0]})
            raise RuntimeError('Frozen worker forward budget reached; completed outputs preserved')
        probe._bard_worker_forwards = used + 1
        counter[0] += 1
    handle = probe.model.register_forward_pre_hook(count)
    tasks = [(method, fault) for fault in [None, *groups]
             for method in ('generalist', 'isolated_mean', 'isolated_geomedian', 'bard')
             if fault is None or method != 'generalist']
    if args.clean_only:
        tasks = [t for t in tasks if t[1] is None]
    if args.include_joint:
        tasks.insert(1, ('joint_all', None))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    native_control = probe.generate_with_usage(row['image'], row['prompt'], max_new_tokens=cfg.max_new_tokens)
    atomic_json(out / f'native-control-shard-{args.shard_index}.json', native_control)
    if args.resume:
        for existing in out.glob('[0-9]*.json'):
            saved = json.loads(existing.read_text())
            if saved.get('method') == 'generalist' and saved['token_ids'] != native_control['token_ids']:
                raise ValueError('Resumed native Generalist differs from saved output')
    try:
        for index, (method, fault) in enumerate(tasks):
            if index % args.shard_count != args.shard_index:
                continue
            dest = out / f'{index:02d}.json'
            if dest.exists():
                old = json.loads(dest.read_text())
                if not args.resume or old.get('bundle_sha256') != hashlib.sha256(Path(args.bundle).read_bytes()).hexdigest() or old.get('method') != method or old.get('corrupted_expert') != fault:
                    raise ValueError('Existing output incompatible with this frozen input')
                continue
            native = NativeSession(probe, row['image'], row['prompt'], row['question'], cfg)
            base, experts, transport = build_isolated_sessions(native, groups)
            class CachedBase:
                def __init__(self, session):
                    self.session, self.prefix, self.scores = session, None, None
                def next_scores(self, prefix):
                    if tuple(prefix) != self.prefix:
                        self.scores = self.session.next_scores(prefix)
                        self.prefix = tuple(prefix)
                    return self.scores
                def decode(self, tokens):
                    return self.session.decode(tokens)
                @property
                def eos_ids(self):
                    return self.session.eos_ids
            base = CachedBase(base)
            if fault is not None:
                if fault not in experts:
                    raise ValueError('Faulted expert was not actually delivered')
                clean_expert = experts[fault]
                class CorruptedExpert:
                    def __init__(self, base, expert):
                        self.base, self.expert = base, expert
                    def next_scores(self, prefix):
                        # decode_bard has already evaluated this exact base prefix.
                        assert self.base.prefix == tuple(prefix)
                        b, mask, residual = _prepare(self.base.scores, [self.expert.next_scores(prefix)])
                        scores = np.full_like(b, -np.inf)
                        scores[mask] = b[mask] - bcfg.fault_probe_scale * residual[0]
                        return scores
                experts[fault] = CorruptedExpert(base, clean_expert)
            if method == 'joint_all':
                from merit_feddg.capability_runtime import NativeState
                joint_image, joint_prompt = native.context(NativeState(items=tuple(v for group in groups.values() for v in group)))
                base = probe.new_answer_session(joint_image, joint_prompt)
                transport = {'joint': native.last_transport}
                experts = {}
            if method == 'generalist':
                experts = {}
            before = counter[0]
            probe.torch.cuda.synchronize()
            probe.torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = decode_bard(base, experts, max_tokens=cfg.max_new_tokens, config=bcfg,
                                 aggregation='mean' if method == 'isolated_mean' else 'geometric_median',
                                 bounded_commit=method in ('bard', 'generalist', 'joint_all'),
                                 fault_probe=method == 'bard' and fault is None)
            probe.torch.cuda.synchronize()
            if method == 'generalist' and result['token_ids'] != native_control['token_ids']:
                raise ValueError('Exact-prefix Generalist differs from native generation')
            result.update(method=method, corrupted_expert=fault, condition='clean' if fault is None else 'single_residual_sign_reversal',
                          case_id=row['id'], model_forwards=counter[0]-before, seconds=time.perf_counter()-start,
                          peak_allocated_bytes=probe.torch.cuda.max_memory_allocated(), isolated_transport=transport,
                          config=asdict(bcfg), torch_version=probe.torch.__version__,
                          bundle_sha256=hashlib.sha256(Path(args.bundle).read_bytes()).hexdigest(),
                          scope='Free generation under one fixed corrupted receiver node; not raw-packet corruption or a clinical adversary')
            atomic_json(dest, result)
            atomic_json(out / f'cost-shard-{args.shard_index}.json', {'model_forwards': counter[0]})
            print('DONE', index, method, fault, 'forwards', result['model_forwards'], flush=True)
    finally:
        atomic_json(cost_path, {'model_forwards': counter[0]})
        handle.remove()


def load_probe(spec):
    if spec.get('backend') == 'huatuo':
        from huatuo_bard_receiver import HuatuoReceiver
        return HuatuoReceiver(spec)
    return load_generalist(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    parser.add_argument('--queue', help='Frozen JSONL of bundle path, SHA256 and unique case key')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-forwards', type=int, default=100000)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--include-joint', action='store_true')
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, default=2)
    args = parser.parse_args()
    if bool(args.bundle) == bool(args.queue):
        parser.error('Specify exactly one of --bundle or --queue')
    if args.bundle:
        run_bundle(args)
        return
    if not args.clean_only or not args.include_joint:
        parser.error('Full queue requires explicit --clean-only --include-joint')
    import copy
    queue = [json.loads(line) for line in Path(args.queue).read_text().splitlines() if line.strip()]
    if len({r['key'] for r in queue}) != len(queue):
        raise ValueError('Duplicate queue key')
    code_paths = [Path(__file__), Path(__file__).with_name('huatuo_bard_receiver.py'), Path(__file__).resolve().parents[1] / 'merit_feddg/bard.py', Path(__file__).resolve().parents[1] / 'merit_feddg/bard_protocol.py']
    frozen = {'queue_sha256': hashlib.sha256(Path(args.queue).read_bytes()).hexdigest(), 'code': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}, 'shard_count': args.shard_count, 'shard_index': args.shard_index, 'max_forwards': args.max_forwards}
    marker = Path(args.output) / f'worker-{args.shard_index}.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists() and json.loads(marker.read_text()) != frozen:
        raise ValueError('Worker code or queue changed; use a new run')
    atomic_json(marker, frozen)
    spent = sum(json.loads(p.read_text())['model_forwards'] for i, r in enumerate(queue) if i % args.shard_count == args.shard_index for p in [Path(args.output) / r['key'] / 'cost-shard-0.json'] if p.exists())
    probe, model_spec = None, None
    for index, row in enumerate(queue):
        if index % args.shard_count != args.shard_index:
            continue
        path = Path(row['bundle'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Queue bundle changed after freeze')
        bundle = json.loads(path.read_text())
        if model_spec is None:
            model_spec = bundle['generalist']
        if bundle['generalist'] != model_spec:
            raise ValueError('Cannot mix model specifications in one worker')
        child = copy.copy(args)
        child.bundle, child.output = str(path), str(Path(args.output) / row['key'])
        child.shard_index, child.shard_count = 0, 1
        if args.resume:
            outputs = [Path(child.output) / f'{i:02d}.json' for i in range(5)]
            if all(p.is_file() for p in outputs):
                previous = [json.loads(p.read_text()) for p in outputs]
                if any(d.get('bundle_sha256') != row['sha256'] for d in previous):
                    raise ValueError('Completed case belongs to different input')
                print('REUSED_COMPLETE', index, row['key'], flush=True)
                continue
        if probe is None:
            probe = load_probe(model_spec)
            probe._bard_worker_forwards = spent
        run_bundle(child, probe)
        print('CASE_COMPLETE', index, row['key'], flush=True)


if __name__ == '__main__':
    main()
