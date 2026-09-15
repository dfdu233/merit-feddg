"""Explicit, audited scheduling adapter around the unchanged frozen runner.

Single process only. A successful real four-cell score audit is required.
Old rows remain untouched; every newly generated row labels its cost backend.
"""
import json
import sys
import time
from pathlib import Path

import run_class_text_guidance as runner
from run_soft_guidance_full import sha
from merit_feddg.open_study import atomic_json
from merit_feddg.persistent_scores import PersistentScores


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__ + '\nAdditional required option: --cache-audit PATH')
        runner.main()
        return
    if '--cache-audit' not in sys.argv:
        raise RuntimeError('--cache-audit is required')
    index = sys.argv.index('--cache-audit')
    audit_path = Path(sys.argv[index+1])
    del sys.argv[index:index+2]
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'exact_parity_passed' or len(audit.get('cases', [])) != 4:
        raise RuntimeError('four real dataset/channel audit cells required')
    if not audit['checks'] or not all(c['exact'] and c['max_abs_delta'] == 0 for c in audit['checks']):
        raise RuntimeError('exact score parity required')
    admission_path = audit_path.with_suffix('.admission.json')
    admission = json.loads(admission_path.read_text())
    scorer_path = Path(__file__).parents[1]/'merit_feddg/persistent_scores.py'
    if admission != {'audit_sha256': sha(audit_path), 'scorer_sha256': sha(scorer_path)}:
        raise RuntimeError('audit/scorer admission identity mismatch')
    provenance = {'name': 'persistent-production-kv-v1', **admission,
        'wrapper_sha256': sha(__file__), 'scope': 'semantic classification/generation only',
        'parity_scope': 'four fixed real dataset/channel cells; complete score vectors',
        'vision_reuse': 'retained per-branch multimodal prefill KV; no cross-case cache'}
    original_load, original_case = runner.load_generalist, runner.guided_case
    sessions = []

    def load(*args, **kwargs):
        probe = original_load(*args, **kwargs)
        factory = probe.new_answer_session
        def new_session(*a, **k):
            session = PersistentScores(factory(*a, **k))
            sessions.append(session)
            return session
        probe.new_answer_session = new_session
        return probe

    def case(*args, **kwargs):
        sessions.clear()
        try:
            result = original_case(*args, **kwargs)
            result['persistent_scoring_cost'] = {
                'multimodal_prefills': sum(s.prefills for s in sessions),
                'incremental_forwards': sum(s.incremental_calls for s in sessions),
                'scope': 'soft/CAD scoring only; ordinary controls and inherited expert costs remain separate'}
        finally:
            sessions.clear()
        result['execution_backend'] = provenance
        if result['status'] == 'real_candidate':
            result['soft']['cache_policy'] = 'persistent production-prefill; independent branch KV'
            result['cad']['implementation'] = 'same-prefix persistent production KV'
        return result

    runner.load_generalist, runner.guided_case = load, case
    atomic_json(audit_path.parent/f'cache-launch-{time.time_ns()}.json', provenance)
    print('CACHE BACKEND', json.dumps(provenance), flush=True)
    runner.main()


if __name__ == '__main__':
    main()
