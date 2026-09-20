"""Audited device-only successor for a pre-model resource failure.

Preserves the original frozen directory and exhausted-attempt accounting. The
successor stays blocked until the standard retry command grants its next attempt.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merit_feddg.algorithm_chain.storage import digest, fingerprint, lock, read, verify_plan, write


def migrate(source, output, gpu_uuid, allowed_display_contexts=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Successor directory must not already exist')
    with lock(source / '.controller.lock'):
        plan, state = read(source / 'plan.json'), read(source / 'state.json')
        verify_plan(plan)
        if state['plan_sha'] != plan['identity']:
            raise ValueError('State/plan mismatch')
        expansion = plan.get('experiment_mode') == 'fixed_candidate_exploratory_expansion' and state.get('blocked_node') == 'C'
        if state['node'] != 'BLOCKED_TECHNICAL' or (state.get('blocked_node') != 'A' and not expansion):
            raise ValueError('Only a blocked initial A or exploratory C resource attempt can migrate')
        if (state['forwards_used'] != 0 and not expansion) or state.get('runtime_sha') or state['holdout_consumed']:
            raise ValueError('Device-only migration requires no model execution or holdout use')
        if not expansion and (state['attempts'] != {'A': 1} or len(state['history']) != 1):
            raise ValueError('Expected exactly one prior A attempt')
        if expansion:
            if state['attempts'].get('C') != 2 or plan['policy']['max_node_attempts'] != 3:
                raise ValueError('Expansion must retain the second C attempt and authorized third attempt')
            spent=0
            for item in state['history']:
                directory=source/item['directory']
                if digest(directory/'decision.json') != item['report_sha']:
                    raise ValueError('Historical decision changed')
                spent += read(directory/'budget.json')['used'] if (directory/'budget.json').exists() else 0
            for item in state.get('preflight_costs',[]):
                if read(source/item['directory']/'budget.json')['used'] != item['forwards']:
                    raise ValueError('Historical preflight cost changed')
                spent += item['forwards']
            if spent != state['forwards_used']:
                raise ValueError('Inherited model budget mismatch')
        entry = state['history'][-1]
        previous = source / entry['directory']
        if entry['verdict'] != 'technical_failure' or digest(previous / 'decision.json') != entry['report_sha']:
            raise ValueError('Prior failure evidence does not match')
        if 'Another compute process owns the authorized GPU' not in (previous / 'inference.log').read_text():
            raise ValueError('Prior failure must be the strict device ownership check')
        if any((previous / name).exists() for name in ('runtime.json', 'budget.json', 'predictions.json')):
            raise ValueError('Unexpected model execution artifacts')
        if gpu_uuid == plan['runtime']['gpu_uuid'] or not gpu_uuid.startswith('GPU-'):
            raise ValueError('A different physical GPU UUID is required')
        successor = deepcopy(plan)
        successor.pop('identity')
        successor['runtime']['gpu_uuid'] = gpu_uuid
        if allowed_display_contexts:
            successor['runtime']['allowed_display_contexts'] = list(allowed_display_contexts)
        else:
            successor['runtime'].pop('allowed_display_contexts',None)
        successor['device_constraint'] = 'Latest user instruction authorizes this physical GPU: '+gpu_uuid
        successor['device_migration'] = dict(
            predecessor_plan_sha=plan['identity'], predecessor_state_sha256=digest(source / 'state.json'),
            predecessor_gpu_uuid=plan['runtime']['gpu_uuid'], successor_gpu_uuid=gpu_uuid,
            reason='User-authorized device migration after zero-forward ownership-check failure; previous device restriction superseded',
            attempts_inherited=state['attempts'], forwards_inherited=state['forwards_used'],
            migration_script_sha256=digest(Path(__file__)),
        )
        successor['identity'] = fingerprint(successor)
        verify_plan(successor)
        inherited = deepcopy(state)
        inherited['plan_sha'] = successor['identity']
        with lock(output / '.controller.lock'):
            for item in state['history']:
                shutil.copytree(source/item['directory'],output/item['directory'])
            for item in state.get('preflight_costs',[]):
                shutil.copytree(source/item['directory'],output/item['directory'])
            write(output / 'predecessor-state.json', state)
            write(output / 'plan.json', successor)
            write(output / 'state.json', inherited)
        print(successor['identity'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--display-contexts', help='JSON list of explicitly frozen desktop-context exceptions')
    args = parser.parse_args()
    migrate(args.source, args.output, args.gpu_uuid, read(args.display_contexts) if args.display_contexts else ())
