"""Bounded, typed evidence workflow; no fitted policy or medical risk certificate.

This module does not import or change the legacy runtime. Execution dependencies
are not independent votes. The ledger keeps raw observations immutable and marks
invalid descendants, rather than dropping unrelated evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Mapping, Sequence


def digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class Action:
    id: str
    operation: str
    inputs: tuple[str, ...]
    output_kind: str
    purpose: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.id, self.operation, self.output_kind, self.purpose)):
            raise ValueError('action fields must be nonempty strings')
        if (not isinstance(self.inputs, tuple) or not self.inputs
                or any(not isinstance(v, str) or not v for v in self.inputs)):
            raise ValueError('action inputs must be a nonempty tuple of references')


# Python enforces data types; this is not a model-generated instruction.
CONTRACTS = {
    'crop': (('region',), 'crop'),
    'inspect': (('crop',), 'observation'),
    'retrieve': (('crop',), 'analogy'),
}


class EvidenceLedger:
    """One case per ledger. Source roots indicate lineage, NOT independence."""

    def __init__(self, case_id: str):
        if not isinstance(case_id, str) or not case_id:
            raise ValueError('nonempty case_id required')
        self.case_id = case_id
        self._nodes: dict[str, dict] = {}
        self._events: list[dict] = []

    def add(self, node_id: str, kind: str, payload: dict, *, parents=(), source=None):
        if (not isinstance(node_id, str) or not node_id or node_id in self._nodes
                or not isinstance(kind, str) or not kind or not isinstance(payload, dict)):
            raise ValueError('unique node ID, kind and dictionary payload required')
        parents = tuple(parents)
        if (len(set(parents)) != len(parents)
                or any(p not in self._nodes or not self.active(p) for p in parents)):
            raise ValueError('duplicate, unknown or inactive dependency')
        data = json.loads(json.dumps(payload, allow_nan=False))
        roots = set()
        for parent in parents:
            roots.update(self._nodes[parent]['roots'])
        if not parents:
            if not isinstance(source, str) or not source:
                raise ValueError('root observation requires a source identity')
            roots.add(source)
        self._nodes[node_id] = {
            'id': node_id, 'case_id': self.case_id, 'kind': kind, 'payload': data,
            'parents': list(parents), 'roots': sorted(roots), 'status': 'active',
            'content_sha256': digest(data),
        }
        return self.get(node_id)

    def get(self, node_id: str) -> dict:
        return copy.deepcopy(self._nodes[node_id])

    def active(self, node_id: str) -> bool:
        return node_id in self._nodes and self._nodes[node_id]['status'] == 'active'

    def invalidate(self, node_id: str, reason: str) -> list[str]:
        if node_id not in self._nodes or not isinstance(reason, str) or not reason:
            raise ValueError('known node and explicit reason required')
        affected = {node_id}
        while True:
            children = {k for k, v in self._nodes.items()
                        if affected.intersection(v['parents'])}
            if children.issubset(affected):
                break
            affected.update(children)
        for key in sorted(affected):
            self._nodes[key]['status'] = 'invalid'
        self._events.append({'event': 'invalidate', 'root': node_id,
                             'affected': sorted(affected), 'reason': reason})
        return sorted(affected)

    def snapshot(self) -> dict:
        return {'case_id': self.case_id, 'nodes': copy.deepcopy(self._nodes),
                'events': copy.deepcopy(self._events),
                'roots_are_statistically_independent': False}


def validate_plan(actions: Sequence[Action], ledger: EvidenceLedger) -> None:
    available = {k: v['kind'] for k, v in ledger.snapshot()['nodes'].items()
                 if ledger.active(k)}
    reserved = set(ledger.snapshot()['nodes'])
    for action in actions:
        if action.id in reserved or action.operation not in CONTRACTS:
            raise ValueError('duplicate action or unregistered operation')
        expected_inputs, expected_output = CONTRACTS[action.operation]
        actual = tuple(available.get(key) for key in action.inputs)
        if actual != expected_inputs or action.output_kind != expected_output:
            raise ValueError('input type, dependency order or output type mismatch')
        available[action.id] = action.output_kind
        reserved.add(action.id)


def make_plan(region_ids: Sequence[str], *, retrieval=False) -> tuple[Action, ...]:
    result = []
    for index, region in enumerate(region_ids):
        crop = f'crop:{index}'
        result.append(Action(crop, 'crop', (region,), 'crop', 'select local observation'))
        result.append(Action(f'inspect:{index}', 'inspect', (crop,), 'observation',
                             'observe local appearance, not global disease absence'))
        if retrieval:
            result.append(Action(f'retrieve:{index}', 'retrieve', (crop,), 'analogy',
                                 'retrieve source analogies, not current patient facts'))
    return tuple(result)


class ToolUnavailable(ValueError):
    """Missing data or no compatible result, not evidence of medical absence."""


class InvalidObservation(ValueError):
    """A previously stored dependency fails an integrity/geometry check."""

    def __init__(self, node_id: str, reason: str):
        super().__init__(reason)
        self.node_id = node_id


def execute_plan(ledger: EvidenceLedger, actions: Sequence[Action],
                 handlers: Mapping[str, Callable], *, max_steps: int,
                 choose: Callable | None = None) -> dict:
    """A planner may choose only ready action IDs. GPU/runtime failures propagate.

    No dynamic code execution, hidden retries, or cross-case mutable memory.
    The runtime checkpoints complete cases, not partially executed GPU sessions.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('positive max_steps required')
    validate_plan(actions, ledger)
    attempted, failed, trace = set(), set(), []
    for _ in range(max_steps):
        ready = [a for a in actions if a.id not in attempted
                 and all(ledger.active(key) for key in a.inputs)]
        if not ready:
            break
        picked = ready[0].id if choose is None else choose(tuple(ready), ledger.snapshot())
        if picked == 'STOP':
            trace.append({'event': 'stop', 'reason': 'planner_stop'})
            break
        matches = [a for a in ready if a.id == picked]
        if len(matches) != 1:
            raise ValueError('planner selected an unavailable action')
        action = matches[0]
        attempted.add(action.id)
        started = perf_counter()
        try:
            if action.operation not in handlers:
                raise ToolUnavailable('no registered handler')
            payload = handlers[action.operation](action, [ledger.get(k) for k in action.inputs])
            ledger.add(action.id, action.output_kind, payload, parents=action.inputs)
            event = {'event': 'executed', 'action': asdict(action)}
        except InvalidObservation as exc:
            failed.add(action.id)
            affected = ledger.invalidate(exc.node_id, str(exc))
            event = {'event': 'invalidated', 'action': asdict(action),
                     'reason': str(exc), 'affected': affected}
        except (ValueError, TypeError, FileNotFoundError) as exc:
            failed.add(action.id)
            event = {'event': 'unavailable' if isinstance(exc, ToolUnavailable) else 'failed',
                     'action': asdict(action), 'reason': f'{type(exc).__name__}: {exc}'}
        event['seconds'] = perf_counter() - started
        trace.append(event)
    pending = [a.id for a in actions if a.id not in attempted]
    return {'ledger': ledger.snapshot(), 'trace': trace, 'attempted': sorted(attempted),
            'failed': sorted(failed), 'pending': pending,
            'budget_exhausted': len(attempted) >= max_steps and bool(pending)}


def commit_or_keep(incumbent: dict, candidate: dict | None, *, decision: str,
                   reason: str, dependencies_valid=True) -> dict:
    """UNKNOWN preserves incumbent text, token IDs and evidence exactly."""
    if decision not in {'accept', 'reject', 'unknown'} or not isinstance(reason, str) or not reason:
        raise ValueError('explicit three-way decision and reason required')
    if type(dependencies_valid) is not bool:
        raise TypeError('dependencies_valid must be boolean')
    accepted = decision == 'accept' and candidate is not None and dependencies_valid
    output = copy.deepcopy(candidate if accepted else incumbent)
    output['agent_commit'] = {
        'decision': decision if dependencies_valid else 'reject',
        'reason': reason if dependencies_valid else 'invalid_dependencies',
        'accepted': accepted, 'fallback': 'incumbent',
        'calibrated_probability': False, 'correctness_guaranteed': False,
    }
    return output
