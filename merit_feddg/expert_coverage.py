"""Opt-in, deterministic capability coverage; no fit, labels, or correctness gate.

Uses the existing question parser unchanged. Unknown is reported and may request
an unverified general observation; it is never treated as an incorrect question.
All catalogs are fixed before evaluation, from audited upstream inference code.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import yaml

from .capability_contracts import question_semantics
from .capability_routing import question_type


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get('schema') != 'expert-coverage-v1' or config['max_calls'] < 1:
        raise ValueError('invalid coverage configuration')
    return config


def source_catalog(config, card):
    if 'catalog' in card:
        return card['catalog']
    # Parse data literals, do not execute upstream training/evaluation code.
    tree = ast.parse((Path(config['unimed_source']) / 'constants.py').read_text())
    candidates = [node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == card['catalog_symbol']
                          for t in node.targets)]
    if len(candidates) != 1:
        raise ValueError('upstream catalog symbol missing or ambiguous')
    raw = ast.literal_eval(candidates[0].value)
    if not isinstance(raw, dict) or not raw:
        raise ValueError('expected a fixed upstream prompt dictionary')
    # First published template is frozen, not selected by benchmark results.
    # This is NOT the paper's multi-template ensemble or a calibrated classifier.
    return [{'name': name, 'prompt': prompts[0]} for name, prompts in raw.items()]


def new_specs(config, device='cpu'):
    result = {}
    for name, card in config['new'].items():
        kind = card['kind']
        kwargs = {'kind': kind, 'expert_id': name, 'scope': card['scope'],
                  'modalities': card['modalities'], 'catalog': source_catalog(config, card),
                  'source_family': card['source_family'], 'device': device}
        if kind in ('flair', 'unimed'):
            kwargs['source_path'] = config[f'{kind}_source']
            kwargs['text_path'] = str(Path(config['model_root']) /
                                      ('clinicalbert' if kind == 'flair' else 'biomedbert'))
        result[name] = {'id': str(Path(config['model_root']) / kind),
                        'factory': 'merit_feddg.experts.native_coverage:CatalogExpert',
                        'factory_kwargs': kwargs, 'modalities': card['modalities'],
                        'tasks': ['open_vqa'], 'capabilities': ['classification'],
                        'scope': card['scope'], 'requires_region': False,
                        'description': 'Uncalibrated fixed-catalog image/text matches only'}
    return result


def phrase_present(text, phrase):
    # General token-boundary matching of public capability-card vocabulary.
    words = ' '.join(re.findall(r'[a-z0-9]+', text.lower()))
    target = ' '.join(re.findall(r'[a-z0-9]+', phrase.lower()))
    return bool(target) and f' {target} ' in f' {words} '


def select_experts(config, row, available, *, requested=None):
    parsed = sorted(question_semantics(row['question']))
    needs = set(parsed if requested is None else requested)
    if not needs:
        needs = {'observation'}
    cards = {**config['legacy'], **config['new']}
    audits, candidates = {}, []
    for name, card in cards.items():
        covered = needs.intersection(card['attributes'])
        reason = 'eligible'
        if row.get('input_kind', '2d') != '2d':
            reason = 'unsupported_input_kind'
        elif row['modality'] not in card['modalities']:
            reason = 'incompatible_modality'
        elif not covered:
            reason = 'no_supported_requested_attribute'
        elif card.get('sites') and not any(phrase_present(row['question'], site)
                                           for site in card['sites']):
            reason = 'required_site_or_entity_not_established'
        elif name not in available:
            reason = 'resource_or_native_output_unavailable'
        audits[name] = {'reason': reason, 'covered_attributes': sorted(covered),
                        'source_family': card['source_family'], 'selected': False}
        if reason == 'eligible':
            candidates.append((name, card, covered))
    selected, remaining = [], set(needs)
    while remaining and candidates and len(selected) < config['max_calls']:
        candidates.sort(key=lambda x: (-len(x[2] & remaining), x[1]['priority'], x[0]))
        name, card, covered = candidates.pop(0)
        if not (covered & remaining):
            break
        selected.append(name)
        audits[name]['selected'] = True
        remaining -= covered
    return {'selected': selected, 'requested_attributes': sorted(needs),
            'uncovered_attributes': sorted(remaining), 'old_question_type': question_type(row['question']),
            'automatic_semantics': parsed, 'explicit_request': requested is not None,
            'audit': audits, 'confidence': None,
            'policy': 'unweighted-marginal-attribute-coverage/frozen-priority-v1',
            'warning': 'Eligibility and coverage, not prediction correctness or calibrated utility.'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/expert_coverage_v1.yaml')
    parser.add_argument('--modality', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--input-kind', default='2d')
    args = parser.parse_args()
    config = load_config(args.config)
    result = select_experts(config, {'question': args.question, 'modality': args.modality,
                                    'input_kind': args.input_kind},
                            set(config['legacy']) | set(config['new']))
    result['resource_check_performed'] = False
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
