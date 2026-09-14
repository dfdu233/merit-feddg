"""Offline HTML audit. Patient text, paths, images and native payloads excluded by default."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def short(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]


def render(outputs, include_text=False):
    sections = []
    for method, cases in outputs.items():
        for case_id, answer in cases.items():
            w = answer.get('agent_workflow', {})
            delivery = w.get('delivery', {})
            gate = {a['ref']: a for a in delivery.get('gate_audit', [])}
            shown = set(answer.get('presented_observation_refs', []))
            observations = []
            for o in w.get('observations', []):
                a, g = o['artifact'], gate.get(o['ref'], {})
                observations.append({'source_id_hash':short(a['expert_id']),
                    'scope_hash':short(a['scope']), 'observation_hash':short(o['ref']),
                    'parent_hash':short(o['content']['parent']) if o['content'].get('parent') else None,
                    'capability':a['capability'] if a['capability'] in
                        {'classification','segmentation','detection','generation','retrieval'} else 'unknown',
                    'gate_use':g.get('use','not_run') if g.get('use','not_run') in
                        {'ANSWER','AUXILIARY','IRRELEVANT','UNKNOWN','not_run'} else 'UNKNOWN',
                    'scope_allowed':g.get('scope',{}).get('allowed'),
                    'actually_presented':o['ref'] in shown})
            regions = []
            for e in w.get('events', []):
                box = e.get('action',{}).get('region') or {}
                coords = box.get('box')
                if isinstance(coords, list) and len(coords)==4 and all(type(v) in (int,float) and 0 <= v <= 1 for v in coords):
                    regions.append(coords)
            record = {'method':method if method in {'incumbent','no_new_gate','scope_restored','purpose_gate'} else short(method),
                'case_hash':short(case_id), 'answer_sha256':short(answer.get('text','')),
                'answer':answer.get('text','') if include_text else '[redacted: explicit local opt-in required]',
                'tokens':len(answer.get('token_ids',[])), 'candidate_generated':w.get('candidate') is not None,
                'observations':observations, 'predicted_regions_normalized_xyxy':regions,
                'model_calls':sum(c.get('model_generation_started',False) for c in w.get('calls',[])),
                'seconds':answer.get('seconds'), 'new_seconds':answer.get('new_seconds'),
                'inherited_cost_seconds':answer.get('inherited_cost_upper_bound_seconds')}
            sections.append('<section><pre>'+html.escape(json.dumps(record, indent=2))+'</pre></section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>Evidence purpose audit</title><style>body{font:14px monospace;max-width:1000px;margin:auto}'
            'section{border-top:1px solid #bbb}pre{white-space:pre-wrap}</style>'
            '<h1>Evidence purpose audit</h1><p>Admission is NOT medical correctness. '
            'Predicted regions are NOT confirmed lesions. No patient images exported. '
            'Default view hashes source/scope identifiers and redacts answers.</p>'
            + ''.join(sections) + '</html>')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, help='plug_run shard/results.json')
    p.add_argument('--output', required=True)
    p.add_argument('--include-text', action='store_true', help='Sensitive LOCAL ONLY export; never commit')
    args = p.parse_args()
    data = json.loads(Path(args.results).read_text())
    with Path(args.output).open('x', encoding='utf-8') as f:
        f.write(render(data['outputs'], args.include_text))
    print('wrote offline HTML; images excluded')


if __name__ == '__main__':
    main()
