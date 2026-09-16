"""Dependency-free operator pilot; protocol is docs/graph_equivalence_protocol.md."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time


def fixture(seed):
    rng = random.Random(seed)
    n = rng.randrange(12, 25)
    facts = []

    def add(h, r, t):
        i = len(facts)
        facts.append(dict(id=f'f{i:04d}', head=f'v{h}', relation=r, tail=f'v{t}',
                          source=f'source_{i % 5}', timestamp=1000 + i,
                          polarity='positive' if i % 3 else 'negative'))

    add(0, 'query_first', 1)
    add(1, 'query_second', 2)
    for i in range(n):
        add(i, 'background_ring', (i + 1) % n)
    for _ in range(n * 2):
        h, t = rng.sample(range(n), 2)
        add(h, f'background_{rng.randrange(4)}', t)
    return facts


def encode(facts, mode):
    vertices, edges, events = set(), [], {}
    for f in facts:
        h, t = f['head'], f['tail']
        vertices.update((h, t))
        partial = int(hashlib.sha256(f['id'].encode()).hexdigest(), 16) % 2 == 0
        if mode == 'all' or (mode == 'partial' and partial):
            e = 'event:' + f['id']
            vertices.add(e)
            events[e] = {k: v for k, v in f.items() if k not in ('head', 'tail')}
            edges.extend([(h, e, 'subject', None), (e, t, 'object', None)])
        else:
            edges.append((h, t, 'direct', dict(f)))
    return dict(vertices=vertices, edges=edges, events=events)


def decode(g):
    rows = [dict(f) for _, _, role, f in g['edges'] if role == 'direct']
    for e, meta in g['events'].items():
        heads = [h for h, t, role, _ in g['edges'] if t == e and role == 'subject']
        tails = [t for h, t, role, _ in g['edges'] if h == e and role == 'object']
        assert len(heads) == len(tails) == 1
        rows.append(dict(meta, head=heads[0], tail=tails[0]))
    return sorted(rows, key=lambda f: f['id'])


def graph_query(g, start='v0'):
    # Independently traverse encoded edges; do not call decode or consume gold answers.
    current = {start}
    for relation in ('query_first', 'query_second'):
        following = set()
        for h, t, role, f in g['edges']:
            if h not in current:
                continue
            if role == 'direct' and f['relation'] == relation:
                following.add(t)
            elif role == 'subject' and g['events'][t]['relation'] == relation:
                following.update(b for a, b, r, _ in g['edges'] if a == t and r == 'object')
        current = following
    return current


def pagerank(g, seed):
    nodes = sorted(g['vertices'])
    adj = {n: [] for n in nodes}
    for h, t, _, _ in g['edges']:
        adj[h].append(t)
    assert all(adj.values()), 'fixture deliberately excludes dangling vertices'
    p = {n: float(n == seed) for n in nodes}
    for iteration in range(2000):
        nxt = {n: (0.15 if n == seed else 0.) for n in nodes}
        for h in nodes:
            share = 0.85 * p[h] / len(adj[h])
            for t in adj[h]:
                nxt[t] += share
        residual = math.fsum(abs(nxt[n] - p[n]) for n in nodes)
        p = nxt
        if residual < 1e-12:
            break
    else:
        raise AssertionError('PPR failed to converge')
    assert abs(math.fsum(p.values()) - 1.) < 1e-10
    return p, iteration + 1


def ppr_retrieve(g, facts, seed='v0'):
    p, steps = pagerank(g, seed)
    entities = {f[k] for f in facts for k in ('head', 'tail')}
    mass = math.fsum(p[n] for n in entities)
    conditional = {n: p[n] / mass for n in entities}
    score = {f['id']: conditional[f['head']] + conditional[f['tail']] for f in facts}
    selected = sorted(score, key=lambda i: (-score[i], i))[:6]
    return selected, conditional, steps


def reach(g, source_budget):
    # Dijkstra with 0/1 costs. Relation entry costs zero only under source-fact budget.
    import heapq
    adj = {n: [] for n in g['vertices']}
    for h, t, role, f in g['edges']:
        fid = f['id'] if role == 'direct' else (g['events'][h]['id'] if role == 'object' else None)
        cost = 0 if source_budget and role == 'subject' else 1
        adj[h].append((t, cost, fid))
    queue, distance, found = [(0, 'v0')], {'v0': 0}, set()
    while queue:
        d, h = heapq.heappop(queue)
        if d != distance[h]:
            continue
        for t, cost, fid in adj[h]:
            nd = d + cost
            if nd > 2:
                continue
            if fid is not None:
                found.add(fid)
            if nd < distance.get(t, float('inf')):
                distance[t] = nd
                heapq.heappush(queue, (nd, t))
    return found


def rename(g, facts):
    mapping = {n: f'renamed_{i:04d}' for i, n in enumerate(reversed(sorted(g['vertices'])))}
    out = copy.deepcopy(g)
    out['vertices'] = {mapping[n] for n in g['vertices']}
    out['events'] = {mapping[n]: m for n, m in g['events'].items()}
    out['edges'] = []
    for h, t, role, f in reversed(g['edges']):
        mapped_f = dict(f, head=mapping[f['head']], tail=mapping[f['tail']]) if f else None
        out['edges'].append((mapping[h], mapping[t], role, mapped_f))
    mapped = [dict(f, head=mapping[f['head']], tail=mapping[f['tail']]) for f in facts]
    return out, mapped, mapping


def distance(a, b):
    a, b = set(a), set(b)
    return 1 - len(a & b) / len(a | b) if a | b else 0.


def main():
    started = time.perf_counter()
    rows, fixtures = [], []
    for seed in range(64):
        facts = fixture(seed)
        fixtures.append(dict(seed=seed, facts=facts, query=['v0', 'query_first', 'query_second']))
        base = encode(facts, 'direct')
        base_sel, base_p, _ = ppr_retrieve(base, facts)
        base_reach = reach(base, False)
        assert graph_query(base) == {'v2'}
        for mode in ('direct', 'all', 'partial'):
            g = encode(facts, mode)
            decoded = decode(g)
            assert decoded == facts, 'source/qualifier/identity round-trip failed'
            assert graph_query(g) == {'v2'}, 'query semantics changed'
            assert json.dumps(decoded, sort_keys=True) == json.dumps(facts, sort_keys=True)
            selected, p, steps = ppr_retrieve(g, facts)
            canonical = encode(decoded, 'direct')
            csel, cp, _ = ppr_retrieve(canonical, decoded)
            assert csel == base_sel and cp == base_p
            assert reach(canonical, False) == base_reach
            assert reach(g, True) == base_reach
            renamed, mapped_facts, mapping = rename(g, facts)
            rsel, rp, _ = ppr_retrieve(renamed, mapped_facts, mapping['v0'])
            rename_error = max(abs(p[n] - rp[mapping[n]]) for n in p)
            assert rename_error < 1e-10 and set(rsel) == set(selected)
            wrong = copy.deepcopy(facts)
            wrong[1]['tail'] = 'v3'
            assert graph_query(encode(wrong, mode)) == {'v3'}
            lost = copy.deepcopy(facts)
            lost[0]['polarity'] = 'positive'
            assert decode(encode(lost, mode)) != facts
            support = {'f0000', 'f0001'}
            rows.append(dict(seed=seed, mode=mode, source_facts=len(facts),
                             vertices=len(g['vertices']), edges=len(g['edges']),
                             roundtrip=True, query_equivalence=True,
                             wrong_tail_detected=True, lost_qualifier_detected=True,
                             ppr_selected=selected, base_selected=base_sel,
                             ppr_jaccard_distance=distance(selected, base_sel),
                             entity_l1=math.fsum(abs(p[n]-base_p[n]) for n in p),
                             support_recall=len(set(selected) & support)/2,
                             base_support_recall=len(set(base_sel) & support)/2,
                             both_support=int(support <= set(selected)),
                             ppr_iterations=steps,
                             canonical_jaccard_distance=distance(csel, base_sel),
                             rename_max_error=rename_error,
                             raw_reach_distance=distance(reach(g, False), base_reach),
                             raw_reach_count=len(reach(g, False)),
                             base_reach_count=len(base_reach),
                             fact_budget_reach_distance=distance(reach(g, True), base_reach)))
    summary = dict(experiment='graph-equivalence-dev-20260916-v1', synthetic=True,
                   official_reproduction=False, gpu_used=False,
                   semantic_pairs=len(rows), semantic_failures=0,
                   elapsed_seconds=time.perf_counter()-started, modes={})
    for mode in ('direct', 'all', 'partial'):
        part = [r for r in rows if r['mode'] == mode]
        summary['modes'][mode] = dict(cases=len(part), changed_top6=sum(r['ppr_jaccard_distance'] > 0 for r in part))
        for key in ('ppr_jaccard_distance', 'entity_l1', 'support_recall', 'base_support_recall',
                    'both_support', 'canonical_jaccard_distance', 'raw_reach_distance',
                    'raw_reach_count', 'base_reach_count', 'fact_budget_reach_distance', 'rename_max_error'):
            summary['modes'][mode][key + '_mean'] = statistics.mean(r[key] for r in part)
    out = Path(__file__).resolve().parents[1] / 'runs/graph-equivalence-dev-20260916-v1'
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [('fixtures.json', fixtures), ('pairs.json', rows), ('summary.json', summary)]:
        (out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
