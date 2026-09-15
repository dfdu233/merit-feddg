"""Frozen same-prefix soft decoding probe; no class-to-token logits shortcut.

Uses production replay for correctness before implementing faster dual KV caches.
The evidence branch cannot choose its own prefix. Alpha is a strength, not trust.
"""
import math
from time import perf_counter

import numpy as np


def mix_scores(base, conditioned, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('frozen interpolation strength must be in [0,1]')
    a, b = np.asarray(base, dtype=np.float64), np.asarray(conditioned, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('same finite vocabulary required')
    # Equivalent, up to a token-independent constant, to log p0 + alpha(log p1-log p0).
    return a if alpha == 0 else b if alpha == 1 else (1-alpha)*a + alpha*b


def decode(base, conditioned, *, alpha, max_tokens, eos_ids):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or max_tokens < 1:
        raise ValueError('invalid frozen decoding options')
    start = perf_counter()
    if alpha == 0:
        block = base.propose((), count=1, length=max_tokens)[0]
        return {'text': block.text, 'token_ids': list(block.tokens),
                'seconds': perf_counter()-start, 'conditioned_score_calls': 0,
                'base_score_calls': 0, 'zero_fast_path': True, 'steps': []}
    prefix, steps = [], []
    for _ in range(max_tokens):
        # Each branch replays this exact committed prefix with its own context.
        a = base.next_scores(tuple(prefix))
        b = conditioned.next_scores(tuple(prefix))
        logits = mix_scores(a, b, alpha)
        token = int(np.argmax(logits))
        steps.append({'base_top': int(np.argmax(a)), 'conditioned_top': int(np.argmax(b)),
                      'selected': token, 'max_logit_delta': float(np.max(np.abs(a-b)))})
        prefix.append(token)
        if token in eos_ids:
            break
    return {'text': base.decode(prefix).strip(), 'token_ids': prefix,
            'seconds': perf_counter()-start, 'conditioned_score_calls': len(steps),
            'base_score_calls': len(steps), 'zero_fast_path': False, 'steps': steps,
            'cache_policy': 'exact-production-prefix-replay; intentionally not throughput optimized'}
