"""Continue a malformed critic explanation with a bounded finite control label.

No new comparison, reference, image, answer order, or free-text answer generation.
The original model inputs and generated token prefix are retained, except EOS.
"""
import time

from llava_critic_backend import LlavaCritic


def continuation_prefix(token_ids, eos_ids, suffix_ids, total_budget):
    tokens = list(token_ids)
    while tokens and tokens[-1] in eos_ids:
        tokens.pop()
    prefix = tokens + list(suffix_ids)
    remaining = total_budget - len(prefix)
    if remaining < 2:
        raise ValueError('No original budget left for label/EOS; do not truncate explanation')
    return prefix, min(8, remaining)


class CompletingCritic(LlavaCritic):
    _continuation = None

    def _inputs(self, image_path, prompt):
        ids, pixels, size, upper = super()._inputs(image_path, prompt)
        if self._continuation is not None:
            extra = self.torch.tensor([self._continuation], device=ids.device, dtype=ids.dtype)
            ids = self.torch.cat([ids, extra], dim=1)
            upper += len(self._continuation)
        return ids, pixels, size, upper

    def complete_verdict(self, image, prompt, response, total_budget):
        suffix = self.tokenizer.encode('\nFinal verdict: ', add_special_tokens=False)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        eos.append(self.tokenizer.eos_token_id)
        prefix, budget = continuation_prefix(response['token_ids'], eos, suffix, total_budget)
        start = time.perf_counter()
        self._continuation = prefix
        try:
            value = self.generate_with_usage(image, prompt, budget, allowed_texts=('A','B','C'))
        finally:
            self._continuation = None
        return {'stage':'verdict_format_completion', 'seconds':time.perf_counter()-start,
                'usage':value, 'original_prefix_tokens':len(response['token_ids']),
                'continued_prefix_tokens':len(prefix), 'original_total_budget':total_budget,
                'same_image_question_candidates_order':True, 'new_comparison':False}
