"""Native Huatuo receiver for BARD probes; private incremental KV per branch."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image


class HuatuoReceiver:
    def __init__(self, spec):
        sys.path.insert(0, spec['anchor_root'])
        from anchor.corrected_sgta.models_oe import HuatuoOEAdapter
        self.adapter = HuatuoOEAdapter(Path(spec['checkpoint_path']), Path(spec['source_path']))
        self.model = self.adapter.model.eval().requires_grad_(False)
        self.tokenizer = self.adapter.tokenizer
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.torch = torch
        self.settings = dict(spec['generation_settings'])
        self._optimization = self.adapter._generation_last_token_logits()
        self._optimization.__enter__()

    def inputs(self, image, prompt):
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                image = im.convert('RGB')
        return self.adapter._inputs(image, prompt)

    def context_token_budget(self, image, prompt, reserve_tokens):
        from llava.constants import IMAGE_TOKEN_INDEX
        ids, _pixels = self.inputs(image, prompt)
        tower = self.model.get_vision_tower()
        count = ids.shape[1] + int((ids == IMAGE_TOKEN_INDEX).sum()) * (tower.num_patches - 1)
        limits = [getattr(self.model.config, name, None) for name in
                  ('max_position_embeddings', 'tokenizer_model_max_length')]
        limit = min(int(x) for x in limits if x is not None)
        return {'input_tokens': count, 'reserved_tokens': reserve_tokens, 'context_limit': limit,
                    'remaining_tokens': limit-count-reserve_tokens, 'fits': count+reserve_tokens <= limit}

    def new_answer_session(self, image, prompt):
        return HuatuoSession(self, image, prompt)

    def generate_with_usage(self, image, prompt, max_new_tokens):
        ids, pixels = self.inputs(image, prompt)
        settings = dict(self.settings, max_new_tokens=max_new_tokens)
        with torch.inference_mode():
            result = self.model.generate(ids, images=pixels, **settings)
        # Native Huatuo generate passes inputs_embeds to HF; its output starts at BOS.
        tokens = result[0].tolist()
        bos = self.model.generation_config.bos_token_id
        if tokens and tokens[0] == bos:
            tokens = tokens[1:]
        return {'text': self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'token_ids': tokens}


class HuatuoSession:
    def __init__(self, probe, image, prompt):
        self.probe, self.image, self.prompt = probe, image, prompt
        self.cache, self.consumed = None, ()
        eos = probe.settings['eos_token_id']
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def decode(self, tokens):
        return self.probe.tokenizer.decode(tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def next_scores(self, prefix):
        p = self.probe
        if self.cache is None:
            if prefix:
                raise ValueError('Initial branch prefix must be empty')
            ids, pixels = p.inputs(self.image, self.prompt)
            marks = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            _, positions, mask, _, embeds, aligned = p.model.prepare_inputs_labels_for_multimodal_new(
                ids, None, None, None, marks, pixels)
            from llava.constants import IMAGE_TOKEN_INDEX
            if not torch.equal(aligned[aligned != -100], marks[ids != IMAGE_TOKEN_INDEX]):
                raise ValueError('Native multimodal input truncated or reordered')
            self.length, self.device = embeds.shape[1], embeds.device
            if positions is None:
                positions = torch.arange(self.length, device=self.device).unsqueeze(0)
            if mask is None:
                mask = torch.ones((1, self.length), device=self.device, dtype=torch.long)
            kwargs = {'inputs_embeds': embeds, 'position_ids': positions, 'attention_mask': mask}
        else:
            if tuple(prefix[:-1]) != self.consumed or len(prefix) != len(self.consumed)+1:
                raise ValueError('Branch must consume the same single committed token')
            kwargs = {'input_ids': torch.tensor([[prefix[-1]]], device=self.device),
                          'position_ids': torch.tensor([[self.length+len(prefix)-1]], device=self.device),
                          'attention_mask': torch.ones((1, self.length+len(prefix)), device=self.device, dtype=torch.long),
                          'past_key_values': self.cache}
        out = p.model(**kwargs, use_cache=True, output_attentions=False, return_dict=True)
        self.cache, self.consumed = out.past_key_values, tuple(prefix)
        scores = out.logits[0, -1].float().clone()
        bos = p.model.generation_config.bos_token_id
        seen = sorted(set(([bos] if bos is not None else []) + list(prefix)))
        if seen:
            ids = torch.tensor(seen, device=scores.device)
            old = scores[ids]
            penalty = p.settings['repetition_penalty']
            scores[ids] = torch.where(old < 0, old*penalty, old/penalty)
        if len(prefix) < p.settings['min_new_tokens']:
            scores[list(self.eos_ids)] = -torch.inf
        return scores.cpu().numpy()
