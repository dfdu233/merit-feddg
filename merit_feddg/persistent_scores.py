"""Opt-in semantic-only production-prefill cache; no shared model/file edits."""
import numpy as np


class PersistentScores:
    def __init__(self, session):
        if session.tensor_packet is not None:
            raise ValueError('persistent scorer currently supports semantic evidence only')
        self.session = session
        self.prefix = None
        self.kwargs = None
        self.scores = None
        self.prefills = 0
        self.incremental_calls = 0
        cfg = session.generalist.model.generation_config
        for name in ('suppress_tokens', 'watermarking_config', 'guidance_scale'):
            if getattr(cfg, name, None) is not None:
                raise ValueError('unsupported generation option: ' + name)
        for name in ('renormalize_logits', 'remove_invalid_values', 'token_healing'):
            if getattr(cfg, name, False):
                raise ValueError('unsupported generation option: ' + name)

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _prefill(self):
        model = self.session.generalist.model
        original = model._update_model_kwargs_for_generation
        local = model.__dict__.get('_update_model_kwargs_for_generation')
        captured = []

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(dict(result))
            return result

        model._update_model_kwargs_for_generation = capture
        try:
            # Runs the existing validation and EXACT production vision/prefill.
            scores = self.session.next_scores(())
        finally:
            if local is None:
                del model._update_model_kwargs_for_generation
            else:
                model._update_model_kwargs_for_generation = local
        if len(captured) != 1 or captured[0].get('past_key_values') is None:
            raise RuntimeError('expected exactly one production cached prefill')
        self.kwargs = captured[0]
        self.prefix, self.scores = (), scores
        self.prefills += 1

    def next_scores(self, prefix):
        prefix = tuple(int(t) for t in prefix)
        if self.prefix is None or prefix[:len(self.prefix)] != self.prefix:
            self.kwargs = None
            self._prefill()
        torch, model = self.session.generalist.torch, self.session.generalist.model
        while len(self.prefix) < len(prefix):
            target = prefix[:len(self.prefix)+1]
            ids = torch.tensor([target], dtype=torch.long,
                               device=self.session.inputs['inputs'].device)
            with torch.inference_mode():
                inputs = model.prepare_inputs_for_generation(ids, **self.kwargs)
                result = model(**inputs, return_dict=True)
                self.kwargs = model._update_model_kwargs_for_generation(
                    result, self.kwargs, is_encoder_decoder=model.config.is_encoder_decoder)
                self.scores = result.logits[:, -1, :].to(
                    copy=True, dtype=torch.float32, device=ids.device)[0].cpu().numpy()
            self.prefix = target
            self.incremental_calls += 1
        return self.scores.copy()
