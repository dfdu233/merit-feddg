"""Opt-in native receiver KV replay; validate each backend before use."""

import numpy as np


class CachedHuatuoSession:
    def __init__(self, session):
        self.session = session
        self.cache = None
        self.consumed = None
        self.scores = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def next_scores(self, prefix):
        prefix = tuple(prefix)
        if prefix == self.consumed:
            return self.scores.copy()
        if self.consumed is None or prefix[:-1] != self.consumed:
            self.cache = None
            self._step(())
            for end in range(1, len(prefix) + 1):
                self._step(prefix[:end])
        else:
            self._step(prefix)
        return self.scores.copy()

    def _step(self, prefix):
        p = self.generalist
        torch = p.torch
        with torch.inference_mode():
            if self.cache is None:
                ids = self.inputs["inputs"]
                with self._evidence_context():
                    _, positions, mask, _, embeds, _ = (
                        p.model.prepare_inputs_labels_for_multimodal_new(
                            ids, None, None, None, None, self.inputs["images"]
                        )
                    )
                self.length, self.device = embeds.shape[1], embeds.device
                if positions is None:
                    positions = torch.arange(self.length, device=self.device).unsqueeze(0)
                if mask is None:
                    mask = torch.ones((1, self.length), device=self.device, dtype=torch.long)
                kwargs = {
                    "inputs_embeds": embeds,
                    "position_ids": positions,
                    "attention_mask": mask,
                }
            else:
                if not prefix or tuple(prefix[:-1]) != self.consumed:
                    raise ValueError("KV branch prefix mismatch")
                kwargs = {
                    "input_ids": torch.tensor([[prefix[-1]]], device=self.device),
                    "position_ids": torch.tensor(
                        [[self.length + len(prefix) - 1]], device=self.device
                    ),
                    "attention_mask": torch.ones(
                        (1, self.length + len(prefix)), device=self.device, dtype=torch.long
                    ),
                    "past_key_values": self.cache,
                }
            output = p.model(**kwargs, use_cache=True, output_attentions=False, return_dict=True)
            self.cache, self.consumed = output.past_key_values, tuple(prefix)
            scores = output.logits[0, -1].float().clone()
            bos = p.model.generation_config.bos_token_id
            seen = sorted(set(([bos] if bos is not None else []) + list(prefix)))
            if seen:
                indices = torch.tensor(seen, device=scores.device)
                values = scores[indices]
                scores[indices] = torch.where(
                    values < 0, values * p.repetition_penalty, values / p.repetition_penalty
                )
            if len(prefix) < p.min_new_tokens:
                scores[list(self.eos_ids)] = -torch.inf
            # Production forced-prefix replay masks padded model-only vocabulary IDs.
            if prefix:
                scores[len(p.tokenizer):] = -torch.inf
            self.scores = scores.cpu().numpy()
            if np.isnan(self.scores).any() or np.isposinf(self.scores).any():
                raise RuntimeError("Invalid cached receiver scores")


class CachedLlavaSession:
    """Adapt the existing LLaVA persistent stream to exact-prefix requests."""

    def __init__(self, session):
        self.session = session
        self.stream = None
        self.consumed = ()

    def __getattr__(self, name):
        return getattr(self.session, name)

    def share_vision_cache_from(self, other):
        self.session.share_vision_cache_from(getattr(other, "session", other))

    def next_scores(self, prefix):
        prefix = tuple(prefix)
        if self.stream is None or prefix[:len(self.consumed)] != self.consumed:
            self.stream = self.session.new_incremental_stream()
            self.consumed = ()
        for token in prefix[len(self.consumed):]:
            self.stream.commit(token)
        self.consumed = prefix
        scores = self.stream.current_scores()
        if prefix:
            scores[len(self.generalist.tokenizer):] = -np.inf
        return scores


def enable_cached_sessions(probe):
    original_plain = probe.new_answer_session
    original_spatial = probe.new_tensor_answer_session
    def wrap(session):
        wrapper = CachedLlavaSession if hasattr(session, "new_incremental_stream") else CachedHuatuoSession
        return wrapper(session)

    probe.new_answer_session = lambda *args, **kwargs: wrap(original_plain(*args, **kwargs))
    probe.new_tensor_answer_session = lambda *args, **kwargs: wrap(original_spatial(*args, **kwargs))
