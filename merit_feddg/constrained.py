"""Finite tool-action decoding. Medical answers remain unconstrained free text."""

from __future__ import annotations


def finite_choice_constraint(tokenizer, choices, prompt_length=None, eos_token_id=None):
    if not choices or len(set(choices)) != len(choices):
        raise ValueError("nonempty unique action strings required")
    encoded = [tuple(tokenizer.encode(c, add_special_tokens=False)) for c in choices]
    if any(not ids for ids in encoded) or len(set(encoded)) != len(encoded):
        raise ValueError("action strings must have distinct nonempty token encodings")
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if not eos:
        raise ValueError("finite action decoding requires EOS")
    offsets = {}

    def allowed(batch_id, input_ids):
        sequence = input_ids.tolist()
        if batch_id not in offsets:
            offsets[batch_id] = len(sequence) if prompt_length is None else prompt_length
        suffix = tuple(sequence[offsets[batch_id]:])
        following = set()
        for tokens in encoded:
            if tokens[:len(suffix)] == suffix:
                following.update(eos if len(tokens) == len(suffix) else [tokens[len(suffix)]])
        if not following:
            raise ValueError("decoder departed from finite action grammar")
        return sorted(following)

    return allowed
