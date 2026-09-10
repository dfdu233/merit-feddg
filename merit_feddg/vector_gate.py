"""Training-free, post-acquisition visual-contrast gate for vector evidence.

This is a frozen-model heuristic, NOT a correctness or clinical risk certificate.
Candidate continuations use the same original image/question/committed prefix.
The evidence-free verifier measures their relative support from actual pixels
versus a same-size mean-color control. Neither answers nor question types enter.
"""

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image, ImageStat

from .open_study import fingerprint


@dataclass(frozen=True)
class VectorGateConfig:
    probe_tokens: int = 8
    min_gain: float = 1e-6  # Numerical tie tolerance; not a fitted medical threshold.
    require_image_gain: bool = False

    def __post_init__(self):
        if type(self.require_image_gain) is not bool:
            raise ValueError("require_image_gain must be boolean")
        if type(self.probe_tokens) is not int or not 1 <= self.probe_tokens <= 64:
            raise ValueError("gate probe_tokens must be an integer in [1,64]")
        if type(self.min_gain) not in (int, float) or not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError("gate min_gain must be finite and nonnegative")


def mean_color_control(image):
    """Preserve dimensions/channel means while removing spatial image content."""
    from .experts.base import load_rgb

    native = load_rgb(image)
    color = tuple(round(v) for v in ImageStat.Stat(native).mean)
    return Image.new("RGB", native.size, color)


def token_log_probability(scores, token):
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or np.isnan(values).any()
            or np.isposinf(values).any() or not np.isfinite(values).any()
            or type(token) is not int or not 0 <= token < len(values)):
        raise ValueError("invalid verifier distribution/token")
    maximum = values.max()
    return float(values[token] - maximum - np.log(np.exp(values - maximum).sum()))


def assess_visual_contrast(base_session, evidence_session, image_session, control_session,
                           prefix, *, config, remaining_tokens, semantic_check=None):
    """Compare two unrestricted candidates; verification contains NO tool input.

    Every distinct verifier prefix is replayed once per image. Forward-step
    counts reflect next_scores' production replay (prefix length + 1), not just
    Python method calls. Candidate tokens are probes, never committed output.
    """
    started = perf_counter()
    prefix = tuple(prefix)
    length = min(config.probe_tokens, remaining_tokens)
    if length < 1:
        raise ValueError("no remaining answer budget for gate probe")
    audit = {"schema": "vector-visual-contrast-v1", "accepted": False,
             "gate_trained": False, "verifier_uses_expert_evidence": False,
             "correctness_guaranteed": False, "prefix_sha256": fingerprint(prefix),
             "probe_limit": length, "min_gain": config.min_gain,
             "candidate_generation_calls": 0, "candidate_generated_tokens": 0,
             "verifier_queries": 0, "estimated_replayed_forward_steps": 0}
    try:
        baseline = base_session.propose(prefix, count=1, length=length)[0]
        audit["candidate_generation_calls"] += 1
        audit["candidate_generated_tokens"] += len(baseline.tokens)
        candidate = evidence_session.propose(prefix, count=1, length=length)[0]
        audit["candidate_generation_calls"] += 1
        audit["candidate_generated_tokens"] += len(candidate.tokens)
        sequences = [tuple(baseline.tokens), tuple(candidate.tokens)]
        audit["candidate_token_ids"] = [list(v) for v in sequences]
        if any(not v or len(v) > length for v in sequences):
            raise ValueError("empty or over-budget gate continuation")
        if sequences[0] == sequences[1]:
            audit["reason"] = "no_change_within_probe_horizon"
            return audit
        if semantic_check is not None:
            audit["semantic_change"] = semantic_check(*sequences)
            if not audit["semantic_change"]["passed"]:
                audit["reason"] = "no_established_semantic_change"
                return audit
        cache = {}

        def mean_logp(session, image_key, tokens):
            if hasattr(session, "sequence_mean_logp"):
                audit["verifier_scoring"] = "raw_teacher_forced_sequence"
                audit["verifier_queries"] += 1
                return session.sequence_mean_logp(prefix, tokens)
            audit["verifier_scoring"] = "production_prefix_replay"
            total = 0.0
            for index, token in enumerate(tokens):
                context = prefix + tokens[:index]
                key = (image_key, context)
                if key not in cache:
                    audit["verifier_queries"] += 1
                    audit["estimated_replayed_forward_steps"] += len(context) + 1
                    cache[key] = session.next_scores(context)
                total += token_log_probability(cache[key], token)
            return total / len(tokens)

        support = []
        for tokens in sequences:
            real = mean_logp(image_session, "original", tokens)
            neutral = mean_logp(control_session, "mean_color", tokens)
            if not math.isfinite(real) or not math.isfinite(neutral):
                raise ValueError("nonfinite candidate likelihood")
            support.append({"image_mean_logp": real, "control_mean_logp": neutral,
                            "visual_support": real - neutral})
        gain = support[1]["visual_support"] - support[0]["visual_support"]
        image_gain = support[1]["image_mean_logp"] - support[0]["image_mean_logp"]
        audit.update(support=support, gain=gain, image_gain=image_gain,
                     control_gain=support[1]["control_mean_logp"] - support[0]["control_mean_logp"],
                     accepted=gain > config.min_gain,
                     reason="positive_visual_contrast_gain" if gain > config.min_gain
                     else "no_positive_visual_contrast_gain")
        audit["dimensions"] = {
            "original_image_gain": {"value": image_gain, "passed": image_gain > config.min_gain,
                                    "required": config.require_image_gain},
            "visual_contrast_gain": {"value": gain, "passed": gain > config.min_gain, "required": True},
        }
        if config.require_image_gain and image_gain <= config.min_gain:
            audit.update(accepted=False, reason="no_positive_original_image_gain")
    except (ValueError, TypeError, FloatingPointError) as exc:
        audit.update(accepted=False, reason=f"gate_invalid:{type(exc).__name__}:{exc}")
    finally:
        audit["seconds"] = perf_counter() - started
    return audit
