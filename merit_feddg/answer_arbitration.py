"""Frozen external image-text verification of a complete proposed answer edit.

CheXzero-inspired paired image/text comparisons, not a calibrated utility model.
No target references, CE/OE grammar, disease dictionary, learned fusion or null
image. Two fixed renderings expose prompt-sensitive preferences; unanimity is a
conservative heuristic, NOT a risk certificate or independent repeated trials.
"""
import copy
from difflib import SequenceMatcher
from time import perf_counter

import numpy as np


def changed_spans(before, after):
    a, b = before.split(), after.split()
    return [{'before': ' '.join(a[i:j]), 'after': ' '.join(b[k:l])}
            for tag, i, j, k, l in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
            if tag != 'equal']


def render_pairs(question, baseline, candidate):
    # Both renderings include the entire question and answer. Changed spans are
    # audit-only: isolated "No"/"Left" must never lose their question binding.
    return [[f'Question: {question}\nAnswer: {answer}' for answer in (baseline, candidate)],
            [f'For the question "{question}", the image supports: {answer}'
             for answer in (baseline, candidate)]]


class FrozenImageTextVerifier:
    def __init__(self, adapter, *, model_id, modalities):
        if not model_id or not modalities or '*' in modalities:
            raise ValueError('verifier needs an explicit model identity and modality scope')
        self.adapter, self.model_id, self.modalities = adapter, model_id, frozenset(modalities)
        adapter.model.eval()
        for parameter in adapter.model.parameters():
            parameter.requires_grad_(False)

    def score(self, image, texts):
        adapter = self.adapter
        from .experts.base import load_rgb
        with adapter.torch.inference_mode():
            if hasattr(adapter, '_text_embeddings'):
                # The current BiomedCLIP HF tokenizer truncates silently. Refuse
                # inputs that cannot be scored intact instead of verifying a prefix.
                raw_tokenizer = getattr(adapter.tokenizer, 'tokenizer', None)
                if raw_tokenizer is None:
                    raise ValueError('cannot establish verifier text truncation boundary')
                if any(len(raw_tokenizer.encode(t, add_special_tokens=True)) > 256 for t in texts):
                    raise ValueError('verifier text exceeds intact context budget')
                text_features = adapter._text_embeddings(texts)
            else:
                tokens = adapter.tokenize(adapter.tokenizer, texts, truncate=False).to(adapter.device)
                text_features = adapter.model.encode_text(tokens, normalize=True)
            image_features = adapter._image_embedding(load_rgb(image))
            return (image_features @ text_features.T).squeeze(0).float().cpu().numpy()


def load_verifier(spec, artifacts):
    from .generalist_factory import _local_or_remote
    source = spec.get('checkpoint_path') or _local_or_remote(spec['id'], artifacts)
    if spec['adapter'] == 'contrastive_biomedclip':
        from .experts.biomedclip import BiomedClipAdapter
        adapter = BiomedClipAdapter(source, device=spec.get('device', 'auto'))
    elif spec['adapter'] == 'conch':
        from .experts.conch import ConchConceptExpert
        adapter = ConchConceptExpert(source, device=spec.get('device', 'auto'))
    else:
        raise ValueError('unsupported frozen image-text verifier adapter')
    return FrozenImageTextVerifier(adapter, model_id=spec['id'], modalities=spec['modalities'])


def assess_revision(baseline, candidate, *, image, question, modality, verifier,
                    generalist_id, source_model_ids=(), min_margin=1e-6):
    if not np.isfinite(min_margin) or min_margin < 0:
        raise ValueError('margin must be a fixed nonnegative numerical tolerance')
    start = perf_counter()
    audit = {'decision': 'abstain', 'reason': 'unverified', 'trained': False,
             'calibrated_probability': False, 'correctness_guaranteed': False,
             'score_calls': 0, 'text_pairs': 0,
             'changed_spans': changed_spans(baseline, candidate)}
    try:
        if baseline == candidate:
            audit.update(decision='unchanged', reason='identical_answer')
        elif verifier is None or modality not in verifier.modalities:
            audit['reason'] = 'no_applicable_external_verifier'
        elif verifier.model_id in {generalist_id, *source_model_ids}:
            audit['reason'] = 'verifier_reuses_generator_or_evidence_checkpoint'
        else:
            pairs = render_pairs(question, baseline, candidate)
            audit.update(verifier_id=verifier.model_id, text_pairs=len(pairs), score_calls=1)
            scores = np.asarray(verifier.score(image, [text for pair in pairs for text in pair]), dtype=float)
            if scores.shape != (4,) or not np.isfinite(scores).all():
                raise ValueError('invalid paired verification scores')
            margins = scores[1::2] - scores[::2]
            audit.update(paired_scores=scores.reshape(2, 2).tolist(), margins=margins.tolist(),
                         min_margin=min_margin)
            if np.all(margins > min_margin):
                audit.update(decision='accept', reason='consistent_external_image_preference')
            elif np.all(margins < -min_margin):
                audit.update(decision='reject', reason='external_image_prefers_baseline')
            else:
                audit['reason'] = 'tie_or_rendering_disagreement'
    except (ValueError, TypeError, RuntimeError) as exc:
        audit.update(decision='abstain', reason='verification_unavailable', error=str(exc))
    audit['seconds'] = perf_counter()-start
    return audit


def arbitrate_output(baseline, candidate, **kwargs):
    audit = assess_revision(baseline['text'], candidate['text'], **kwargs)
    output = copy.deepcopy(candidate)
    output['answer_arbitration'] = audit
    output['candidate_answer'] = {'text': candidate['text'], 'token_ids': candidate['token_ids'],
                                  'evidence_transport': candidate.get('evidence_transport', {})}
    if audit['decision'] not in {'accept', 'unchanged'}:
        for key in ('text', 'token_ids', 'finished', 'evidence_transport', 'visual_evidence'):
            if key in baseline:
                output[key] = copy.deepcopy(baseline[key])
        output['candidate_evidence'] = output.get('evidence', [])
        output.update(evidence=[], adopted_evidence_count=0, presented_evidence_count=0)
    output['seconds'] = baseline['seconds'] + candidate['seconds'] + audit['seconds']
    output['timing_includes_baseline_and_candidate'] = True
    output['trace'].append({'event': 'answer_arbitration', **audit})
    return output
