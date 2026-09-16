"""Answer-boundary admission; explicit requests are not an automatic medical parser.

Raw packets stay immutable. Audits are returned separately, never serialized as
answer evidence. A delivery view conveys only configured, returned native fields.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

from .capability_contracts import SEMANTIC_DIMENSIONS, authority_contract


@dataclass(frozen=True)
class EvidenceRequest:
    question: str
    entities: tuple[str, ...]
    dimensions: frozenset[str]
    modality: str
    task: str
    complete: bool = True

    def __post_init__(self):
        if not isinstance(self.entities, tuple) or not all(isinstance(e, str) and e.strip() for e in self.entities):
            raise ValueError('entities must be an explicit tuple of names')
        if not isinstance(self.dimensions, frozenset) or self.dimensions - SEMANTIC_DIMENSIONS:
            raise ValueError('invalid semantic dimensions')
        if type(self.complete) is not bool:
            raise ValueError('complete must be boolean')

    def to_json(self):
        return {**asdict(self), 'dimensions':sorted(self.dimensions)}


def _name(value):
    return ' '.join(str(value).lower().replace('_', ' ').split())


def _view(item, question, request, specs):
    if any(item.provenance.get(key) for key in (
            'target_mask_used', 'target_masks_used', 'target_annotations_used')):
        return None, 'target_annotation_provenance_forbidden'
    if request is None or not request.complete or not request.entities or not request.dimensions:
        return None, 'explicit_complete_request_required'
    if request.question != question:
        return None, 'request_question_mismatch'
    spec = specs.get(item.expert_id)
    if spec is None:
        return None, 'expert_not_configured'
    if (request.modality not in spec.get('modalities', ())
            or request.task not in spec.get('tasks', ())
            or item.capability not in spec.get('capabilities', ())
            or item.scope != spec.get('scope')):
        return None, 'configured_scope_mismatch'
    contract = authority_contract(item.expert_id, spec, item.capability)
    if contract is None:
        return None, 'native_contract_undeclared'
    if not request.dimensions <= contract.supports:
        return None, 'unsupported_dimensions'
    catalog = {_name(alias): name for name, aliases in contract.entity_aliases
               for alias in (name, *aliases)}
    if any(_name(entity) not in catalog for entity in request.entities):
        return None, 'native_entity_not_declared'
    wanted = {catalog[_name(entity)] for entity in request.entities}
    adapter = spec.get('adapter')
    if adapter == 'xrv_classification' and item.capability == 'classification':
        if (not request.dimensions <= {'finding_presence'}
                or contract.native_variable.attribute != 'presence'
                or contract.native_variable.output_semantics != 'independent_sigmoid_score'):
            return None, 'native_classifier_definition_mismatch'
        field, label = 'findings', 'finding'
        metadata = ('score_semantics', 'positive_threshold', 'unlisted_findings',
                    'image_transform', 'query_diagnosis')
        entry_fields = ('finding', 'score')  # CAM is not authorized presence evidence.
        if item.payload.get('score_semantics') != 'uncalibrated_independent_sigmoid':
            return None, 'unsupported_score_semantics'
    elif adapter == 'xrv_anatomy' and item.capability == 'segmentation':
        if (not request.dimensions <= {'anatomy_identity', 'location', 'laterality', 'relative_extent'}
                or contract.native_variable.attribute != 'spatial_extent'
                or contract.native_variable.output_semantics != 'per_structure_soft_mask'):
            return None, 'native_anatomy_definition_mismatch'
        field, label = 'structures', 'anatomical_structure'
        configured = set(spec.get('structures', ['Left Lung', 'Right Lung', 'Heart']))
        if not wanted <= configured:
            return None, 'structure_not_configured'
        metadata = ('image_transform', 'mask_threshold', 'mask_probabilities_calibrated',
                    'disease_or_lesion_segmentation')
        entry_fields = ('anatomical_structure', 'mask', 'soft_mask', 'mask_coordinate_system',
                        'foreground_fraction_of_crop', 'bbox_xyxy_normalized_original_image',
                        'empty_mask_means')
        if item.payload.get('disease_or_lesion_segmentation') is not False:
            return None, 'anatomy_semantics_not_established'
    else:
        return None, 'adapter_delivery_not_implemented'
    entries = item.payload.get(field, ())
    if not isinstance(entries, (list, tuple)) or any(not isinstance(e, dict) for e in entries):
        return None, 'invalid_native_entries'
    selected = [e for e in entries if e.get(label) in wanted]
    if {e.get(label) for e in selected} != wanted:
        return None, 'requested_entry_not_returned'
    if len(selected) != len(wanted):
        return None, 'duplicate_native_entries'
    if field == 'findings' and any('score' not in e for e in selected):
        return None, 'missing_native_score'
    if field == 'structures' and any(not (e.get('mask') or e.get('soft_mask')) for e in selected):
        return None, 'missing_native_mask'
    payload = {k: deepcopy(item.payload[k]) for k in metadata if k in item.payload}
    payload[field] = [{k:deepcopy(e[k]) for k in entry_fields if k in e} for e in selected]
    # Original summary/provenance may mention removed entities or old scope audits.
    return replace(item, payload=payload, summary='', confidence=None,
                   provenance={'adapter':adapter}), 'admitted'


def delivery_view(items, *, question, request=None, specs=None, mode='legacy'):
    """One shared filter for new, cached, inherited, text and spatial packets.

legacy is byte-compatible; audit computes would-be admission but delivers raw
packets; enforce delivers only the view. Missing explicit semantics fail closed
in enforce. Other expert adapters require their own reviewed native contract.
"""
    if mode not in {'legacy', 'audit', 'enforce'}:
        raise ValueError('unknown admission mode')
    raw = tuple(items)
    if mode == 'legacy':
        return raw, {'mode':mode, 'decisions':[], 'delivered_count':len(raw)}
    views, decisions = [], []
    for item in raw:
        view, reason = _view(item, question, request, specs or {})
        if view is not None:
            views.append(view)
        decisions.append({'expert_id':item.expert_id, 'evidence_id':item.evidence_id,
                          'would_admit':view is not None, 'reason':reason,
                          'delivered':mode == 'audit' or view is not None})
    delivered = raw if mode == 'audit' else tuple(views)
    return delivered, {'mode':mode, 'decisions':decisions, 'delivered_count':len(delivered)}
