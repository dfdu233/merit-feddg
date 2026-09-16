# Native capability contract Vector

## Vector

Can a frozen specialist's native task be represented as a machine-readable semantic authority boundary that prevents out-of-scope global influence before reliability is considered?

This is deliberately narrower than a new gate. Reliability, uncertainty and calibration are frozen for this iteration.

## First contracts

Two existing XRV specialists are declared:

- `cxr_findings`: native variable = finding presence from independent sigmoid scores; authorized only for `finding_presence`.
- `cxr_anatomy`: native variable = thoracic anatomical spatial extent from per-structure masks; authorized for anatomy identity, location, laterality and relative extent, but not disease presence or physical measurement.

The contract is derived from the model task/output semantics, not from target labels or pilot correctness.

## Enforcement

`assess_authority()` returns `exact`, `partial`, `denied`, `unknown`, or `undeclared`.

- Exact: the declared specialist covers every semantic dimension detected in the request. Existing global tool/text transport may use it.
- Partial: the specialist covers only a subset. It may be useful to a future local intervention, but is blocked from legacy global text transport.
- Denied/unknown: no intervention is admitted.
- Undeclared: historical experts preserve old routing behavior so this branch does not silently redefine every tool at once.

Examples:

- XRV finding classifier + `Is there a pneumothorax?` -> exact.
- XRV finding classifier + `Is there a left pneumothorax?` -> partial: presence authorized, laterality not authorized; global transport blocked.
- XRV finding classifier + `Where is the pneumothorax?` -> denied.
- XRV anatomy mask + `Where is the left lung?` -> exact.
- XRV anatomy mask + `Is there pneumonia?` -> denied.
- XRV anatomy mask + `How large is the heart in centimeters?` -> denied.

## Scientific boundary

This branch establishes only an explicit capability/applicability layer. It does not establish that an admitted expert is correct, calibrated, or useful. A low-confidence or wrong expert can still be wrong inside its authority boundary; that is a later Vector only if this boundary mechanism survives controlled tests.

## Stop rule

Do not expand to BiomedParse, retrieval, generated-text specialists, reliability weighting, segmentation-derived diagnosis, or full benchmark runs until the controlled boundary tests show that declared contracts reliably block out-of-scope influence without destroying in-scope coverage.
