# CS197 Research Skill

Use this skill for every research iteration in this repository. The goal is not to add mechanisms quickly; it is to reduce uncertainty about one publishable scientific claim at a time.

## 1. Start from the evidence snapshot

Before proposing anything, freeze the current facts: exact commit, exact experiment identity, what was actually measured, what was not measured, what failed, and what remains confounded. Separate engineering success, diagnostic behavior, and scientific evidence. Never promote a canary, partial prefix, parser score, or synthetic test into a general empirical claim.

## 2. Build an affinity map before a Bit Flip

Group the nearest literature by shared problem setup, mechanism, or assumption. A cluster should usually contain several genuinely related works; do not force the whole field to share one assumption. For every cluster record:

- what problem the papers solve;
- what variable or proxy they manipulate;
- what assumptions are required for that manipulation to be useful;
- the nearest-neighbor paper and why this project is not already contained in it;
- collision risks from adjacent fields and recent work.

Only after this synthesis may a Bit Flip be proposed.

## 3. Bit Flip test

A valid Bit Flip must pass all of these checks:

1. **Shared assumption**: the bit is actually shared by the relevant literature cluster, not invented after seeing our failure.
2. **Operational flip**: both the original bit and the flipped bit can be stated as measurable variables or mechanisms.
3. **Importance**: if the flip is true, it changes how the research problem should be solved, not merely one implementation detail.
4. **Nearest-neighbor separation**: the flip is not already implemented by the closest prior work under another name.
5. **Falsifiability**: there is a realistic experiment whose negative result would make us abandon or revise the idea.
6. **No engineering disguise**: restoring a missing routing check, changing a prompt format, fixing a parser, or replacing a slow implementation is not itself a research Bit Flip.

If any item fails, treat the idea as an engineering hypothesis or baseline, not the paper's novelty.

## 4. Vectoring

At the start of each iteration, state exactly one **Vector**:

> What is the riskiest unanswered scientific question in the project right now?

The Vector must be a question whose answer could change the direction of the project. Then divide work into:

- **Core**: only the components needed to answer this Vector.
- **Periphery**: useful features that do not reduce uncertainty on the Vector this iteration.

Freeze the periphery. Do not add RAG, planners, gates, verifiers, extra experts, visualizations, or new losses unless they are necessary to answer the current Vector.

## 5. Design the minimum discriminating experiment

The experiment should distinguish competing explanations, not merely seek a higher aggregate score. Prefer matched interventions and local contrast sets.

For every experiment specify before execution:

- H1 and H0;
- exact baselines, including the strongest simple baseline;
- what is held constant;
- what one variable changes;
- negative controls and counterfactual controls;
- oracle or upper bound only when it does not leak target labels into inference;
- primary mechanism metric and final task metric;
- stop conditions;
- what result would force re-vectoring.

Never select examples, thresholds, prompts, or hyperparameters from test outcomes. Development diagnostics must stay on TRAIN/development data with frozen evaluation rules.

## 6. Prefer behavioral properties before aggregate gains

Before large-scale runs, test the mechanism's defining property directly. Examples:

- invariance: changing information outside an expert's authority should not change the controlled variable;
- directionality: strengthening positive source evidence should not systematically move a target claim in the negative direction;
- locality: a capability-bounded intervention should not alter distinctions the expert cannot make;
- contrast support: paired candidates that differ only in the controlled claim should behave differently in the intended direction.

Use aggregate accuracy only after the mechanism is shown to exist.

## 7. Novelty audit before implementation and before writing claims

Perform two novelty checks:

### Pre-implementation
Search nearest neighbors and adjacent fields for the exact mechanism, not just keywords. Inspect papers and official code when available. Ask whether our proposed method is merely:

- a new alpha schedule;
- CAD/CFG under a new name;
- a prompt serialization change;
- a standard constrained-decoding method;
- a known counterfactual-generation method;
- a standard posterior projection;
- an existing agent/tool-use pattern.

If yes, use it as a baseline or substrate and move novelty to the genuinely new scientific question.

### Post-result
After the experiment, reassess whether the evidence still supports the novelty story. A new failure mode can become a contribution only if it is general, reproducible, and not explained by an implementation bug or a known prior result.

## 8. Velocity

Velocity is not lines of code, GPU utilization, branch count, or number of experiments. It is the amount of uncertainty removed from the current Vector.

At the end of each iteration report:

- **Vector**: the risky question we tried to answer;
- **Plan**: the smallest experiment intended to reduce that uncertainty;
- **Result**: what happened, including negative results;
- **Belief update**: which hypotheses became more or less plausible;
- **Residual uncertainty**: what remains unresolved;
- **Next Vector**: the new riskiest unanswered question;
- **Periphery kept frozen**: what we deliberately did not expand.

Re-vector immediately when a result makes the previous Vector no longer the main risk. Do not continue an experiment only because code already exists.

## 9. Repository discipline

- Preserve historical branches and results.
- New scientific hypotheses get independent branches/identities.
- Do not overwrite frozen evaluations.
- Report exact commit and experiment identity.
- Keep references out of inference and candidate construction unless the protocol explicitly allows them.
- Distinguish training-free, calibration-free, and target-free; they are different claims.
- Cost and coverage are first-class results.
- A method that falls back to the incumbent must report fallback coverage separately from successful intervention.

## 10. Required response template for future research turns

When planning or interpreting a new research iteration, use this order:

1. **Evidence update** — what the latest result actually establishes.
2. **Affinity-map location** — which literature cluster this result speaks to.
3. **Bit Flip status** — supported, weakened, or unchanged.
4. **Current Vector** — one sentence.
5. **Core / Periphery** — what we will and will not touch.
6. **Minimum experiment** — matched design, controls, metrics, stop rule.
7. **Novelty collision check** — closest prior work and why this is not merely a rename.
8. **Velocity criterion** — what new knowledge must exist after the experiment.
9. **Re-vector rules** — explicit branches for positive, negative, and ambiguous outcomes.

## Current project interpretation after the authority TRAIN pilot

The completed 12-case authority pilot does **not** demonstrate benefit from minimum-KL authority projection. Projection was feasible in only 1/12 cases because the historical free-text candidate pool usually lacked both sides of the target semantic variable. Therefore the current bottleneck is not the projection coefficient; it is whether we can obtain matched free-text support for both states of the specialist-authorized claim without changing out-of-scope content or using target labels.

The next Vector should test that support problem directly before expanding authority projection to segmentation, retrieval, more experts, or full datasets.
