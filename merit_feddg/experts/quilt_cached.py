"""Frozen offline Quilt-LLaVA predictions exposed through the existing factory API.

Prepare/infer with the isolated Quilt worker first. No silent download, synthetic
fallback, prompt rewriting, or loading the second llava package in the actor.
"""
from pathlib import Path

from ..pathology_pilot import (
    file_sha, prediction_jobs, quilt_item, read_json, validate_predictions,
)


class CachedQuiltExpert:
    def __init__(self, *, run_dir, cache_sha256, expert_id="quilt_pathology",
                 scope="histology_question_observation"):
        root = Path(run_dir).expanduser().resolve()
        cache = root / "quilt_predictions.json"
        if file_sha(cache) != cache_sha256:
            raise ValueError("explicit prediction-cache SHA256 mismatch")
        self.frozen = read_json(root / "frozen.json")
        self.records = validate_predictions(self.frozen, read_json(cache))
        self.rows = {r["id"]: r for r in self.frozen["rows"]}
        self.jobs = {j["target_id"]: j for j in prediction_jobs(self.frozen) if j["variant"] == "matched"}
        self.expert_id, self.scope = expert_id, scope

    def infer(self, request):
        from ..capabilities import CapabilityResult

        if (request.capability != "generation" or request.modality != "pathology"
                or request.task != "open_vqa" or request.region is not None
                or request.scope != self.scope):
            return CapabilityResult(self.expert_id, request.capability, (), reason="wrong_capability_or_scope")
        row = self.rows.get(str(request.sample_id))
        if (row is None or request.question != row["question"]
                or (request.query and request.query != request.question)
                or file_sha(request.image) != row["image_file_sha256"]):
            raise ValueError("Quilt cache miss: run the real worker for this exact image/question")
        record = self.records[self.jobs[row["id"]]["key"]]
        item = quilt_item(record, expert_id=self.expert_id, scope=self.scope)
        return CapabilityResult(self.expert_id, request.capability, (item,))


def build_cached_quilt(model_id, **kwargs):
    # CapabilityPool's existing factory always supplies model_id.
    if model_id != "wisdomik/Quilt-Llava-v1.5-7b" and not Path(model_id).is_dir():
        raise ValueError("factory model_id is not Quilt-LLaVA or a local model directory")
    return CachedQuiltExpert(**kwargs)
