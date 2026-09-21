"""Deterministic answer-blind matched controls for MERIT-Tx evidence specificity."""
import hashlib


_LABEL_KEYS = frozenset(
    {"answer", "answers", "label", "labels", "reference", "references", "ground_truth"}
)


def select_matched_knockoffs(
    source_rows,
    current_row,
    *,
    expert_id,
    count=4,
):
    """Select stable wrong-patient controls without consulting labels or scores."""
    if type(count) is not int or count < 1:
        raise ValueError("knockoff count must be positive")
    required = ("id", "image", "modality", "task", "group_id")
    if any(not str(current_row.get(key, "")).strip() for key in required):
        raise ValueError("current row needs id/image/modality/task/group_id")
    if not expert_id:
        raise ValueError("expert_id cannot be empty")

    candidates = []
    for row in source_rows:
        if any(not str(row.get(key, "")).strip() for key in required):
            raise ValueError("source knockoff rows need id/image/modality/task/group_id")
        # Label fields may exist in a source-training record, but selection is
        # deliberately projected onto answer-blind metadata before ranking.
        public = {
            key: row[key]
            for key in required
        }
        if (
            public["id"] == current_row["id"]
            or public["group_id"] == current_row["group_id"]
            or public["modality"] != current_row["modality"]
            or public["task"] != current_row["task"]
        ):
            continue
        digest = hashlib.sha256(
            (
                str(expert_id)
                + "|"
                + str(current_row["id"])
                + "|"
                + str(public["id"])
            ).encode()
        ).hexdigest()
        candidates.append((digest, public))
    candidates.sort(key=lambda value: (value[0], str(value[1]["id"])))
    if len(candidates) < count:
        raise ValueError(
            f"insufficient matched source controls: need {count}, found {len(candidates)}"
        )
    return [
        {
            **row,
            "role": "knockoff_control",
            "selection": "same-modality-task different-group stable-hash",
            "labels_consulted": False,
            "discarded_label_keys": sorted(_LABEL_KEYS.intersection(source_rows[0]))
            if source_rows
            else [],
        }
        for _digest, row in candidates[:count]
    ]
