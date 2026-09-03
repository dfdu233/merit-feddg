from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .types import EvidenceRecord


@dataclass(frozen=True)
class ClientStatistic:
    domain: str
    expert: str
    correct: int
    total: int
    squared_error: float


class FederatedReliabilityCalibrator:
    """Privacy-friendly source-domain reliability calibration.

    Each institution emits only counts and aggregate calibration error. The
    server never sees examples, embeddings, logits, or target-domain labels.
    Reliability gates an intervention; it is not an ensemble weight.
    """

    def __init__(self, prior: float = 4.0, lcb_z: float = 1.0) -> None:
        self.prior = float(prior)
        self.lcb_z = float(lcb_z)
        self.reliability: dict[str, float] = {}
        self.client_statistics: list[ClientStatistic] = []

    @staticmethod
    def client_summary(domain: str, records: Iterable[EvidenceRecord]) -> list[ClientStatistic]:
        buckets: dict[str, list[tuple[int, float]]] = {}
        for record in records:
            expert = record.modality
            scores = record.expert_scores[expert]
            predicted = int(np.argmax(scores))
            margin = float(np.max(scores) - np.min(scores))
            confidence = 1.0 / (1.0 + np.exp(-margin))
            correct = int(predicted == record.label)
            buckets.setdefault(expert, []).append((correct, confidence))
        return [
            ClientStatistic(
                domain=domain,
                expert=expert,
                correct=sum(value[0] for value in values),
                total=len(values),
                squared_error=sum((value[1] - value[0]) ** 2 for value in values),
            )
            for expert, values in sorted(buckets.items())
        ]

    def fit(self, records: Iterable[EvidenceRecord], source_domains: set[str]) -> None:
        grouped: dict[str, list[EvidenceRecord]] = {}
        for record in records:
            if record.domain in source_domains:
                grouped.setdefault(record.domain, []).append(record)
        if len(grouped) < 2:
            raise ValueError("federated calibration requires at least two source domains")

        summaries: list[ClientStatistic] = []
        for domain, items in sorted(grouped.items()):
            summaries.extend(self.client_summary(domain, items))
        self.client_statistics = summaries

        experts = sorted({item.expert for item in summaries})
        for expert in experts:
            rows = [item for item in summaries if item.expert == expert]
            correct = sum(row.correct for row in rows)
            total = sum(row.total for row in rows)
            alpha = self.prior + correct
            beta = self.prior + total - correct
            mean = alpha / (alpha + beta)
            variance = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
            lower = mean - self.lcb_z * np.sqrt(variance)
            # Penalize experts whose confidence is poorly calibrated on source clients.
            brier = sum(row.squared_error for row in rows) / max(total, 1)
            self.reliability[expert] = float(np.clip(lower * (1.0 - brier), 0.05, 1.0))

    def score(self, expert: str) -> float:
        return self.reliability.get(expert, 0.5)

    def report(self) -> dict:
        return {
            "privacy_contract": "aggregate counts and squared calibration error only",
            "target_labels_used": False,
            "reliability": self.reliability,
            "clients": [item.__dict__ for item in self.client_statistics],
        }


def continuous_frequency_mix(
    source: Image.Image,
    peer: Image.Image,
    alpha: float = 0.05,
) -> Image.Image:
    """FedDG-style amplitude interpolation retained as a baseline only."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    original_mode = source.mode
    left = np.asarray(source.convert("RGB"), dtype=np.float32)
    right_image = peer.convert("RGB").resize(source.size, Image.Resampling.BILINEAR)
    right = np.asarray(right_image, dtype=np.float32)
    left_fft = np.fft.fft2(left, axes=(0, 1))
    right_fft = np.fft.fft2(right, axes=(0, 1))
    mixed_amplitude = (1.0 - alpha) * np.abs(left_fft) + alpha * np.abs(right_fft)
    mixed = np.fft.ifft2(mixed_amplitude * np.exp(1j * np.angle(left_fft)), axes=(0, 1)).real
    clipped = np.clip(mixed, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped).convert(original_mode)
