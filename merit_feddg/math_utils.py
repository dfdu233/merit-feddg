from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(values, dtype=float) / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(np.clip(scaled, -60.0, 60.0))
    return exp / np.sum(exp)


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-value))


def cosine(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> float:
    left = np.asarray(left, dtype=float) - np.mean(left)
    right = np.asarray(right, dtype=float) - np.mean(right)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < eps:
        return 0.0
    return float(np.dot(left, right) / denom)


def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / max(float(probs.sum()), eps)
    raw = -float(np.sum(probs * np.log(np.clip(probs, eps, 1.0))))
    return raw / max(float(np.log(len(probs))), eps)


def clip_norm(vector: np.ndarray, maximum: float, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm < eps:
        return vector
    return vector * (maximum / norm)
