"""Input-specific expert admission from source-only, expert-native neighborhoods.

No target fitting, diagnosis from distance, or arbitrary-shift guarantee. The
LODO penalty is an empirical error correction, not a confidence bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .capability_features import action_key


def unit(vector):
    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("expected a finite one-dimensional native embedding")
    norm = np.linalg.norm(value)
    if norm < 1e-10:
        raise ValueError("zero native embedding")
    return value / norm


@dataclass(frozen=True)
class ApplicabilityConfig:
    neighbors: int = 4
    min_per_domain: int = 2
    min_domains: int = 2
    max_risk: float = .1
    max_distance: float = .4
    residual_quantile: float = .9
    allow_proxy: bool = False

    def __post_init__(self):
        if type(self.allow_proxy) is not bool:
            raise ValueError("allow_proxy must be an explicit boolean")
        if any(type(x) is not int or x < 1 for x in (
            self.neighbors, self.min_per_domain, self.min_domains
        )) or self.neighbors < self.min_per_domain:
            raise ValueError("positive support budgets; neighbors >= min_per_domain")
        if not (np.isfinite([self.max_risk, self.max_distance, self.residual_quantile]).all()
                and 0 <= self.max_risk <= 1 and 0 < self.max_distance <= 2
                and 0 <= self.residual_quantile <= 1):
            raise ValueError("invalid risk/distance/quantile")


def local_risk(records, feature, config, *, exclude_group=None, exclude_pixel=None,
               exclude_domain=None):
    query = unit(feature)
    by_domain = {}
    for record in records:
        if (record["group_id"] == exclude_group or record["image_sha256"] == exclude_pixel
                or record["domain"] == exclude_domain):
            continue
        vector = unit(record["feature"])
        if vector.shape != query.shape:
            raise ValueError("native embedding dimension changed")
        distance = float(np.clip(1-query@vector, 0, 2))
        if distance <= config.max_distance:
            by_domain.setdefault(record["domain"], []).append((distance, record))
    domains = {}
    for domain, values in sorted(by_domain.items()):
        nearest = sorted(values, key=lambda item: (item[0], item[1]["id"]))[:config.neighbors]
        if len(nearest) < config.min_per_domain:
            continue
        distances = np.array([v[0] for v in nearest])
        weights = np.exp(-distances/config.max_distance)
        weights /= weights.sum()
        domains[domain] = {"n": len(nearest), "risk": float(weights @ np.array(
            [v[1]["loss"] for v in nearest])), "max_distance": float(distances.max())}
    return {"supported": len(domains) >= config.min_domains, "domains": domains,
            "mean": float(np.mean([d["risk"] for d in domains.values()])) if domains else 1.,
            "worst": max((d["risk"] for d in domains.values()), default=1.)}


def build_memory(records, config, provenance, metric):
    """One native embedding and one observed loss per independent image/scope."""
    if not records:
        raise ValueError("no source observations; do not manufacture expert errors")
    scopes, identities = {}, set()
    for row in records:
        if row["role"] != "source":
            raise ValueError("target observations cannot build applicability memory")
        if (not config.allow_proxy and row["domain_kind"] != "independent"):
            raise ValueError("real source domains required; explicit allow_proxy is diagnostic only")
        loss = row["loss"]
        if isinstance(loss, bool) or not np.isfinite(loss) or not 0 <= loss <= 1:
            raise ValueError("loss must be a measured continuous value in [0,1]")
        for key in ("id", "scope_key", "group_id", "image_sha256", "domain"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"missing {key}")
        for field in ("group_id", "image_sha256"):
            identity = (row["scope_key"], field, row[field])
            if identity in identities:
                raise ValueError("duplicate independent group/pixel in expert scope")
            identities.add(identity)
        item = {**row, "feature": unit(row["feature"]).tolist()}
        scopes.setdefault(row["scope_key"], []).append(item)
    penalties, audits = {}, {}
    for key, values in scopes.items():
        if len({len(v["feature"]) for v in values}) != 1:
            raise ValueError("native embedding dimension changed within expert scope")
        residuals, held_out = [], []
        for row in values:
            # Entire domain removed, not only the query or its own neighbor.
            fold = local_risk(values, row["feature"], config, exclude_domain=row["domain"],
                              exclude_group=row["group_id"], exclude_pixel=row["image_sha256"])
            if fold["supported"]:
                residual = max(0., row["loss"]-fold["worst"])
                residuals.append(residual)
                held_out.append({"id": row["id"], "domain": row["domain"],
                                 "predicted_risk": fold["worst"], "observed_loss": row["loss"]})
        # Require every domain represented in residual calibration, not just easy folds.
        complete = {r["domain"] for r in held_out} == {r["domain"] for r in values}
        penalties[key] = float(np.quantile(residuals, config.residual_quantile)) if (
            residuals and complete) else None
        audits[key] = {"supported_lodo_records": len(residuals), "all_domains_supported": complete,
                       "held_out": held_out}
    return {"schema": "expert-applicability-v1", "config": asdict(config), "scopes": scopes,
            "penalties": penalties, "lodo": audits, "provenance": provenance, "metric": metric,
            "real_source_domains": all(r["domain_kind"] == "independent" for r in records)}


class ApplicabilityGate:
    """Runtime plugin: filter tool descriptors before the existing controller.

    Native encoding can load/run an expert even when admission is rejected.
    This is evidence admission, NOT a guarantee of zero model-load cost.
    """
    def __init__(self, memory, pool, *, mode="robust", exclude_domain=False):
        if memory.get("schema") != "expert-applicability-v1":
            raise ValueError("unsupported applicability schema")
        if mode not in {"global", "distance", "local", "robust"}:
            raise ValueError("unknown applicability ablation")
        self.memory, self.pool, self.mode = memory, pool, mode
        self.config = ApplicabilityConfig(**memory["config"])
        self.exclude_domain, self.cache = exclude_domain, {}

    def decide(self, row, descriptor):
        key = action_key(row, descriptor)
        identity = (key, row["id"], row["image_sha256"], row["group_id"], row["domain"])
        if identity in self.cache:
            return self.cache[identity]
        records = self.memory["scopes"].get(key, [])
        audit = {"scope_key": key, "mode": self.mode, "allowed": False,
                 "reason": "NONE:missing_applicability_scope", "metric": self.memory["metric"]}
        if records:
            retained = [r for r in records if r["group_id"] != row["group_id"]
                        and r["image_sha256"] != row["image_sha256"]
                        and (not self.exclude_domain or r["domain"] != row["domain"])]
            if self.mode == "global":
                groups = {}
                for record in retained:
                    groups.setdefault(record["domain"], []).append(record["loss"])
                supported = len(groups) >= self.config.min_domains and all(
                    len(v) >= self.config.min_per_domain for v in groups.values())
                risk = float(np.mean([np.mean(v) for v in groups.values()])) if groups else 1.
                audit.update(supported=supported, risk=risk)
            else:
                feature = self.pool.domain_embedding(descriptor["expert"], row["image"])
                result = local_risk(retained, feature, self.config)
                risk = result["mean"]
                supported = result["supported"]
                if self.mode == "distance":
                    risk = 0.  # explicit distance-only ablation, not estimated correctness
                elif self.mode == "robust":
                    # Refit residuals without evaluation domain for source cross-fitting.
                    penalty = self.memory["penalties"].get(key)
                    if self.exclude_domain and retained:
                        fold_memory = build_memory(retained, self.config, {}, self.memory["metric"])
                        penalty = fold_memory["penalties"].get(key)
                    supported = supported and penalty is not None
                    risk = min(1., result["worst"]+(penalty if penalty is not None else 1.))
                audit.update(supported=supported, risk=risk, domains=result["domains"])
            allowed = bool(supported and risk <= self.config.max_risk)
            audit.update(allowed=allowed, reason=("admitted" if allowed else
                         "NONE:input_risk" if supported else "NONE:insufficient_local_support"))
        self.cache[identity] = audit
        return audit
