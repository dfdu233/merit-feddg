"""Scope-preserving presentation of native observations, never clinical labels."""

from __future__ import annotations

from .evidence_need import evidence_memory
from .open_study import fingerprint


def active_packet(items, question, config, *, prefix_tokens, lifetime, control="real"):
    """Deduplicate exact same-expert observations and expire by token distance.

    This does not claim semantic contradiction detection or multi-view DG.
    Different expert capabilities are not forced to agree. Source-family
    correlation is not converted to an unjustified independent evidence count.
    """
    if control not in {"real", "format_only"}:
        raise ValueError("unknown packet control")
    active, audit, seen = [], [], set()
    for item in items:
        acquired = item.provenance.get("merit_acquired_token", 0)
        if type(acquired) is not int or not 0 <= acquired <= prefix_tokens:
            raise ValueError("invalid evidence acquisition position")
        digest = fingerprint([item.expert_id, item.capability, item.scope, item.payload])
        reason = "active"
        if prefix_tokens - acquired >= lifetime:
            reason = "expired"
        elif digest in seen:
            reason = "duplicate"
        else:
            active.append(item)
            seen.add(digest)
        audit.append({"evidence_id": item.evidence_id, "expert": item.expert_id,
                      "scope": item.scope, "reason": reason, "acquired_token": acquired})
    packet = evidence_memory(active, question, config)
    if control == "format_only":
        # Identical outer schema and tool IDs, NOT an exact token-length match.
        # No observations, score values, diagnoses or native summaries survive.
        packet = [{"expert_id": v["expert_id"], "evidence_id": v["evidence_id"],
                   "capability": v["capability"], "scope": v["scope"],
                   "payload": {"status": "unknown", "observation": "withheld control"}}
                  for v in packet]
    return packet, audit
