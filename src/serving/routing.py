"""Random audit (TASK-023, ADR-007, ADR-030). Standard library only.

A prediction that would be auto-accepted (no ``low_confidence`` or ``ood`` reason) is also sent
to review if a salted hash of its request id falls below ``audit_rate``:

    u = first 8 bytes of SHA-256(request_id + salt) / 2**64,   audited if u < audit_rate

The request id is a random server-made UUID, so the decision is fixed for a given request and
independent of the image, the confidence and the energy; with a secret salt it cannot be
predicted or steered by a client.
"""

from __future__ import annotations

import hashlib

REASON_AUDIT = "audit"
_SCALE = float(2**64)


def audit_draw(request_id: str, salt: str) -> float:
    """A number in [0, 1), uniform over request ids, fixed for a given id and salt."""
    digest = hashlib.sha256((request_id + salt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / _SCALE


def audit_selected(request_id: str, salt: str, rate: float) -> bool:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"audit rate must be in [0, 1], got {rate}")
    return rate > 0.0 and audit_draw(request_id, salt) < rate


def with_audit(reasons: list[str], request_id: str, salt: str, rate: float) -> list[str]:
    """The routing reasons after the audit: only auto-accepted predictions can be audited."""
    if reasons:
        return list(reasons)
    return [REASON_AUDIT] if audit_selected(request_id, salt, rate) else []
