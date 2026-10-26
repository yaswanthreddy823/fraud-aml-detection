"""Business-rule layer for fraud decisions.

In production, the ML score is only one input: a layered validation step
combines it with business rules and compliance checks before a transaction
is approved, sent for review, or blocked. These rules mirror that layer in
simplified form.

A transaction is a dict with the same fields as ``generate_data`` rows.
"""

from __future__ import annotations

HIGH_AMOUNT = 1000.0
ODD_HOUR_END = 5
VELOCITY_LIMIT = 6
FAR_DISTANCE_KM = 500.0


def evaluate_rules(txn: dict) -> list[str]:
    """Return the names of all rules triggered by this transaction."""
    flags: list[str] = []
    amount = float(txn.get("amount", 0))
    hour = int(txn.get("hour", 12))
    if amount > HIGH_AMOUNT:
        flags.append("high_amount")
    if hour < ODD_HOUR_END:
        flags.append("odd_hour")
    if txn.get("is_new_device") and amount > 300:
        flags.append("new_device_high_amount")
    if int(txn.get("txns_last_24h", 0)) >= VELOCITY_LIMIT:
        flags.append("velocity")
    if float(txn.get("distance_km", 0)) > FAR_DISTANCE_KM:
        flags.append("far_from_home")
    if txn.get("is_international") and txn.get("is_new_device"):
        flags.append("intl_new_device")
    return flags


def combine_decision(score: float, flags: list[str],
                     review_threshold: float = 0.5,
                     block_threshold: float = 0.8) -> str:
    """Combine ML score and rule flags into approve / review / block.

    Rules escalate but never clear: two or more flags force at least a
    review, and a high score blocks outright.
    """
    if score >= block_threshold:
        return "block"
    if score >= review_threshold or len(flags) >= 2:
        return "review"
    return "approve"
