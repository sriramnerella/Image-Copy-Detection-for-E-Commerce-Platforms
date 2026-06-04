"""Compatibility aliases for older imports.

New code should import from ``duplicate_bank.evaluation``.
"""

from .evaluation import choose_best_threshold, embed_query_rows, evaluate_store_26_attacks

best_threshold = choose_best_threshold
run_store_eval = evaluate_store_26_attacks

__all__ = [
    "best_threshold",
    "choose_best_threshold",
    "embed_query_rows",
    "evaluate_store_26_attacks",
    "run_store_eval",
]
