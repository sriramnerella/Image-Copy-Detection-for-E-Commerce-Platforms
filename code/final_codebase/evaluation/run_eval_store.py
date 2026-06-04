from __future__ import annotations

import argparse
from pathlib import Path

from duplicate_bank.evaluation import evaluate_store_26_attacks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 26-attack known-bank eval from store.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default="convnext", choices=["convnext", "resnet", "vit"])
    parser.add_argument("--query-known", type=int, default=100)
    parser.add_argument("--query-unknown", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    result = evaluate_store_26_attacks(
        checkpoint=Path(args.checkpoint),
        store_json=Path(args.store_json),
        output_json=Path(args.output_json),
        model_name=args.model,
        query_known=args.query_known,
        query_unknown=args.query_unknown,
        batch_size=args.batch_size,
    )
    for attack, row in result["attacks"].items():
        print(f"{attack:24s} known={row['known']:6.2f} unknown={row['unknown']:6.2f} total={row['total']:6.2f} thr={row['threshold']}")


if __name__ == "__main__":
    main()
