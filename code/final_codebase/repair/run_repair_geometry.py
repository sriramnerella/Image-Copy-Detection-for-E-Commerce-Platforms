from __future__ import annotations

import argparse
from pathlib import Path

from duplicate_bank.geometry_repair import repair_store_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair strict-panel geometry on an existing store.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store-json", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--model", default="convnext", choices=["convnext", "resnet", "vit"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    out = repair_store_geometry(
        checkpoint=Path(args.checkpoint),
        store_json=Path(args.store_json),
        output_checkpoint=Path(args.output_checkpoint),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"saved repaired checkpoint: {out}")


if __name__ == "__main__":
    main()
