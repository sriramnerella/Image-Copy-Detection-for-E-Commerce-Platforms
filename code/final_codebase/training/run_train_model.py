from __future__ import annotations

import argparse
from pathlib import Path

from duplicate_bank.config import StoreConfig, TrainConfig
from duplicate_bank.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train duplicate-bank ConvNeXt model.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--source-pool-json", required=True)
    parser.add_argument("--store-json", required=True)
    parser.add_argument("--active-pool-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--known-active", type=int, default=1000)
    parser.add_argument("--unknown-active", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    store = StoreConfig(
        store_json=Path(args.store_json),
        active_pool_json=Path(args.active_pool_json),
        source_pool_json=Path(args.source_pool_json),
        known_active=args.known_active,
        unknown_active=args.unknown_active,
    )
    config = TrainConfig(
        manifest_json=Path(args.manifest_json),
        output_dir=Path(args.output_dir),
        store=store,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_batches_per_epoch=args.max_batches,
    )
    train(config)


if __name__ == "__main__":
    main()
