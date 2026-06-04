from __future__ import annotations

import argparse
from pathlib import Path

from duplicate_bank.manifest import build_pairs, collect_images, split_known_unknown, write_dataset_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source pool and training pair manifest.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="dataset")
    parser.add_argument("--known-count", type=int, required=True)
    parser.add_argument("--total-pairs", type=int, required=True)
    parser.add_argument("--hard-pos", type=int, required=True)
    parser.add_argument("--soft-pos", type=int, required=True)
    parser.add_argument("--hard-neg", type=int, required=True)
    parser.add_argument("--soft-neg", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    rows = split_known_unknown(collect_images(args.image_root), args.known_count, args.seed)
    pairs = build_pairs(
        rows,
        total_pairs=args.total_pairs,
        hard_pos=args.hard_pos,
        soft_pos=args.soft_pos,
        hard_neg=args.hard_neg,
        soft_neg=args.soft_neg,
        seed=args.seed,
    )
    write_dataset_files(rows, pairs, Path(args.output_dir), args.prefix)
    print(f"wrote {len(rows)} source rows and {len(pairs)} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
