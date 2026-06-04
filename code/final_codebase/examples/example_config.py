from pathlib import Path

from duplicate_bank.config import StoreConfig, TrainConfig


RUN_ROOT = Path("runs/example")

STORE = StoreConfig(
    store_json=RUN_ROOT / "embedding_store.json",
    active_pool_json=RUN_ROOT / "active_pool.json",
    source_pool_json=RUN_ROOT / "source_pool_dataset.json",
    known_active=1000,
    unknown_active=250,
)

TRAIN = TrainConfig(
    manifest_json=RUN_ROOT / "training_pairs_dataset.json",
    output_dir=RUN_ROOT / "checkpoints",
    store=STORE,
    epochs=20,
    batch_size=16,
    learning_rate=3e-6,
    max_batches_per_epoch=1200,
)
