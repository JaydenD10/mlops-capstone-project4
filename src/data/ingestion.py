"""
Data Ingestion Stage for MLOps Pipeline.
Loads official datasets, verifies cryptographic checksums, splits train/val/test deterministically,
and isolates incoming operational batches for drift monitoring.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger

logger = setup_logger("ingestion")


def calculate_file_hash(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file for lineage tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def ingest_data(config: PipelineConfig = None) -> Dict[str, Any]:
    """
    Ingests official training and batches datasets.
    Splits training data into train/val/test and saves splits.
    Saves individual incoming batches.
    Returns ingestion metadata.
    """
    if config is None:
        config = load_config()

    logger.info("Starting Data Ingestion stage...")

    # Verify raw files exist
    raw_train_path = Path(config.paths.raw_train_data)
    raw_batches_path = Path(config.paths.raw_incoming_batches)

    # Fallback to current directory if not found in data/raw
    if not raw_train_path.exists():
        fallback_train = Path("capstone_house_prices_train.csv")
        if fallback_train.exists():
            raw_train_path = fallback_train
        else:
            raise FileNotFoundError(f"Training data not found at {raw_train_path} or root")

    if not raw_batches_path.exists():
        fallback_batches = Path("capstone_incoming_batches.csv")
        if fallback_batches.exists():
            raw_batches_path = fallback_batches
        else:
            raise FileNotFoundError(f"Batches data not found at {raw_batches_path} or root")

    train_hash = calculate_file_hash(raw_train_path)
    batches_hash = calculate_file_hash(raw_batches_path)

    logger.info(f"Loaded raw training dataset: {raw_train_path} (SHA-256: {train_hash[:12]}...)")
    logger.info(f"Loaded raw incoming batches: {raw_batches_path} (SHA-256: {batches_hash[:12]}...)")

    df_train_raw = pd.read_csv(raw_train_path)
    df_batches_raw = pd.read_csv(raw_batches_path)

    logger.info(f"Raw train shape: {df_train_raw.shape}, Raw batches shape: {df_batches_raw.shape}")

    # Prepare directories
    processed_dir = Path(config.paths.processed_dir)
    batches_dir = Path(config.paths.batches_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified split for Train / Val / Test
    random_seed = config.project.random_seed
    val_ratio = config.split.val_ratio
    test_ratio = config.split.test_ratio
    temp_ratio = val_ratio + test_ratio  # 0.30

    stratify_col = config.split.stratify_by
    stratify_data = df_train_raw[stratify_col] if stratify_col in df_train_raw.columns else None

    df_train, df_temp = train_test_split(
        df_train_raw,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_data
    )

    # Split the temp set into Val and Test evenly (0.15 each out of 0.30)
    temp_stratify = df_temp[stratify_col] if stratify_col in df_temp.columns else None
    test_size_fraction = test_ratio / temp_ratio  # 0.50

    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_size_fraction,
        random_state=random_seed,
        stratify=temp_stratify
    )

    # Save processed splits
    train_dest = Path(config.paths.train_data)
    val_dest = Path(config.paths.val_data)
    test_dest = Path(config.paths.test_data)

    df_train.to_csv(train_dest, index=False)
    df_val.to_csv(val_dest, index=False)
    df_test.to_csv(test_dest, index=False)

    logger.info(f"Saved Train set: {train_dest} ({len(df_train)} rows)")
    logger.info(f"Saved Val set:   {val_dest} ({len(df_val)} rows)")
    logger.info(f"Saved Test set:  {test_dest} ({len(df_test)} rows)")

    # 2. Extract and save incoming operational batches (Batch 1, 2, 3)
    batch_col = config.monitoring.batch_column
    batch_metadata = {}
    if batch_col in df_batches_raw.columns:
        for batch_id in sorted(df_batches_raw[batch_col].unique()):
            sub_batch = df_batches_raw[df_batches_raw[batch_col] == batch_id].copy()
            batch_dest = batches_dir / f"batch_{batch_id}.csv"
            sub_batch.to_csv(batch_dest, index=False)
            batch_metadata[f"batch_{batch_id}"] = {
                "path": str(batch_dest),
                "rows": len(sub_batch),
                "hash": calculate_file_hash(batch_dest),
                "mean_price": float(sub_batch[config.features.target].mean())
            }
            logger.info(f"Saved Operational Batch {batch_id}: {batch_dest} ({len(sub_batch)} rows)")

    metadata = {
        "raw_train_hash": train_hash,
        "raw_batches_hash": batches_hash,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "train_hash": calculate_file_hash(train_dest),
        "val_hash": calculate_file_hash(val_dest),
        "test_hash": calculate_file_hash(test_dest),
        "batches": batch_metadata
    }

    metadata_path = processed_dir / "ingestion_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Data Ingestion stage completed successfully.")
    return metadata


if __name__ == "__main__":
    meta = ingest_data()
    print(f"Ingestion complete. Metadata: {meta}")
