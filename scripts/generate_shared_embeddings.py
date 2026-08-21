"""
Generate a shared vocabulary (id_to_token) and embedding matrix across
four synthetic data configurations so they can be used together in FrESCO.

Workflow:
  1. Load each config's parquet + id_to_token.json
  2. Decode the X column (List[Int64]) back to token strings (List[str])
  3. Filter out NJ rows
  4. Combine all decoded text into one dataset
  5. Train Word2Vec via Bardi's CPUEmbeddingGenerator on the combined data
  6. Re-encode each dataset with CPUVocabEncoder using the new shared vocab
  7. Save shared artifacts, re-encoded parquets, and provenance manifest
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from bardi.nlp_engineering import CPUEmbeddingGenerator, CPUVocabEncoder


# ── Pydantic models for provenance manifest ──────────────────────────────

class W2VParams(BaseModel):
    min_word_count: int
    vector_size: int
    window: int
    epochs: int
    seed: int


class ConfigRecord(BaseModel):
    original_vocab_size: int
    original_token_count: int
    rows_before_filter: int
    rows_after_filter: Optional[int] = None
    duplicate_record_document_ids: Optional[int] = None
    registry_counts: Optional[Dict[str, int]] = None
    token_lists_contributed: int = 0
    tokens_in_shared_vocab: Optional[int] = None
    tokens_dropped_from_original: Optional[int] = None
    tokens_new_from_other_configs: Optional[int] = None


class Manifest(BaseModel):
    timestamp_utc: str
    completed_utc: Optional[str] = None
    python_version: str
    base_path: Path
    subdirs: List[str]
    output_dir: Path
    w2v_params: W2VParams
    input_checksums: Dict[str, str] = {}
    output_checksums: Dict[str, str] = {}
    per_config: Dict[str, ConfigRecord] = {}
    log: List[str] = []
    total_training_sequences: Optional[int] = None
    shared_vocab_size: Optional[int] = None
    shared_token_count: Optional[int] = None
    embedding_matrix_shape: Optional[List[int]] = None
    w2v_model_params: Optional[Dict[str, Any]] = None


# ── Configuration ──────────────────────────────────────────────────────────

BASE_PATH = Path("/mnt/nci/scratch/krawczukp/gen_synth_data_results/data")

SUBDIRS = [
    "filtered_kwrds_sampled_with_oracle_run3",
    "gen_cluster_kwrds_no_oracle",
    "note_gen_oracle_only_run3",
    "words_open_vocab_N_70_ep_10_delta_1e-05_words_70_no_oracle_run3",
]

OUTPUT_DIR = Path("shared_embedding_output")

# Word2Vec parameters — adjust as needed
W2V_PARAMS = {
    "min_word_count": 1,    # Include all tokens (set higher to prune rare ones)
    "vector_size": 300,
    "window": 5,
    "epochs": 30,
    "seed": 42,
}


# ── Provenance helpers ─────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str, manifest: Manifest):
    """Print a message and append it to the manifest log."""
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = f"[{timestamp}] {msg}"
    print(entry)
    manifest.log.append(entry)


# ── Helper: decode X column from integers to token strings ─────────────────

def decode_x_column(df: pl.DataFrame, id_to_token: dict) -> pl.DataFrame:
    """Replace each integer in the X column with its token string.

    Parameters
    ----------
    df : polars DataFrame with an 'X' column of type List[Int64]
    id_to_token : dict mapping int -> str (e.g. {0: '<pad>', 1: 'cancer', ...})

    Returns
    -------
    polars DataFrame with 'X' replaced by a 'text' column of type List[str]
    """
    decoded_lists = []
    for row in df["X"].to_list():
        if row is None:
            decoded_lists.append(None)
        else:
            decoded_lists.append(
                [id_to_token.get(token_id, "<unk>") for token_id in row]
            )

    return df.drop("X").with_columns(
        pl.Series("text", decoded_lists, dtype=pl.List(pl.Utf8))
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Provenance record — written to manifest.json at the end
    manifest = Manifest(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        base_path=BASE_PATH,
        subdirs=SUBDIRS,
        output_dir=OUTPUT_DIR.resolve(),
        w2v_params=W2VParams(**W2V_PARAMS),
    )

    # ── Step 1 & 2: Load, decode, and filter each dataset ──────────────

    decoded_datasets = {}  # subdir -> polars DataFrame (with 'text' column)
    all_text_lists = []    # combined token lists for Word2Vec training
    original_tokens_map = {}  # subdir -> set of original tokens (transient)

    for subdir in SUBDIRS:
        dir_path = BASE_PATH / subdir

        log(f"Processing: {subdir}", manifest)

        # Checksum input files
        parquet_path = dir_path / "bardi_processed_data.parquet"
        vocab_path = dir_path / "id_to_token.json"
        manifest.input_checksums[f"{subdir}/bardi_processed_data.parquet"] = sha256_file(parquet_path)
        manifest.input_checksums[f"{subdir}/id_to_token.json"] = sha256_file(vocab_path)

        # Load id_to_token.json
        id_to_token = json.loads(vocab_path.read_text())
        # JSON keys are strings — convert to int
        id_to_token = {int(k): v for k, v in id_to_token.items()}

        # Extract the set of real tokens (excluding <pad> and <unk>)
        original_tokens = {
            v for v in id_to_token.values() if v not in ("<pad>", "<unk>")
        }
        original_vocab_size = len(id_to_token)
        original_token_count = len(original_tokens)
        log(f"  Original vocab size (with special tokens): {original_vocab_size}", manifest)

        # Load parquet
        table = pq.read_table(parquet_path)
        df = pl.from_arrow(table)
        rows_before_filter = df.height
        log(f"  Rows before NJ filter: {rows_before_filter}", manifest)

        # Check for duplicate record_document_id (Ada noted 82 dupes in one config)
        duplicate_record_document_ids = None
        if "record_document_id" in df.columns:
            n_dupes = df.height - df["record_document_id"].n_unique()
            duplicate_record_document_ids = n_dupes
            if n_dupes > 0:
                log(f"  WARNING: {n_dupes} duplicate record_document_id values", manifest)

        # Filter out NJ rows
        rows_after_filter = None
        registry_counts = None
        if "_meta_registry" in df.columns:
            df = df.filter(pl.col("_meta_registry") != "NJ")
            rows_after_filter = df.height
            log(f"  Rows after NJ filter:  {df.height}", manifest)

            # Record registry breakdown
            rc = (
                df.group_by("_meta_registry")
                .len()
                .sort("_meta_registry")
            )
            registry_counts = {
                row["_meta_registry"]: row["len"]
                for row in rc.iter_rows(named=True)
            }

        # Decode X column from List[Int64] -> List[str]
        df = decode_x_column(df, id_to_token)
        decoded_datasets[subdir] = df

        # Collect all token lists for combined Word2Vec training
        text_lists = df["text"].drop_nulls().to_list()
        all_text_lists.extend(text_lists)
        token_lists_contributed = len(text_lists)
        log(f"  Token lists collected:  {token_lists_contributed}", manifest)

        # Build ConfigRecord and store
        manifest.per_config[subdir] = ConfigRecord(
            original_vocab_size=original_vocab_size,
            original_token_count=original_token_count,
            rows_before_filter=rows_before_filter,
            rows_after_filter=rows_after_filter,
            duplicate_record_document_ids=duplicate_record_document_ids,
            registry_counts=registry_counts,
            token_lists_contributed=token_lists_contributed,
        )
        original_tokens_map[subdir] = original_tokens

    log(f"Total token lists for Word2Vec training: {len(all_text_lists)}", manifest)
    manifest.total_training_sequences = len(all_text_lists)

    # ── Step 3: Train shared embeddings via Bardi ──────────────────────

    # Build a PyArrow Table with a single 'text' column of List[str]
    # This is the format CPUEmbeddingGenerator expects
    combined_df = pl.DataFrame({"text": all_text_lists})
    combined_table = combined_df.to_arrow()

    log("Training Word2Vec on combined data...", manifest)
    embedding_generator = CPUEmbeddingGenerator(
        fields=["text"],
        **W2V_PARAMS,
    )

    combined_table, artifacts = embedding_generator.run(
        data=combined_table, artifacts={}
    )

    shared_id_to_token = artifacts["id_to_token"]
    shared_embedding_matrix = artifacts["embedding_matrix"]

    # Record shared vocab info
    shared_tokens = {
        v for v in shared_id_to_token.values() if v not in ("<pad>", "<unk>")
    }
    manifest.shared_vocab_size = len(shared_id_to_token)
    manifest.shared_token_count = len(shared_tokens)
    manifest.embedding_matrix_shape = list(shared_embedding_matrix.shape)
    log(f"Shared vocab size (with special tokens): {len(shared_id_to_token)}", manifest)
    log(f"Embedding matrix shape: {shared_embedding_matrix.shape}", manifest)

    # Record Word2Vec model parameters from Bardi
    manifest.w2v_model_params = embedding_generator.get_parameters()

    # ── Vocab diff: what changed per config ────────────────────────────

    for subdir in SUBDIRS:
        record = manifest.per_config[subdir]
        original_tokens = original_tokens_map[subdir]

        tokens_kept = original_tokens & shared_tokens
        tokens_dropped = original_tokens - shared_tokens
        tokens_new = shared_tokens - original_tokens

        record.tokens_in_shared_vocab = len(tokens_kept)
        record.tokens_dropped_from_original = len(tokens_dropped)
        record.tokens_new_from_other_configs = len(tokens_new)

        # Write the dropped tokens list for audit (if any)
        if tokens_dropped:
            dropped_path = OUTPUT_DIR / f"dropped_tokens_{subdir}.json"
            dropped_path.write_text(json.dumps(sorted(tokens_dropped), indent=2))
            log(f"  {subdir}: {len(tokens_dropped)} tokens dropped (see dropped_tokens_{subdir}.json)", manifest)

        log(
            f"  {subdir}: {record.original_token_count} original -> "
            f"{record.tokens_in_shared_vocab} kept, "
            f"{record.tokens_new_from_other_configs} new from other configs",
            manifest,
        )

    # ── Step 4: Save shared artifacts ──────────────────────────────────

    embedding_generator.write_artifacts(write_path=str(OUTPUT_DIR), artifacts=artifacts)  # str(): Bardi API expects str
    log(f"Shared id_to_token.json and embedding_matrix.npy written to: {OUTPUT_DIR}", manifest)

    # Checksum shared artifacts
    manifest.output_checksums["id_to_token.json"] = sha256_file(
        OUTPUT_DIR / "id_to_token.json"
    )
    manifest.output_checksums["embedding_matrix.npy"] = sha256_file(
        OUTPUT_DIR / "embedding_matrix.npy"
    )

    # ── Step 5: Re-encode each dataset with the shared vocab ───────────

    vocab_encoder = CPUVocabEncoder(fields=["text"])

    for subdir, df in decoded_datasets.items():
        log(f"Re-encoding: {subdir}", manifest)

        table = df.to_arrow()

        # Re-encode using the shared vocab — produces 'X' column
        encoded_table, _ = vocab_encoder.run(
            data=table,
            artifacts={"id_to_token": shared_id_to_token},
        )

        # Write re-encoded parquet
        subdir_output = OUTPUT_DIR / subdir
        subdir_output.mkdir(exist_ok=True)
        output_path = subdir_output / "bardi_processed_data.parquet"
        pq.write_table(
            encoded_table,
            output_path,
            compression="snappy",
            use_dictionary=False,
        )

        # Checksum output parquet
        manifest.output_checksums[f"{subdir}/bardi_processed_data.parquet"] = sha256_file(output_path)
        log(f"  Written to: {output_path}", manifest)

    # ── Step 6: Copy shared artifacts into each subdir ─────────────────
    #    (so each config dir is self-contained for FrESCO)

    for subdir in SUBDIRS:
        subdir_output = OUTPUT_DIR / subdir

        # Copy id_to_token.json
        (subdir_output / "id_to_token.json").write_text(
            json.dumps(shared_id_to_token, indent=4)
        )

        # Copy embedding_matrix.npy
        np.save(
            subdir_output / "embedding_matrix.npy",
            shared_embedding_matrix,
        )

        # Copy the original id_to_label.json (consistent across all 4)
        orig_label_path = BASE_PATH / subdir / "id_to_label.json"
        if orig_label_path.exists():
            id_to_label = json.loads(orig_label_path.read_text())
            (subdir_output / "id_to_label.json").write_text(
                json.dumps(id_to_label, indent=4)
            )

            manifest.input_checksums[f"{subdir}/id_to_label.json"] = sha256_file(orig_label_path)

    # ── Step 7: Write provenance manifest ──────────────────────────────

    manifest.completed_utc = datetime.now(timezone.utc).isoformat()

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    log(f"Provenance manifest written to: {manifest_path}", manifest)
    log("Done.", manifest)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Shared vocab size:        {manifest.shared_vocab_size}")
    print(f"Embedding matrix shape:   {manifest.embedding_matrix_shape}")
    print(f"Total training sequences: {manifest.total_training_sequences}")
    print(f"Output directory:         {manifest.output_dir}")
    print(f"Manifest:                 {manifest_path}")
    for subdir in SUBDIRS:
        r = manifest.per_config[subdir]
        print(f"\n  {subdir}:")
        print(f"    Rows (after filter): {r.rows_after_filter if r.rows_after_filter is not None else r.rows_before_filter}")
        print(f"    Original tokens:     {r.original_token_count}")
        print(f"    Kept in shared:      {r.tokens_in_shared_vocab}")
        print(f"    Dropped:             {r.tokens_dropped_from_original}")
        print(f"    New from others:     {r.tokens_new_from_other_configs}")


if __name__ == "__main__":
    main()
