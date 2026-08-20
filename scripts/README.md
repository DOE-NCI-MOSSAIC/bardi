# Generate Shared Embeddings Across Synthetic Data Configurations

## Problem

Four synthetic data configurations were each processed independently through
Bardi, producing separate `id_to_token.json` and `embedding_matrix.npy` files.
For federated learning in FrESCO, all four configurations must share a common
vocabulary and embedding matrix. The `id_to_label.json` files are already
consistent across configurations and do not require merging.

## What This Script Does

`generate_shared_embeddings.py` takes the four independently-processed datasets
and produces a single shared vocabulary and embedding matrix that can be used
across all of them. It does this in seven steps:

### Step 1 — Load Input Data

For each of the four configuration directories, the script loads:

- `bardi_processed_data.parquet` — contains the `X` column (integer-encoded
  token sequences), `note_text` (original report text), and `_meta_registry`
  (KY, LA, NJ)
- `id_to_token.json` — maps integer IDs to token strings (e.g.,
  `{0: "<pad>", 1: "cancer", ..., N: "<unk>"}`)

SHA-256 checksums of every input file are recorded for provenance.

### Step 2 — Decode and Filter

Each dataset's `X` column is a list of integers per row (the output of Bardi's
`CPUVocabEncoder`). The script reverses this encoding using the per-config
`id_to_token.json`, mapping each integer back to its token string. This produces
a `text` column of type `List[str]` — the same format that Bardi's
`CPUEmbeddingGenerator` expects as input.

Rows where `_meta_registry == "NJ"` are filtered out.

Row counts before and after filtering, registry breakdowns, and duplicate
`record_document_id` counts are recorded in the manifest.

### Step 3 — Train Shared Word2Vec Embeddings

All decoded token sequences from all four configurations are combined into a
single dataset. This combined dataset is fed through Bardi's
`CPUEmbeddingGenerator`, which trains a Gensim Word2Vec model on the combined
token sequences.

This produces:

- **`id_to_token`** — a new shared vocabulary mapping `{int: str}`, sorted
  alphabetically, with `<pad>` at index 0 and `<unk>` at the last index
- **`embedding_matrix`** — a numpy array of shape `(vocab_size, vector_size)`
  where each row is the Word2Vec embedding for the token at that index

Because Word2Vec learns from co-occurrence patterns in context windows, training
on the combined data produces embeddings that reflect token relationships across
all four configurations.

The Word2Vec parameters used are recorded in the manifest. Defaults:

| Parameter        | Value | Description                                 |
| ---------------- | ----- | ------------------------------------------- |
| `min_word_count` | 1     | Include all tokens regardless of frequency  |
| `vector_size`    | 300   | Dimensionality of embedding vectors         |
| `window`         | 5     | Context window size for Word2Vec            |
| `epochs`         | 30    | Training iterations over the corpus         |
| `seed`           | 42    | Random seed (deterministic with 1 CPU core) |

### Step 4 — Save Shared Artifacts

The shared `id_to_token.json` and `embedding_matrix.npy` are written to the
output directory root.

### Step 5 — Re-encode Datasets

Each of the four decoded datasets is re-encoded using Bardi's `CPUVocabEncoder`
with the new shared vocabulary. This maps each token string back to its integer
ID in the shared `id_to_token`, producing a new `X` column. Tokens not present
in the shared vocabulary are mapped to the `<unk>` ID.

The re-encoded data is written as `bardi_processed_data.parquet` in per-config
subdirectories.

### Step 6 — Distribute Artifacts

The shared `id_to_token.json`, `embedding_matrix.npy`, and each config's
original `id_to_label.json` are copied into each config's output subdirectory so
that every directory is self-contained and ready for FrESCO.

### Step 7 — Write Provenance Manifest

A `manifest.json` is written to the output directory containing everything
needed to audit and reproduce the run (see Provenance section below).

## Output Directory Structure

```
shared_embedding_output/
├── manifest.json                          # Provenance record (see below)
├── id_to_token.json                       # Shared vocabulary
├── embedding_matrix.npy                   # Shared embedding matrix
├── dropped_tokens_<config>.json           # Tokens dropped per config (if any)
├── filtered_kwrds_sampled_with_oracle_run3/
│   ├── bardi_processed_data.parquet       # Re-encoded with shared vocab
│   ├── id_to_token.json                   # Copy of shared vocab
│   ├── embedding_matrix.npy               # Copy of shared embeddings
│   └── id_to_label.json                   # Original labels (unchanged)
├── gen_cluster_kwrds_no_oracle/
│   └── ...
├── note_gen_oracle_only_run3/
│   └── ...
└── words_open_vocab_N_70_ep_10_delta_1e-05_words_70_no_oracle_run3/
    └── ...
```

## Provenance

The script records extensive provenance information in `manifest.json`, validated
at construction time by three pydantic models defined at the top of the script:

- **`W2VParams`** — Word2Vec hyperparameters
- **`ConfigRecord`** — per-configuration statistics (vocab sizes, row counts,
  vocab diff results)
- **`Manifest`** — top-level record containing timestamps, checksums, all
  `ConfigRecord` entries, and the run log

Using pydantic gives typed fields, immediate validation (typos in field names or
wrong types raise errors instead of silently creating bad data), and
`model_dump_json()` for serialization. The JSON output structure is unchanged.

### File Integrity

- **SHA-256 checksums** of every input file (parquets, `id_to_token.json`,
  `id_to_label.json`) and every output file. Researchers can verify that input
  data has not changed since the run and that output files are intact.

### Data Lineage

- **Row counts** before and after NJ filtering, per configuration
- **Registry breakdown** (counts per KY, LA, etc.) after filtering
- **Duplicate `record_document_id` detection** — flags the known issue in
  `gen_cluster_kwrds_no_oracle` (82 duplicates) and any others

### Vocabulary Audit

- **Per-config vocab diff** against the shared vocabulary:
  - How many of the config's original tokens are kept in the shared vocab
  - How many are dropped (and which ones — written to
    `dropped_tokens_<config>.json`)
  - How many new tokens come from other configurations
- Original and shared vocab sizes

### Reproducibility

- **Word2Vec parameters** used for training (from
  `CPUEmbeddingGenerator.get_parameters()`)
- **Python version** and **UTC timestamps** (start and end)
- **Total training sequences** contributed to Word2Vec

### Example manifest.json structure

```json
{
  "timestamp_utc": "2026-08-19T...",
  "completed_utc": "2026-08-19T...",
  "python_version": "3.10.x",
  "base_path": "/mnt/nci/scratch/...",
  "subdirs": ["filtered_kwrds_...", "gen_cluster_...", "..."],
  "output_dir": "/abs/path/to/shared_embedding_output",
  "w2v_params": {"min_word_count": 1, "vector_size": 300, "window": 5, "epochs": 30, "seed": 42},
  "input_checksums": {
    "filtered_kwrds_.../bardi_processed_data.parquet": "abc123...",
    "filtered_kwrds_.../id_to_token.json": "def456..."
  },
  "output_checksums": {
    "id_to_token.json": "789abc...",
    "embedding_matrix.npy": "012def...",
    "filtered_kwrds_.../bardi_processed_data.parquet": "345ghi..."
  },
  "per_config": {
    "filtered_kwrds_sampled_with_oracle_run3": {
      "original_vocab_size": 5432,
      "original_token_count": 5430,
      "rows_before_filter": 178247,
      "rows_after_filter": 165000,
      "duplicate_record_document_ids": 0,
      "registry_counts": {"KY": 80000, "LA": 85000},
      "token_lists_contributed": 165000,
      "tokens_in_shared_vocab": 5400,
      "tokens_dropped_from_original": 32,
      "tokens_new_from_other_configs": 1200
    }
  },
  "log": ["[2026-08-19T...] Processing: filtered_kwrds_...", "..."],
  "total_training_sequences": 650000,
  "shared_vocab_size": 6632,
  "shared_token_count": 6630,
  "embedding_matrix_shape": [6632, 300],
  "w2v_model_params": {"...from Bardi's get_parameters()..."}
}
```

## Usage

```bash
cd /path/to/bardi
python scripts/generate_shared_embeddings.py
```

### Configuration

Edit the constants at the top of the script before running:

- `BASE_PATH` — path to the directory containing the four config subdirectories
- `SUBDIRS` — the four subdirectory names
- `OUTPUT_DIR` — where outputs are written (default: `shared_embedding_output`)
- `W2V_PARAMS` — Word2Vec hyperparameters

### Verifying Results

After running, researchers can verify the output by:

1. **Checking file integrity**: Compare SHA-256 checksums in `manifest.json`
   against the actual files using `sha256sum <file>`.

2. **Inspecting the vocab diff**: Review `manifest.json`'s `per_config` section
   to see how each config's vocabulary maps to the shared vocabulary. If any
   tokens were dropped (due to `min_word_count` filtering), they are listed in
   `dropped_tokens_<config>.json`.

3. **Validating row counts**: Confirm that `rows_before_filter` matches the
   known dataset size (178,247 for most configs, 178,329 for
   `gen_cluster_kwrds_no_oracle` due to the 82 duplicate rows).

4. **Spot-checking the decode/re-encode round-trip**: Load an original parquet,
   decode its `X` column with the original `id_to_token.json`, then re-encode
   with the shared `id_to_token.json`. The result should match the corresponding
   output parquet. Any differences will be limited to tokens that were dropped
   from the shared vocabulary (mapped to `<unk>`).

5. **Reproducing the run**: Using the same Python version, Bardi version, and
   `W2V_PARAMS` from the manifest, the run should produce identical results
   (assuming `seed` is set and a single CPU core is used for full determinism).

## Dependencies

- Python 3.8+
- [Bardi](https://github.com/DOE-NCI-MOSSAIC/bardi) (provides
  `CPUEmbeddingGenerator`, `CPUVocabEncoder`)
- Gensim (Word2Vec, pulled in by Bardi)
- Polars
- PyArrow
- NumPy
- Pydantic (manifest validation; not a Bardi dependency — install separately in
  the enclave environment)

## Notes

- **NJ filtering**: All rows with `_meta_registry == "NJ"` are removed before
  processing, as these are not to be shared with LANL.

- **Duplicate rows**: The script detects and logs duplicate `record_document_id`
  values. The known 82 duplicates in `gen_cluster_kwrds_no_oracle` will be
  flagged in the manifest. The script does not deduplicate — it only reports. If
  deduplication is desired, it should be done before running this script.

- **Word2Vec determinism**: Full reproducibility requires `seed` to be set
  (default: 42) and `cores=1`. With multiple cores, Gensim's Word2Vec may
  produce slightly different results across runs due to thread scheduling. The
  script uses all available cores by default for performance; set `cores=1` in
  `W2V_PARAMS` if exact reproducibility is required.

- **`min_word_count`**: Set to 1 by default so that every token from every
  configuration is included in the shared vocabulary. Increasing this value will
  exclude infrequent tokens, which will be mapped to `<unk>` during re-encoding.
  Any dropped tokens are recorded in `dropped_tokens_<config>.json`.
