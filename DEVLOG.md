# bardi development log

Reverse-chronological log of maintenance changes. Because this repository is
mirrored into an air-gapped enclave, **every entry records the exact package
and version deltas** (from `uv.lock`) so enclave engineers can update their
package mirror before pulling the change.

---

## 2026-07-30 — Unpin duckdb (0.8.0 → 1.x)

**Change**

- `pyproject.toml`: `duckdb==0.8.0` → `duckdb>=1.0,<2`.
- `tests/utils/generate_mock_data.py`: deterministic fixture generation
  (seeds both `random` and `numpy.random`; disjoint, collision-free vocabs of
  200/100/300 = 600 unique words). Added idempotent `ensure_*` helpers that
  create `tests/test_data/{pipeline,embed_gen,split}_test_df.pkl` on demand,
  anchored to the repo (not CWD). Tests call these from `setUpClass`, so the
  suite is self-contained on a fresh checkout.
  - *Snapshot semantics*: the `split_correct` golden column in
    `split_test_df.pkl` is produced by running `CPUSplitter(NewSplit(...))`
    itself, so the splitter test validates determinism/regressions, not
    first-time correctness of the split algorithm.
- `tests/data_handlers_tests.py`: setup now creates `tests/test_data/` and
  removes any leftover `test_db.duckdb`/`.wal` before connecting (duckdb 1.x
  cannot open 0.8-format files).

**Rationale**

- duckdb 0.8.0 is ABI-incompatible with pyarrow 25: exporting query results
  to Arrow (`fetch_arrow_table`) segfaults, killing the whole test run
  (`tests/data_handlers_tests.py`, `from_duckdb`).
- 0.8.0 ships no wheels past cp311, blocking any future Python upgrade.

**Air-gap impact** (deltas in `uv.lock`)

| Package | Old | New | Note |
|---|---|---|---|
| duckdb | 0.8.0 | **1.5.5** | runtime dependency; constraint `>=1.0,<2` |

**Enclave note — DuckDB storage format**: DuckDB 1.x cannot open `.db`/
`.duckdb` files created by 0.8-era releases. Any existing database files in
the enclave must be migrated (`EXPORT DATABASE` with the old version →
`IMPORT DATABASE` with the new) or regenerated from source data. Stale
0.8-format files will cause hard open failures, not silent upgrades.

---

## 2026-07-30 — uv-based dev environment

**Change**

- Adopted `uv` as the environment/dependency manager; committed `uv.lock`.
- `.python-version` → 3.11 (duckdb 0.8.0 wheel ceiling at the time) and
  `requires-python = ">=3.9,<3.12"` cap in `pyproject.toml` (also because
  `setup.py` still uses distutils, removed in Python 3.12).
- `[dependency-groups] dev`: black, flake8,
  replacing the loose `requirements.txt` workflow.
- `flake.nix` devShell for NixOS hosts (provides `uv`; nix-ld hosts can also
  just use uv directly).

**Rationale**: reproducible, identical environments on NixOS and Ubuntu from
a single lockfile; no system Python or PPA required (`uv sync` provisions
CPython 3.11 itself).

**Air-gap impact** (key resolved versions in the initial `uv.lock`; full list
in the lockfile):

| Package | Version |
|---|---|
| python | 3.11 (uv-managed, python-build-standalone) |
| duckdb | 0.8.0 (unpinned to 1.5.5 by the entry above) |
| pyarrow | 25.0.0 |
| polars-u64-idx | 1.33.1 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| gensim | 4.4.0 |
| transformers | 5.14.1 |
| tokenizers | 0.22.2 |
| datasets | 5.0.1 |
| black (dev) | 26.5.1 |
| flake8 (dev) | 7.3.0 |

The enclave mirror must carry all packages resolved in `uv.lock` (plus uv
itself and the CPython 3.11 python-build-standalone archive if interpreters
are also mirrored).
