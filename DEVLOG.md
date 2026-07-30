# bardi development log

Reverse-chronological log of maintenance changes. Because this repository is
mirrored into an air-gapped enclave, **every entry records the exact package
and version deltas** (from `uv.lock`) so enclave engineers can update their
package mirror before pulling the change.

---

## 2026-07-30 — CI: run the test suite on GitHub Actions with uv

**Change**

- `.github/workflows/tests.yml` (new): on pull requests, pushes to `main`,
  and manual dispatch — `actions/checkout@v7`, `astral-sh/setup-uv@v9` with
  uv pinned to **0.11.26** (the version the dev environment was proven with),
  then `uv sync --locked` and `uv run pytest -ra` on `ubuntu-latest`.
- `README.md`: status badge and a Development quickstart documenting the same
  two commands (plus the `BARDI_HF_CACHE` note for on-cluster tokenizer tests).

**Rationale**

- CI executes *exactly* the documented local setup — uv provisions the
  interpreter from `.python-version`, installs from `uv.lock` — so a green
  badge is continuous proof the onboarding path works from a clean machine.
- `uv sync --locked` fails if `uv.lock` drifts from `pyproject.toml`, so no
  dependency change can merge without re-locking. This keeps this log's
  air-gap deltas trustworthy.

**Air-gap impact**: none — no changes to `uv.lock`. GitHub Actions do not run
inside the enclave; the workflow is upstream-only. The enclave's equivalent of
this proof is the same two commands against the internal mirror index.

---

## 2026-07-30 — Unpin duckdb (0.8.0 → 1.x); pytest as test runner; self-contained tests

**Change**

- `pyproject.toml`: `duckdb==0.8.0` → `duckdb>=1.0,<2`.
- `pyproject.toml`: added `pytest>=8` to the `dev` dependency group and
  `[tool.pytest.ini_options]` (`testpaths = ["tests"]`,
  `python_files = ["*_tests.py"]` — test files use nonstandard naming).
  `uv run pytest` is now the documented test runner;
  `python -m tests.main_test` is kept for compatibility.
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
- `tests/tokenizer_tests.py`: HF model cache path is now read from
  `BARDI_HF_CACHE` (default remains the ORNL cluster path); the 8
  cache-dependent tests skip when the cache directory is absent.
- `tests/data_handlers_tests.py`: setup now creates `tests/test_data/` and
  removes any leftover `test_db.duckdb`/`.wal` before connecting (duckdb 1.x
  cannot open 0.8-format files).

**Rationale**

- duckdb 0.8.0 is ABI-incompatible with pyarrow 25: exporting query results
  to Arrow (`fetch_arrow_table`) segfaults, killing the whole test run
  (`tests/data_handlers_tests.py`, `from_duckdb`).
- 0.8.0 ships no wheels past cp311, blocking any future Python upgrade.
- Test suite previously required manually generated, gitignored fixtures and
  an ORNL cluster mount; it now runs green anywhere:
  `uv run pytest` → 62 passed, 8 skipped (tokenizer tests, off-cluster).

**Air-gap impact** (deltas in `uv.lock`)

| Package | Old | New | Note |
|---|---|---|---|
| duckdb | 0.8.0 | **1.5.5** | runtime dependency; constraint `>=1.0,<2` |
| pytest | — | **9.1.1** | new, dev group only |
| pluggy | — | **1.6.0** | new, pytest dependency |
| iniconfig | — | **2.3.0** | new, pytest dependency |

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
- `[dependency-groups] dev`: black, flake8 (pytest added by the change above),
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
