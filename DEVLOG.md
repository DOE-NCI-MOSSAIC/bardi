# bardi development log

Reverse-chronological log of maintenance changes. Because this repository is
mirrored into an air-gapped enclave, **every entry records the exact package
and version deltas** (from `uv.lock`) so enclave engineers can update their
package mirror before pulling the change.

> **Note on version tables**: `uv.lock` is a universal lockfile covering the
> full `requires-python` range (3.9–3.11), so some packages resolve to
> *multiple* versions (e.g. one for 3.9/3.10, another for 3.11). The tables
> below list the versions resolved for the primary dev interpreter
> (Python 3.11, per `.python-version`). The enclave mirror must carry
> **everything in `uv.lock`**, not just the versions tabled here.

---

## 2026-07-30 — Enclave pull checklist (branch ada/20260730/dev-env-uv → enclave)

Consolidated pre-pull review of the whole branch (16 commits vs `main`). The
entries below this one carry the per-change details and delta tables; this is
the walk-down list for the engineer doing the enclave pull. **Air-gap impact:
the checklist itself** — no code or dependency changes in this entry.

### A. Environment / mirror prerequisites

- [ ] **Mirror the entire `uv.lock`** (2,515 lines, brand-new file on this
  branch) — not deltas. It is a universal 3.9–3.11 lock, so some packages pin
  **multiple versions**: hypothesis 6.164.0 + 6.141.1, transformers 5.14.1 +
  4.57.6, duckdb 1.5.5 + 1.4.5, pytest 9.1.1 + 8.4.2, pandas 3.0.5 + 2.3.3,
  pyarrow 25.0.0 + 21.0.0, and *three* numpy (2.4.6/2.2.6/2.0.2) and scipy
  streams. Primary 3.11 resolution:
  duckdb 1.5.5, polars-u64-idx 1.33.1, pyarrow 25.0.0, pandas 3.0.5,
  numpy 2.4.6, gensim 4.4.0, transformers 5.14.1, tokenizers 0.22.2,
  hypothesis 6.164.0, pytest 9.1.1.
- [ ] **uv itself**: proven with **0.11.26** (the version pinned in CI).
  Offline operation needs either a configured internal index
  (`UV_INDEX_URL`/`UV_DEFAULT_INDEX`) or a pre-populated uv cache plus
  `--offline`.
- [ ] **Interpreter (hard blocker)**: `.python-version` = 3.11, and `uv sync`
  by default downloads python-build-standalone **from GitHub** — blocked in
  the enclave. Either mirror the archive and set
  `UV_PYTHON_INSTALL_MIRROR`, or use a system CPython 3.11 with
  `UV_PYTHON_DOWNLOADS=never`.
- [ ] Ignore inside the enclave: `.github/workflows/tests.yml` (Actions don't
  run there) and `flake.nix` (NixOS-only convenience).

#### Mirror build procedure (for the enclave admins)

`uv.lock` records the **exact artifact URL and SHA256 hash of every wheel and
sdist** — reproducibility means mirroring exactly those files. `uv sync
--locked` then refuses to re-resolve, fails on any drift from
`pyproject.toml`, and hash-checks every install, so the enclave build is
bit-for-bit auditable against the committed lock.

1. **Extract the artifact set** (connected host). Either parse `uv.lock`
   directly (TOML; each `[[package]]` lists `wheels`/`sdist` with `url` +
   `hash`) and fetch exactly those URLs, or:

   ```
   uv export --frozen --all-groups --no-emit-project -o requirements-enclave.txt
   pip download -r requirements-enclave.txt --require-hashes -d wheelhouse/ \
       --python-version 311 --platform manylinux2014_x86_64 --only-binary=:all:
   ```

   `--require-hashes` enforces the lock's hashes at download time. Repeat per
   target platform/python if the enclave runs more than 3.11-on-linux-x86_64.
   Prune only by *platform/python slice*, never by hand-picking versions
   (see the multi-version pins above).
2. **Serve it inside**: a PEP 503 simple index (pypiserver/devpi/Nexus) via
   `UV_DEFAULT_INDEX`, or a flat wheelhouse directory via `UV_FIND_LINKS`.
3. **Mirror the two non-PyPI artifacts**: the standalone **uv 0.11.26**
   binary, and the CPython 3.11 python-build-standalone archive
   (`UV_PYTHON_INSTALL_MIRROR`) — or use system CPython 3.11 with
   `UV_PYTHON_DOWNLOADS=never` (the hard-blocker item above).
4. **Reproduce and verify in the enclave**:

   ```
   uv sync --locked --offline
   uv run pytest -ra
   ```

- Caveat — **sdist-only packages**: `--only-binary=:all:` in step 1 fails
  fast if any locked package lacks a wheel for the enclave platform; if so,
  pre-build the wheel outside rather than mirroring build backends.
- Ongoing: CI runs `uv sync --locked`, so no dependency change can merge
  without re-locking — future entries' delta tables are the mirror update
  list. The full mirror is a one-time cost for this branch (first `uv.lock`);
  afterwards it's deltas.

### B. Behavior changes that touch real data

- [ ] **duckdb 0.8.0 → 1.5.5 (breaking unpin; highest-risk change)**:
  - **Storage format (hard blocker)**: 1.x **cannot open** `.db`/`.duckdb`
    files created by 0.8-era releases. Migrate existing enclave database
    files (`EXPORT DATABASE` under 0.8 → `IMPORT DATABASE` under 1.x) or
    regenerate from source data. Stale files fail hard on open — no silent
    upgrade. (Also noted in the unpin entry below.)
  - `from_duckdb()` executes caller SQL verbatim, and the 0.8→1.5 SQL
    dialect changed (stricter casts, division semantics, new reserved
    words) — enclave queries written for 0.8 may error or silently differ.
- [ ] **Deprecation migrations are drop-in** (verified value- *and*
  dtype-identical on polars 1.33.1): `replace`→`replace_strict` in
  label_processor/vocab_encoder/splitter; duckdb `to_arrow_table()`.
  Caveat: the splitter `split_type="map"` path has **no test coverage** —
  if enclave pipelines use it, watch it.
- [ ] **Regex library: zero pattern changes.** Only a docstring escape fix in
  `regex_lib.py`; `git diff main..HEAD -- tests/regex_tests.py` is empty
  (frozen file intact for the planned enclave comparison). The 51 vectors
  are duplicated verbatim in `tests/regex_vectors.py` with
  production-engine (polars) outputs — all byte-identical.
- [ ] **Lock resolves far newer libs than the code was written against**
  (pandas 3.0.5, numpy 2.4.6, transformers 5.14.1). The suite is green
  off-cluster, but the enclave-only paths in section C are untested outside.

### C. Tests that will activate for the FIRST time inside the enclave

- [ ] **`tests/regex_multi_tests.py`** (was `regex_multi_test.py`; never
  collected before this branch):
  - Needs the real pickle: default
    `tests/test_data/recurrence_raw_data_sample.pkl`, or set
    `BARDI_RECURRENCE_SAMPLE=/path/to/it`. Row selection is deterministic
    via `BARDI_SAMPLE_ROW_SEED` (default 0); expects a `text_all` column.
  - **Pickle-compat risk**: `pd.read_pickle` under pandas 3.0.5 of a pickle
    presumably written by pandas 1.x. If it fails to load, that is an
    environment finding, not a bardi bug (workaround: re-pickle/convert
    with the old enclave environment).
  - It now **asserts invariants** on real text (no `\r`/`\n`/`\t`, no
    double spaces, no backslashes, uppercase runs ⊆ the 13 `*TOKEN`
    strings, determinism). A failure is a *regex-chain finding* to record,
    not necessarily a regression. Stepwise output:
    `uv run pytest tests/regex_multi_tests.py -s`; sweep rows by varying
    the seed.
- [ ] **8 tokenizer tests** (gated on `BARDI_HF_CACHE`, default
  `/mnt/nci/scratch/hf_shared_cache`) will unskip on-cluster. Risk:
  transformers **5.x** reading a shared HF cache populated by an older
  transformers (cache-layout / `from_pretrained` compat; no re-download
  possible in the enclave). Failures there → suspect transformers 5 vs the
  cache before suspecting bardi changes.
- [ ] All other tests self-generate fixtures in setup (`tests/test_data/` is
  gitignored); no enclave data needed for them.

### D. Runner expectations

- [ ] Runner is `uv run pytest -ra`. Off-cluster baseline: **102 passed,
  10 skipped** (8 tokenizer + 2 multi-expression). Inside the enclave with
  both gates satisfied, expect **0 skips** (~112 passed).
- [ ] `python -m tests.main_test` still exists but **always exits 0** — do
  not use it for pass/fail in enclave scripting.

---

## 2026-07-30 — Resolve all DeprecationWarnings surfaced by the test suite

**Change**

- `bardi/nlp_engineering/{label_processor,vocab_encoder,splitter}.py`: polars
  `.replace(..., default=...)` / `return_dtype=...` (deprecated since polars
  1.0 — the deprecation shim already redirected to `replace_strict`) →
  `.replace_strict(...)` with identical arguments. Verified drop-in
  equivalent (same values *and* dtypes) on polars 1.33.1 before switching.
  The splitter's `split_type="map"` path didn't warn in CI only because no
  test exercises it; it used the same deprecated parameter and was migrated
  too.
- `bardi/data/data_handlers.py`: duckdb `fetch_arrow_table()` (deprecated in
  duckdb 1.x) → `to_arrow_table()`.
- `bardi/nlp_engineering/regex_library/regex_lib.py`: docstring-only fix for
  `DeprecationWarning: invalid escape sequence '\ '` (a future SyntaxError) —
  `The result\ \r` → `The result\\ \\r` in the `get_whitespace_regex`
  example, matching the `\\n`/`\\t` escaping already used on the same line.
  **No regex pattern strings were touched**; the polars-engine parity suite
  (51 frozen vectors, byte-identical) passes unchanged.
- Verified with `uv run pytest -W error::DeprecationWarning`: 102 passed,
  10 skipped — any remaining deprecation in an exercised path would now fail.

**Air-gap impact** (deltas in `uv.lock`)

None — no dependency changes.

---

## 2026-07-30 — Regex test hardening: production-engine parity, gated sample test, hypothesis

**Change**

- `tests/regex_vectors.py` (new, data-only): the 51 vectors from
  `tests/regex_tests.py` copied verbatim (that file is frozen for enclave
  comparison), each annotated with the production polars-engine output, plus
  new edge vectors (empty/unicode/must-not-match/near-miss) with empirically
  captured expected values.
- `tests/regex_polars_tests.py` (new): drives every vector through the
  production path (`CPUNormalizer.run`, polars' Rust regex engine) — the
  frozen tests only prove behavior under Python `re.sub`, which production
  never uses. Includes null-passthrough and full-default-chain smoke tests.
- `tests/regex_multi_test.py` → `tests/regex_multi_tests.py`: now collected
  (pytest matches `*_tests.py`), skips unless the ORNL sample pickle exists
  (`BARDI_RECURRENCE_SAMPLE`, like the `BARDI_HF_CACHE` gate), deterministic
  row selection (`BARDI_SAMPLE_ROW_SEED`), and real invariant assertions
  including a production full-chain run. Previously it was never collected,
  used a random row, and asserted nothing.
- `tests/regex_property_tests.py` (new): hypothesis property tests of the
  full default `PathologyReportRegexSet` chain through `CPUNormalizer` —
  never raises; schema/row count preserved; no `\r`/`\n`/`\t`; no consecutive
  whitespace; no backslashes; uppercase runs limited to the 13 `*TOKEN`
  substitution strings; deterministic; null in → null out. Derandomized CI
  profile (`max_examples=75`), so runs are reproducible.
- `pyproject.toml`: `hypothesis>=6` added to the `dev` group; re-locked.
- `.gitignore`: added `.hypothesis/` (local example database).

**Findings (characterization, no library/pattern changes)**

- **No re-vs-polars divergence** on any of the 51 frozen vectors: both
  engines produce byte-identical output for every one (captured 2026-07-30
  with polars-u64-idx 1.33.1). The suspected hotspots behave identically,
  including unset-group expansion in the cassette-spacing alternation
  (both engines expand unset `$1-$3`/`\1-\3` groups to empty strings — which
  silently *erases* second-branch matches like `c2-c3`).
- Edge vectors document current over-/under-matching (e.g.
  `12 to 15 percent of ca 12345` → `ADDRESSTOKEN`; `2 x 4` prose →
  `DIMENSIONTOKEN`; `\test` → ` est`; 4-digit-middle specimen IDs unmatched)
  and an order dependence (`block: 1-e` ends as `block 1 e`, not
  `CASSETTETOKEN`, because rule 7 rewrites it before rule 29 runs). These are
  recorded as-is; fixing them is future work now protected by this net.

**Air-gap impact** (deltas in `uv.lock`)

| Package | Old | New | Note |
|---|---|---|---|
| hypothesis | — | **6.164.0** | new, dev group only (3.10+; universal lock also pins **6.141.1** for 3.9) |
| sortedcontainers | — | **2.4.0** | new, hypothesis dependency |

hypothesis' remaining dependencies (`attrs` 26.1.0, `exceptiongroup` 1.3.1
for <3.11) were already in the lock.

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
