# bardi development log

Reverse-chronological log of maintenance changes. Because this repository is
mirrored into an air-gapped enclave, **every entry records the exact package
and version deltas** (from `uv.lock`) so enclave engineers can update their
package mirror before pulling the change.

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
| duckdb | 0.8.0 |
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
