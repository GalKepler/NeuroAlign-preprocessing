# Plan: Restart neuroalign-preprocessing

## Context

The package has grown overcomplicated (~4,600 lines across 12 source files) with SQLite caching, wide-format transformers, TIV extraction, Pydantic configs, feature stores, and more. The actual requirement (demonstrated in `notebooks/demo.ipynb`) is simple: **aggregate pre-parcellated TSV + JSON metadata pairs into long-format Parquet files**. TIV extraction is handled elsewhere. This plan strips the package down to ~310 lines that do exactly that.

## New Architecture

```
src/neuroalign_preprocessing/
    __init__.py        # version + public API exports
    sessions.py        # sanitize_subject_code, sanitize_session_id, load_sessions
    parsers.py         # read TSV + JSON sidecar -> DataFrame (reusable core)
    aggregate.py       # aggregate_cat12(), aggregate_qsiparc() with streaming Parquet writes
    cli.py             # single CLI entry point, reads .env defaults
```

**Output**: One Parquet file per metric, nested by source:
```
output_dir/
    cat12/
        gm.parquet              # all sessions, GM tissue
        wm.parquet              # all sessions, WM tissue
        ct.parquet              # all sessions, CT tissue
    qsiparc/
        DIPYMAPMRI/
            mapmri_msd.parquet  # all sessions, this workflow+model+param
            mapmri_rtop.parquet
            ...
        NODDI/
            noddi_icvf.parquet
            ...
```

## Key Design Decisions

1. **PyArrow streaming writes** (`pq.ParquetWriter`) — memory is O(one session) regardless of dataset size, not O(all sessions)
2. **One file per metric** — CAT12 splits by tissue (3 files), QSIParc splits by workflow/model+param (nested dirs). Each file is independently loadable with `pd.read_parquet()`
3. **Serial iteration** with tqdm — each session reads tiny files (<100KB); threading adds complexity for no real I/O benefit
4. **Schema alignment** — handle potential column mismatches across sessions by aligning to the first session's schema
5. **No Pydantic** — flat config (a few paths + atlas name) doesn't need it; argparse + dotenv suffice
6. **Drop unnecessary deps** — remove `nibabel`, `scipy`, `pyyaml`, `duckdb`, `pydantic`

## Files to Delete

All existing source files under `src/neuroalign_preprocessing/`:
- `io/` (entire directory) — writer replaced by inline PyArrow writes
- `loaders/` (entire directory) — replaced by `parsers.py` + `aggregate.py`
- `preprocessing/` (entire directory) — pipeline, feature_store, config, transformers, export_cache, cli all eliminated
- `utils/` (entire directory) — `sessions.py` moved to top-level module

Also delete:
- `.cache.db` if it exists in any output directory
- `setup/` directory (migration docs from old repo split, no longer relevant)

## Implementation Steps

### 1. Create `src/neuroalign_preprocessing/sessions.py`
- Migrate `sanitize_subject_code()` and `sanitize_session_id()` from `utils/sessions.py`
- Add `load_sessions(csv_path)` that reads CSV and returns sanitized `subject_code` + `session_id` columns

### 2. Create `src/neuroalign_preprocessing/parsers.py`
- `parse_bids_entities(filename, keys=None)` — extract key-value entities from BIDS filenames
- `read_parcellation(tsv_path, extra_columns=None)` — read TSV, read+flatten JSON sidecar, add extra columns, return DataFrame

### 3. Create `src/neuroalign_preprocessing/aggregate.py`
- `aggregate_cat12(cat12_root, sessions_csv, output_dir, atlas, tissues, mask, compression)` — for each tissue, iterate all sessions and stream-write to `output_dir/cat12/{tissue}.parquet`
- `aggregate_qsiparc(qsiparc_root, sessions_csv, output_dir, atlas, mask, compression)` — auto-discover workflows, then for each unique (workflow, model, param) combo, iterate sessions and stream-write to `output_dir/qsiparc/{workflow}/{model}_{param}.parquet`
- Two-pass approach for QSIParc: first pass discovers all (workflow, model, param) tuples from the first available session; second pass aggregates
- Internal `_StreamingParquetWriter` helper class to handle schema alignment across sessions

### 4. Create `src/neuroalign_preprocessing/cli.py`
- Single entry point with argparse
- Defaults from `.env` via `python-dotenv`
- Args: `--sessions`, `--cat12-root`, `--qsiparc-root`, `--atlas`, `--mask`, `--output`, `--tissues`, `--compression`, `--verbose`

### 5. Update `src/neuroalign_preprocessing/__init__.py`
- Export `aggregate_cat12`, `aggregate_qsiparc`, `read_parcellation`, `load_sessions`, sanitizers

### 6. Delete old source directories
- Remove `io/`, `loaders/`, `preprocessing/`, `utils/`

### 7. Update `pyproject.toml`
- Dependencies: keep `numpy`, `pandas`, `pyarrow`, `tqdm`, `python-dotenv`
- Single entry point: `neuroalign-preprocess = "neuroalign_preprocessing.cli:main"`

### 8. Simplify `.env`
- Keep only: `SESSIONS_CSV`, `CAT12_PARCELLATED_ROOT`, `QSIPARC_PATH`, `ATLAS_NAME`

## Suggested Improvements

1. **Incremental mode**: Before aggregating, read existing Parquet file's `(subject_code, session_id)` pairs and skip already-processed sessions. Cheap to implement with `pq.read_table(columns=["subject_code","session_id"]).to_pandas().drop_duplicates()`.
2. **Validation summary**: After aggregation, print summary stats (sessions found/missing, rows per tissue/workflow, any schema mismatches).
3. **DuckDB consumption examples**: Document how to query the output Parquet files with DuckDB for filtering/pivoting (no code needed in the package, just docs/notebook examples).

## Verification

1. Run `neuroalign-preprocess --verbose` with current `.env` paths
2. Confirm output directory structure: `cat12/gm.parquet`, `cat12/wm.parquet`, `cat12/ct.parquet`, `qsiparc/{workflow}/{model}_{param}.parquet`
3. Load in notebook: `pd.read_parquet("output/cat12/gm.parquet")` — verify columns match demo notebook output
4. Verify each file contains `subject_code` and `session_id` columns for joining
5. Check memory: should stay flat regardless of session count (streaming writes)
6. Verify row counts: sessions x ~400 regions per file

## Size Comparison

| Component | Before | After |
|-----------|--------|-------|
| Source lines | ~4,600 | ~310 |
| Source files | 12 | 5 |
| Dependencies | 10 | 5 |
| CLI entry points | 2 | 1 |
