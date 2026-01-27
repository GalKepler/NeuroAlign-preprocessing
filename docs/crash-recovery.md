# Crash Recovery and Incremental Loading

The pipeline automatically saves progress to a SQLite cache as sessions are loaded. If the pipeline crashes or is interrupted, you can recover your data.

## How It Works

### During Normal Pipeline Run

```bash
neuroalign-preprocess
```

The pipeline:
1. Loads each session from disk
2. **Immediately saves to SQLite cache** (`.cache.db`)
3. Continues to next session
4. At the end, exports cache → Parquet
5. Deletes cache

**If it crashes at step 3:** Your data is safe in the cache!

## Recovering from a Crash

### Option 1: Resume Loading (Recommended)

Simply rerun the pipeline:

```bash
neuroalign-preprocess
```

The pipeline will:
- Detect cached sessions
- Skip them
- Continue loading remaining sessions
- Export everything to Parquet

### Option 2: Export What You Have

If you want to use the partially loaded data immediately:

```bash
# Check what's in the cache
neuroalign-export-cache data/processed --status

# Export cache to Parquet files
neuroalign-export-cache data/processed

# Now you can use FeatureStore normally
```

### Option 3: Load Directly from Cache

The FeatureStore can read from cache automatically:

```python
from neuroalign_preprocessing.preprocessing import FeatureStore

store = FeatureStore("data/processed")

# This works even if export never happened
# It will try Parquet first, then fall back to SQLite cache
df = store.load_long("anatomical_gm")

# Check what's available
summary = store.summary()
print(f"Cache status: {summary['cache']}")
```

## Cache Status

Check cache status at any time:

```bash
neuroalign-export-cache data/processed --status
```

Output:
```
============================================================
CACHE STATUS
============================================================
Location: data/processed/.cache.db
Size: 45.23 MB
Anatomical sessions: 287
Diffusion sessions: 0

Tables: 3
  - anatomical_gm
  - anatomical_wm
  - anatomical_ct
============================================================
```

## Example: Interrupted Pipeline

```bash
# Start pipeline
neuroalign-preprocess

# ... pipeline runs for 30 minutes, loads 300/500 sessions ...
# ❌ Computer crashes or pipeline errors

# Check what was saved
neuroalign-export-cache data/processed --status
# Shows: 300 sessions in cache

# Option A: Resume and finish loading
neuroalign-preprocess
# Skips 300 cached sessions, loads remaining 200

# Option B: Export what you have now
neuroalign-export-cache data/processed
# Creates Parquet files with 300 sessions
```

## Cache Location

The cache is stored as `.cache.db` in your output directory:

```
data/processed/
├── .cache.db          # ← SQLite cache (auto-created during loading)
├── long/              # Parquet long format (created after export)
├── wide/              # Parquet wide format (created after export)
├── metadata.parquet
└── manifest.json
```

## Cleaning Up

The cache is automatically deleted after successful export. To manually delete:

```bash
rm data/processed/.cache.db
```

Or keep it when exporting:

```bash
neuroalign-export-cache data/processed --keep-cache
```

## Technical Details

- **Format**: SQLite database (single file)
- **Performance**: ~1ms per session write (negligible overhead)
- **Size**: ~10-20% of final Parquet size
- **Transactions**: Each session write is committed (ACID guarantees)
- **Concurrency**: Thread-safe (supports parallel loading)

## FAQ

**Q: Does this slow down the pipeline?**
A: No, SQLite writes are < 1ms per session. The overhead is negligible compared to file I/O.

**Q: Can I delete the cache?**
A: Yes, once data is exported to Parquet, the cache is redundant. It's auto-deleted by default.

**Q: What if I force-quit (Ctrl+C)?**
A: The last session being processed might be incomplete, but all previous sessions are safely committed.

**Q: Can I run multiple pipelines in parallel?**
A: Yes, each output directory has its own cache. Just use different `--output` paths.

**Q: Does this work with `--force`?**
A: Yes, `--force` clears both cache and Parquet, starting fresh.
