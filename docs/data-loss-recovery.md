# Data Loss Recovery Guide

## What Happened

When you ran `neuroalign-export-cache`, there was a critical bug where the export process **overwrote** your existing Parquet files instead of **merging** with them.

**Before:**
- Parquet files: >1000 sessions
- Cache: 99 sessions

**After `export-cache`:**
- Parquet files: 99 sessions (only what was in cache)
- Lost: >900 sessions

## Bug Fixed

The bug has been fixed in:
- `save_anatomical_long()`: Now merges with existing Parquet
- `save_diffusion_long()`: Now merges with existing Parquet
- `save_tiv()`: Now merges with existing TIV data

Future exports will **merge** instead of **overwrite**.

## Recovery Options

### Option 1: Reload from Source Data (Recommended)

If you still have the original CAT12 and QSIRecon data:

```bash
# Clear everything and start fresh
rm -rf /media/storage/neuroalign/data/*

# Reload all sessions
neuroalign-preprocess
```

This will reload all >1000 sessions from the original neuroimaging data.

### Option 2: Restore from Backup

If you have a backup of `/media/storage/neuroalign/data`:

```bash
# Restore Parquet files from backup
cp -r /path/to/backup/long /media/storage/neuroalign/data/
cp -r /path/to/backup/wide /media/storage/neuroalign/data/
cp /path/to/backup/metadata.parquet /media/storage/neuroalign/data/
cp /path/to/backup/manifest.json /media/storage/neuroalign/data/
```

### Option 3: Check for Recovery Files

Some file systems keep deleted files in a trash/recycle bin:

```bash
# Check user trash
ls ~/.local/share/Trash/files/

# Or use recovery tools (if filesystem supports)
# This depends on your filesystem and how recently the files were deleted
```

### Option 4: Partial Recovery (If Cache Still Has Data)

If you have the cache still:

```bash
# Check cache status
neuroalign-export-cache /media/storage/neuroalign/data --status

# The cache has your 99 most recent sessions
# You would need to reload the missing >900 sessions from source
```

## Preventing Future Data Loss

The bug is now fixed, but here are additional safety measures:

### 1. Always Use `--status` First

Before exporting, check what's in the cache:

```bash
neuroalign-export-cache /media/storage/neuroalign/data --status
```

### 2. Backup Before Major Operations

Before running export or force reload:

```bash
# Create timestamped backup
cp -r /media/storage/neuroalign/data \
      /media/storage/neuroalign/data.backup.$(date +%Y%m%d_%H%M%S)
```

### 3. Use Incremental Loading

The pipeline now properly tracks which sessions are loaded:

```bash
# First run: loads all sessions
neuroalign-preprocess

# If interrupted, just rerun - it will skip completed sessions
neuroalign-preprocess

# No need to manually export cache - the pipeline does it automatically
```

## What NOT to Do

❌ **Don't manually export cache** unless you know it's safe
❌ **Don't use `--force` unless you want to reload everything**
❌ **Don't delete cache until export completes successfully**

## Recommended Workflow

```bash
# 1. Start loading
neuroalign-preprocess

# 2. If interrupted, check what's loaded
neuroalign-export-cache /media/storage/neuroalign/data --status

# 3. Resume loading (will skip completed sessions)
neuroalign-preprocess

# 4. Cache is automatically exported and cleared at the end
# No manual intervention needed!
```

## Getting Help

If you need help recovering your data:

1. **Check if source data is available** - Can you reload from CAT12/QSIRecon?
2. **Check for backups** - Do you have any backups of the Parquet files?
3. **File an issue** with details about your situation

## Summary

- ✅ **Bug is fixed** - Future exports will merge, not overwrite
- ⚠️ **Lost data** - >900 sessions were overwritten
- 🔄 **Recovery** - Reload from source if possible
- 🛡️ **Prevention** - Backups + incremental loading workflow
