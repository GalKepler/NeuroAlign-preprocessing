# Low Memory Configuration Guide

If you're experiencing memory issues (OOM crashes, swap thrashing) during data loading or cache export, use these optimizations.

## Symptoms

- Process crashes with "Killed" or "Out of Memory"
- System becomes unresponsive (swap usage at 100%)
- Export or loading hangs for extended periods

## Solutions

### 1. Reduce Chunk Size for Export

The cache export now uses chunked reading to minimize memory usage:

```bash
# Default (5000 rows per chunk)
neuroalign-export-cache /path/to/data

# For very low memory (1000 rows per chunk - slower but safer)
neuroalign-export-cache /path/to/data --chunk-size 1000

# For high memory systems (increase for speed)
neuroalign-export-cache /path/to/data --chunk-size 20000
```

**Tradeoff:** Lower chunk size = less memory but slower export

### 2. Install DuckDB for Large Files

DuckDB provides memory-efficient deduplication for files >100k rows:

```bash
pip install duckdb>=0.9.0

# Or install with optional dependencies
pip install -e ".[memory-efficient]"
```

Without DuckDB, large files fall back to pandas (uses more memory).

### 3. Reduce Parallel Workers

Lower the number of parallel workers to reduce concurrent memory usage:

```bash
# In your .env or command line
export N_JOBS=1  # Serial processing (slowest, least memory)
export N_JOBS=2  # 2 workers (moderate)
export N_JOBS=4  # 4 workers (default)
```

Edit `.env`:
```bash
N_JOBS=1
```

### 4. Monitor Memory Usage

Before starting, check available memory:

```bash
# Linux
free -h

# macOS
vm_stat | head -n 10
```

**Rule of thumb:**
- For 8GB RAM: Use `--chunk-size 1000` and `N_JOBS=1`
- For 16GB RAM: Use `--chunk-size 5000` and `N_JOBS=2`
- For 32GB+ RAM: Use defaults

### 5. Process in Batches

If you have limited memory, process your data in batches:

```bash
# Create subset CSVs
head -n 100 sessions.csv > batch1.csv
tail -n +101 sessions.csv | head -n 100 > batch2.csv

# Process each batch
neuroalign-preprocess --sessions-csv batch1.csv
neuroalign-preprocess --sessions-csv batch2.csv  # Merges automatically

# Export at the end
neuroalign-export-cache /path/to/data --chunk-size 1000
```

### 6. Clear Swap and Cache (Linux)

Before large operations:

```bash
# Clear page cache (safe, doesn't lose data)
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# Disable swap temporarily if you have enough RAM
sudo swapoff -a
# Re-enable after: sudo swapon -a
```

### 7. Increase Swap Space (Linux)

If you have disk space but limited RAM:

```bash
# Create 16GB swap file
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
free -h
```

**Warning:** Swap is slow - better to reduce memory usage than rely on swap.

## Memory Optimization Features

The pipeline now includes several memory optimizations:

### ✅ Chunked Cache Export
- Reads SQLite cache in small chunks
- Writes to Parquet incrementally
- Configurable chunk size

### ✅ Streaming Merge
- Doesn't load full existing Parquet into memory
- Uses PyArrow for efficient file operations
- Falls back to DuckDB for large files

### ✅ Incremental Saving
- Sessions saved to cache as they're loaded
- Crash recovery without memory spike
- Per-modality tracking avoids reloading

### ✅ Memory-Efficient Deduplication
- Small files (<100k rows): pandas
- Large files: DuckDB (streaming SQL)
- No full-table loads

## Example: Low Memory Workflow

```bash
# 1. Check memory
free -h

# 2. Set conservative settings
export N_JOBS=1

# 3. Start loading (will save to cache incrementally)
neuroalign-preprocess

# 4. If interrupted, export cache with minimal memory
neuroalign-export-cache /media/storage/neuroalign/data \
  --chunk-size 1000 \
  --verbose

# 5. Resume loading (will skip completed sessions)
neuroalign-preprocess
```

## Troubleshooting

### "Database is locked" errors
- Reduce N_JOBS to decrease concurrent writes
- SQLite WAL mode is enabled, but high concurrency can still cause locks

### Export hangs indefinitely
- Check disk space: `df -h`
- Check if swap is full: `free -h`
- Try smaller chunk size: `--chunk-size 500`

### "Cannot allocate memory" during merge
- Install DuckDB: `pip install duckdb`
- Use smaller chunks: `--chunk-size 1000`
- Process fewer sessions at once

### Process "Killed" without error
- OOM killer is terminating the process
- Reduce memory usage with above steps
- Check kernel logs: `dmesg | grep -i oom`

## Recommended Hardware

**Minimum:**
- RAM: 8GB (with optimizations)
- Disk: SSD recommended for swap
- CPU: 2+ cores

**Recommended:**
- RAM: 16GB+
- Disk: NVMe SSD
- CPU: 4+ cores

**Optimal:**
- RAM: 32GB+
- Disk: NVMe SSD
- CPU: 8+ cores

## Getting Help

If you're still experiencing memory issues after trying these steps:

1. Check cache size: `neuroalign-export-cache /path/to/data --status`
2. Check Parquet sizes: `du -sh /path/to/data/long/*`
3. Report issue with:
   - RAM available (`free -h`)
   - Cache size
   - Chunk size used
   - Error messages

## Summary

**Quick fixes for memory issues:**

```bash
# Minimal memory export
neuroalign-export-cache /path/to/data --chunk-size 1000 --verbose

# Minimal memory loading
export N_JOBS=1
neuroalign-preprocess

# Install for large datasets
pip install duckdb
```

**Prevention:**
- Use incremental loading (don't use `--force` unless needed)
- Export cache regularly during long runs
- Monitor memory with `htop` or `free -h`
