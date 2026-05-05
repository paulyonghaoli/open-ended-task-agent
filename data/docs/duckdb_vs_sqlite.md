# DuckDB vs SQLite for Local Analytics

## Overview

DuckDB and SQLite are both embedded, serverless SQL engines that run inside the application process with no network round-trips and no external server daemon. Both store data in a single file and are trivially deployable. Despite sharing these properties, their execution engines reflect opposing design philosophies: SQLite optimizes for row-oriented transactional workloads (OLTP), while DuckDB was designed from first principles for column-oriented analytical workloads (OLAP).

## SQLite

SQLite is the most widely deployed database engine in existence, embedded in virtually every operating system, mobile device, browser, and IoT platform. Its row-oriented B-tree storage layout makes point reads extremely fast. Key characteristics:

- **Extreme portability**: a single `.db` file with no external runtime dependencies, under 600 KB compiled
- **Full ACID transactions**: write-ahead logging (WAL mode) enables concurrent reads with a single writer
- **Universal ecosystem**: bindings exist in C, Python, Go, Rust, JavaScript, Java, and more
- **Ideal workload**: structured logging, settings/metadata storage, CRUD applications, and mobile apps
- **Concurrency**: multiple reader threads are supported; write serialization is enforced per database

SQLite's row layout means that aggregation queries (SUM, AVG, GROUP BY) over large tables must deserialize entire rows even when only one or two columns are needed. On datasets beyond ~500k rows, this becomes a measurable bottleneck.

## DuckDB

DuckDB uses a columnar storage format (similar to Parquet) and a vectorized query engine that processes data in batches using SIMD instructions. This is the same fundamental architecture used by commercial OLAP systems (Redshift, BigQuery, Snowflake). Key characteristics:

- **Vectorized execution**: processes 1,024 rows per batch using 256-bit SIMD, yielding 10–100× speedups over row-by-row iteration
- **Columnar compression**: repeated values in a column compress far better than mixed-type rows; typical 3–8× space reduction on analytic datasets
- **Zero-copy Arrow/Pandas integration**: DuckDB can query Pandas DataFrames, Polars DataFrames, and Apache Arrow tables directly without copying data into the engine
- **Advanced SQL**: full window functions, CTEs, ASOF joins, PIVOT/UNPIVOT, QUALIFY, LIST aggregate, and JSON operators
- **Parallel query execution**: multi-core-aware; automatically parallelizes scans and aggregations across available CPU cores
- **CSV, Parquet, JSON reading**: can query files on disk without importing them first (`SELECT * FROM 'data.parquet'`)

DuckDB's write path (single-row INSERTs) is slower than SQLite because columnar reorganization is more expensive. High-frequency write workloads (e.g., event logging at 100k writes/sec) favor SQLite.

## Performance Reference Points

The following are representative benchmarks on a 5-million-row dataset (integer + float + string columns):

| Query Type | SQLite | DuckDB | Speedup |
|---|---|---|---|
| Full scan + SUM aggregation | ~4.2 s | ~0.09 s | ~47× |
| GROUP BY 10 categories | ~3.8 s | ~0.12 s | ~32× |
| JOIN two 1M-row tables | ~12.1 s | ~0.45 s | ~27× |
| Point lookup by primary key | ~0.3 ms | ~0.6 ms | 0.5× |
| Bulk INSERT (10k rows) | ~0.04 s | ~0.11 s | 0.4× |
| CSV import (1M rows) | ~2.1 s | ~0.22 s | ~10× |

Benchmarks are indicative. Results vary by dataset shape and hardware.

## Ecosystem and Integration

SQLite integrates into Python via the built-in `sqlite3` module. DuckDB requires `pip install duckdb` but offers tighter integration with the modern data stack:

```python
import duckdb
# Query a Pandas DataFrame without copying
result = duckdb.sql("SELECT category, AVG(value) FROM df GROUP BY 1").df()

# Read a Parquet file on disk without importing
result = duckdb.sql("SELECT * FROM 'data/*.parquet' WHERE year = 2024").df()
```

## Decision Criteria

**Choose DuckDB when:**
- Queries involve aggregations, GROUP BY, window functions, or full table scans
- Dataset size exceeds ~100k rows and query latency matters
- Working with Pandas, Polars, Arrow, or Parquet in a notebook or data pipeline
- Building a local BI tool, exploration environment, or analytics dashboard
- Columnar compression is a space constraint

**Choose SQLite when:**
- Workload is primarily transactional: INSERT, UPDATE, DELETE on individual rows
- Portability is paramount (embedded systems, mobile, browser via WebAssembly)
- Ecosystem breadth matters (every language has a mature SQLite driver)
- Concurrent write throughput is low but must be correct and durable
- You want zero additional dependencies in a shipped binary

## Recommendation

For a **local analytics application** that reads and aggregates data, DuckDB is the stronger choice. Its vectorized columnar engine is purpose-built for the exact workload—full scans, aggregations, and joins over medium-to-large local datasets. SQLite remains the best choice for applications that combine occasional analytics with frequent transactional writes, or where deployment portability is the dominant constraint.
