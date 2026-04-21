from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def sqlite_uri(path: str | Path, immutable: bool = True) -> str:
    p = Path(path).expanduser().resolve()
    if immutable:
        return f"file:{p}?mode=ro&immutable=1"
    return f"file:{p}?mode=ro"


def connect_readonly(path: str | Path, *, immutable: bool = True, cache_mb: int = 256) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_uri(path, immutable=immutable), uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute(f"PRAGMA cache_size = {-1024 * cache_mb}")
    conn.execute("PRAGMA mmap_size = 268435456")
    return conn


@contextmanager
def readonly_connection(path: str | Path, *, immutable: bool = True, cache_mb: int = 256) -> Iterator[sqlite3.Connection]:
    conn = connect_readonly(path, immutable=immutable, cache_mb=cache_mb)
    try:
        yield conn
    finally:
        conn.close()


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(row["name"]) for row in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row["name"]) for row in rows]


def candidate_timestamp_column(columns: Sequence[str]) -> str | None:
    for name in ("timestamp", "timestamp_us", "time_us", "utime", "token_timestamp"):
        if name in columns:
            return name
    return None


def candidate_token_column(columns: Sequence[str]) -> str | None:
    for name in ("token", "lidar_pc_token", "log_token", "track_token"):
        if name in columns:
            return name
    return None


def stream_query(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[object] | None = None,
    batch_size: int = 4096,
) -> Iterator[list[sqlite3.Row]]:
    cur = conn.execute(sql, tuple(params or ()))
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def batched_in_query(
    conn: sqlite3.Connection,
    table: str,
    key_column: str,
    keys: Sequence[object],
    columns: Sequence[str] | None = None,
    batch_size: int = 1024,
) -> Iterator[list[sqlite3.Row]]:
    cols = ", ".join(columns) if columns else "*"
    for i in range(0, len(keys), batch_size):
        chunk = keys[i : i + batch_size]
        placeholders = ", ".join(["?"] * len(chunk))
        sql = f"SELECT {cols} FROM {table} WHERE {key_column} IN ({placeholders})"
        yield conn.execute(sql, tuple(chunk)).fetchall()


def range_query(
    conn: sqlite3.Connection,
    table: str,
    ts_column: str,
    start_ts: int,
    end_ts: int,
    columns: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {table} WHERE {ts_column} >= ? AND {ts_column} <= ? ORDER BY {ts_column}"
    return conn.execute(sql, (start_ts, end_ts)).fetchall()
