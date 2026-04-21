from __future__ import annotations

import sqlite3
from pathlib import Path


FAST_PRAGMAS = (
    'PRAGMA query_only = 1;',
    'PRAGMA journal_mode = OFF;',
    'PRAGMA synchronous = OFF;',
    'PRAGMA temp_store = MEMORY;',
    'PRAGMA foreign_keys = OFF;',
)


def connect_sqlite_ro(db_path: str | Path, mmap_mb: int = 512) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    for pragma in FAST_PRAGMAS:
        conn.execute(pragma)
    conn.execute('PRAGMA cache_size = -200000;')
    conn.execute(f'PRAGMA mmap_size = {int(mmap_mb) * 1024 * 1024};')
    return conn
