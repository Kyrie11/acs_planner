from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from planner.common.io import ensure_dir


def save_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    records: List[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
