from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()[:16]


def write_pickle_gz(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle_gz(path: str | Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
