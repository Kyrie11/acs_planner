from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from shapely.geometry import box

from acs_planner.preprocess.cache import stable_hash, write_pickle_gz
from acs_planner.preprocess.gpkg_reader import GPKGFeature, load_layer_features, list_layers
from acs_planner.preprocess.sqlite_reader import (
    candidate_timestamp_column,
    connect_readonly,
    list_tables,
    range_query,
    table_columns,
)


_RELEVANT_LAYERS = {
    "lane",
    "lane_connector",
    "roadblock",
    "roadblock_connector",
    "stop_line",
    "crosswalk",
    "intersection",
    "drivable_area",
}


@dataclass(slots=True)
class PrefixRecord:
    db_file: str
    table: str
    timestamp: int
    start_timestamp: int
    end_timestamp: int
    meta: dict[str, Any]


class NuPlanMetadataExtractor:
    """Fast, schema-tolerant metadata extractor for nuPlan-like SQLite DBs."""

    def __init__(self, db_file: str | Path) -> None:
        self.db_file = str(db_file)

    def discover_time_tables(self) -> list[tuple[str, str]]:
        with connect_readonly(self.db_file) as conn:
            out: list[tuple[str, str]] = []
            for table in list_tables(conn):
                columns = table_columns(conn, table)
                ts_col = candidate_timestamp_column(columns)
                if ts_col is not None:
                    out.append((table, ts_col))
            return out

    def build_prefix_index(
        self,
        stride_us: int = 500_000,
        history_horizon_us: int = 2_000_000,
        target_table: str | None = None,
    ) -> list[PrefixRecord]:
        with connect_readonly(self.db_file) as conn:
            candidates = self.discover_time_tables()
            if target_table is None:
                table, ts_col = self._pick_primary_time_table(candidates)
            else:
                table = target_table
                ts_col = candidate_timestamp_column(table_columns(conn, target_table))
                if ts_col is None:
                    raise ValueError(f"Table {target_table} has no timestamp-like column")
            rows = conn.execute(f"SELECT {ts_col} FROM {table} ORDER BY {ts_col}").fetchall()
            timestamps = [int(row[ts_col]) for row in rows]
            if not timestamps:
                return []
            out: list[PrefixRecord] = []
            next_keep = timestamps[0]
            for ts in timestamps:
                if ts < next_keep:
                    continue
                out.append(
                    PrefixRecord(
                        db_file=self.db_file,
                        table=table,
                        timestamp=ts,
                        start_timestamp=max(timestamps[0], ts - history_horizon_us),
                        end_timestamp=ts,
                        meta={"ts_column": ts_col},
                    )
                )
                next_keep = ts + stride_us
            return out

    @staticmethod
    def _pick_primary_time_table(candidates: Sequence[tuple[str, str]]) -> tuple[str, str]:
        preferred = ["lidar_pc", "ego_pose", "scene", "scenario"]
        for pref in preferred:
            for table, ts_col in candidates:
                if table == pref:
                    return table, ts_col
        if not candidates:
            raise ValueError("No time-indexed table found in database")
        return candidates[0]

    def extract_time_window(self, table: str, start_ts: int, end_ts: int, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        with connect_readonly(self.db_file) as conn:
            ts_col = candidate_timestamp_column(table_columns(conn, table))
            if ts_col is None:
                raise ValueError(f"Table {table} has no timestamp-like column")
            rows = range_query(conn, table, ts_col, start_ts, end_ts, columns=columns)
            return [dict(row) for row in rows]


class NuPlanMapMetadataExtractor:
    def __init__(self, gpkg_file: str | Path) -> None:
        self.gpkg_file = str(gpkg_file)

    def relevant_layers(self) -> list[str]:
        return [layer for layer in list_layers(self.gpkg_file) if layer.lower() in _RELEVANT_LAYERS]

    def extract(self, simplify_tolerance: float | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"gpkg_file": self.gpkg_file, "layers": {}}
        for layer in self.relevant_layers():
            features = load_layer_features(self.gpkg_file, layer)
            layer_records = []
            for feat in features:
                geom = feat.geometry.simplify(simplify_tolerance) if simplify_tolerance else feat.geometry
                layer_records.append(
                    {
                        "row_id": feat.row_id,
                        "bounds": geom.bounds,
                        "geom_type": geom.geom_type,
                        "attrs": {k: _jsonable_scalar(v) for k, v in feat.attrs.items()},
                    }
                )
            data["layers"][layer] = layer_records
        return data


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _process_db(db_file: str, output_root: str, stride_us: int, history_horizon_us: int) -> str:
    extractor = NuPlanMetadataExtractor(db_file)
    prefix_index = extractor.build_prefix_index(stride_us=stride_us, history_horizon_us=history_horizon_us)
    out_path = Path(output_root) / "prefix_index" / (Path(db_file).stem + ".pkl.gz")
    write_pickle_gz(out_path, prefix_index)
    return str(out_path)


def _process_map(gpkg_file: str, output_root: str, simplify_tolerance: float | None) -> str:
    extractor = NuPlanMapMetadataExtractor(gpkg_file)
    data = extractor.extract(simplify_tolerance=simplify_tolerance)
    out_path = Path(output_root) / "map_metadata" / (Path(gpkg_file).stem + ".pkl.gz")
    write_pickle_gz(out_path, data)
    return str(out_path)


def run_extraction(
    db_root: str | Path,
    map_root: str | Path,
    output_root: str | Path,
    *,
    workers: int = 4,
    stride_us: int = 500_000,
    history_horizon_us: int = 2_000_000,
    simplify_tolerance: float | None = None,
) -> dict[str, list[str]]:
    db_files = sorted(str(p) for p in Path(db_root).rglob("*.db"))
    gpkg_files = sorted(str(p) for p in Path(map_root).rglob("*.gpkg"))
    output_root = str(output_root)

    results: dict[str, list[str]] = {"db": [], "map": []}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for db_file in db_files:
            futures.append(("db", ex.submit(_process_db, db_file, output_root, stride_us, history_horizon_us)))
        for gpkg_file in gpkg_files:
            futures.append(("map", ex.submit(_process_map, gpkg_file, output_root, simplify_tolerance)))
        for kind, future in futures:
            results[kind].append(future.result())
    manifest = {
        "db_root": str(db_root),
        "map_root": str(map_root),
        "output_root": output_root,
        "stride_us": stride_us,
        "history_horizon_us": history_horizon_us,
        "simplify_tolerance": simplify_tolerance,
        "outputs": results,
        "config_hash": stable_hash(
            {
                "stride_us": stride_us,
                "history_horizon_us": history_horizon_us,
                "simplify_tolerance": simplify_tolerance,
            }
        ),
    }
    manifest_path = Path(output_root) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fast nuPlan metadata caches from .db and .gpkg files")
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stride-us", type=int, default=500_000)
    parser.add_argument("--history-horizon-us", type=int, default=2_000_000)
    parser.add_argument("--simplify-tolerance", type=float, default=None)
    args = parser.parse_args()
    run_extraction(
        db_root=args.db_root,
        map_root=args.map_root,
        output_root=args.output_root,
        workers=args.workers,
        stride_us=args.stride_us,
        history_horizon_us=args.history_horizon_us,
        simplify_tolerance=args.simplify_tolerance,
    )


if __name__ == "__main__":
    main()
