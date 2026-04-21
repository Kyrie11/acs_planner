from __future__ import annotations

import argparse
import time
from pathlib import Path

from acs_planner.preprocess.gpkg_reader import list_layers, load_layer_features
from acs_planner.preprocess.metadata_extractor import NuPlanMetadataExtractor
from acs_planner.preprocess.sqlite_reader import connect_readonly


def profile_db(db_file: str) -> dict[str, float]:
    extractor = NuPlanMetadataExtractor(db_file)
    t0 = time.perf_counter()
    candidates = extractor.discover_time_tables()
    t1 = time.perf_counter()
    index = extractor.build_prefix_index()
    t2 = time.perf_counter()
    return {
        "discover_time_tables_s": t1 - t0,
        "build_prefix_index_s": t2 - t1,
        "num_time_tables": float(len(candidates)),
        "num_prefixes": float(len(index)),
    }


def profile_map(gpkg_file: str) -> dict[str, float]:
    t0 = time.perf_counter()
    layers = list_layers(gpkg_file)
    t1 = time.perf_counter()
    num_features = 0
    for layer in layers[:3]:
        try:
            feats = load_layer_features(gpkg_file, layer, limit=512)
            num_features += len(feats)
        except Exception:
            continue
    t2 = time.perf_counter()
    return {
        "list_layers_s": t1 - t0,
        "load_sample_features_s": t2 - t1,
        "num_layers": float(len(layers)),
        "num_sample_features": float(num_features),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick profiler for nuPlan metadata extraction")
    parser.add_argument("--db-file", default=None)
    parser.add_argument("--gpkg-file", default=None)
    args = parser.parse_args()

    if args.db_file:
        stats = profile_db(args.db_file)
        print("[DB]")
        for k, v in stats.items():
            print(f"{k}: {v}")

    if args.gpkg_file:
        stats = profile_map(args.gpkg_file)
        print("[MAP]")
        for k, v in stats.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
