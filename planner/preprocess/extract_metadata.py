from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planner.common.config import load_yaml
from planner.common.io import ensure_dir, save_json
from planner.common.nuplan_compat import NuPlanScenarioBuilder, ScenarioFilter
from planner.preprocess.io_utils import save_jsonl

try:
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
except Exception:  # pragma: no cover
    Sequential = None


def discover_db_files(data_root: str | Path, split_dirs: Sequence[str] | None = None) -> List[str]:
    root = Path(data_root)
    files: List[str] = []
    if split_dirs:
        for split_dir in split_dirs:
            files.extend([str(p.resolve()) for p in sorted((root / split_dir).rglob('*.db'))])
    else:
        files.extend([str(p.resolve()) for p in sorted(root.rglob('*.db'))])
    return files


def infer_map_version(maps_root: str | Path) -> str:
    root = Path(maps_root)
    json_files = sorted(root.glob('nuplan-maps-v*.json'))
    if json_files:
        return json_files[0].stem
    return 'nuplan-maps-v1.0'


def build_scenarios_for_db(
    data_root: str,
    maps_root: str,
    db_file: str,
    map_version: str,
    max_scenarios: int | None = None,
):
    if NuPlanScenarioBuilder is None or Sequential is None:
        raise RuntimeError('nuPlan devkit is required to extract metadata. Please install nuplan-devkit first.')

    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=maps_root,
        sensor_root=data_root,
        db_files=[db_file],
        map_version=map_version,
        include_cameras=False,
        max_workers=None,
        verbose=False,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=max_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=1.0,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        ego_route_radius=30.0,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def infer_split_name(db_path: str | Path, split_dirs: Sequence[str]) -> str:
    parts = Path(db_path).parts
    for split_dir in split_dirs:
        if split_dir in parts:
            return split_dir
    raise ValueError(f'Could not infer split dir for {db_path}')


def extract_records_for_db(
    db_path: str,
    split: str,
    split_dir: str,
    scenarios: Sequence[Any],
) -> tuple[dict, list[dict]]:
    db_record = {
        'db_path': str(Path(db_path).resolve()),
        'split': split,
        'split_dir': split_dir,
        'num_scenarios': len(scenarios),
    }
    scene_records: list[dict] = []
    for scenario in scenarios:
        db_interval = float(getattr(scenario, 'database_interval', 0.1))
        num_iterations = int(scenario.get_number_of_iterations())
        scene_records.append(
            {
                'db_path': str(Path(db_path).resolve()),
                'split': split,
                'split_dir': split_dir,
                'scenario_token': str(scenario.token),
                'scenario_type': str(getattr(scenario, 'scenario_type', 'unknown')),
                'log_name': str(getattr(scenario, 'log_name', Path(db_path).stem)),
                'map_name': str(getattr(scenario, 'map_name', 'unknown')),
                'database_interval': db_interval,
                'num_iterations': num_iterations,
                'first_iteration': 0,
                'last_iteration': max(0, num_iterations - 1),
            }
        )
    return db_record, scene_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--maps-root', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--map-version', type=str, default=None)
    parser.add_argument('--train-splits', nargs='*', default=['train_boston', 'train_vegas_1', 'train_pittsburgh', 'train_singapore'])
    parser.add_argument('--val-splits', nargs='*', default=['val'])
    parser.add_argument('--max-train-scenarios', type=int, default=None)
    parser.add_argument('--max-val-scenarios', type=int, default=None)
    args = parser.parse_args()

    if args.config:
        _ = load_yaml(args.config)

    output_root = ensure_dir(args.output_root)
    map_version = args.map_version or infer_map_version(args.maps_root)
    all_split_dirs = list(args.train_splits) + list(args.val_splits)

    train_db_files = discover_db_files(args.data_root, args.train_splits)
    val_db_files = discover_db_files(args.data_root, args.val_splits)

    db_records: list[dict] = []
    scene_records: list[dict] = []

    for split_name, db_files, max_scenarios in (
        ('train', train_db_files, args.max_train_scenarios),
        ('val', val_db_files, args.max_val_scenarios),
    ):
        for db_path in db_files:
            split_dir = infer_split_name(db_path, all_split_dirs)
            scenarios = build_scenarios_for_db(
                data_root=args.data_root,
                maps_root=args.maps_root,
                db_file=db_path,
                map_version=map_version,
                max_scenarios=max_scenarios,
            )
            db_record, per_db_scene_records = extract_records_for_db(
                db_path=db_path,
                split=split_name,
                split_dir=split_dir,
                scenarios=scenarios,
            )
            db_records.append(db_record)
            scene_records.extend(per_db_scene_records)

    save_jsonl(db_records, output_root / 'db_manifest.jsonl')
    save_jsonl(scene_records, output_root / 'scene_manifest.jsonl')
    save_json(
        {
            'data_root': str(Path(args.data_root).resolve()),
            'maps_root': str(Path(args.maps_root).resolve()),
            'map_version': map_version,
            'train_splits': list(args.train_splits),
            'val_splits': list(args.val_splits),
            'num_db_files': len(db_records),
            'num_scenes': len(scene_records),
        },
        output_root / 'schema.json',
    )
    print(f'[INFO] wrote {len(db_records)} db records to {output_root / "db_manifest.jsonl"}', flush=True)
    print(f'[INFO] wrote {len(scene_records)} scene records to {output_root / "scene_manifest.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
