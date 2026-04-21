from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planner.common.config import load_yaml
from planner.common.io import ensure_dir
from planner.preprocess.io_utils import load_jsonl, save_jsonl

INTERACTIVE_SCENARIO_TYPES = {
    'unprotected_left_turn',
    'unprotected_right_turn',
    'left_turn',
    'right_turn',
    'crosswalk',
    'high_lateral_acceleration',
    'lane_change',
    'merge',
    'near_multiple_vehicles',
    'starting_left_turn',
    'starting_right_turn',
    'traversing_intersection',
    'waiting_for_pedestrian_to_cross',
}


def _step_from_seconds(seconds: float, db_interval: float) -> int:
    return max(1, int(round(float(seconds) / float(db_interval))))


def build_prefix_records(
    scene_records: list[dict],
    history_horizon_s: float,
    future_horizon_s: float,
    prefix_stride_s: float,
    interactive_stride_s: float,
) -> tuple[list[dict], list[dict]]:
    train_records: list[dict] = []
    val_records: list[dict] = []

    for scene in scene_records:
        db_interval = float(scene['database_interval'])
        num_iterations = int(scene['num_iterations'])
        history_frames = _step_from_seconds(history_horizon_s, db_interval) + 1
        future_frames = _step_from_seconds(future_horizon_s, db_interval)
        start_iter = history_frames - 1
        last_valid_iter = num_iterations - future_frames - 1
        if last_valid_iter < start_iter:
            continue

        scenario_type = str(scene.get('scenario_type', 'unknown')).lower()
        is_interactive_candidate = scenario_type in INTERACTIVE_SCENARIO_TYPES
        stride_frames = _step_from_seconds(
            interactive_stride_s if is_interactive_candidate else prefix_stride_s,
            db_interval,
        )

        out_list = train_records if scene['split'] == 'train' else val_records
        for iteration_index in range(start_iter, last_valid_iter + 1, stride_frames):
            out_list.append(
                {
                    'split': scene['split'],
                    'split_dir': scene['split_dir'],
                    'db_path': scene['db_path'],
                    'scenario_token': scene['scenario_token'],
                    'scenario_type': scene.get('scenario_type', 'unknown'),
                    'log_name': scene.get('log_name', ''),
                    'map_name': scene.get('map_name', 'unknown'),
                    'database_interval': db_interval,
                    'iteration_index': int(iteration_index),
                    'history_frames': int(history_frames),
                    'future_frames': int(future_frames),
                    'is_interactive_candidate': bool(is_interactive_candidate),
                }
            )
    return train_records, val_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--scene-manifest', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--history-horizon-s', type=float, default=None)
    parser.add_argument('--future-horizon-s', type=float, default=None)
    parser.add_argument('--prefix-stride-s', type=float, default=None)
    parser.add_argument('--interactive-stride-s', type=float, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else {}
    planner_cfg = config.get('planner', {})
    training_cfg = config.get('training', {})

    history_horizon_s = float(args.history_horizon_s if args.history_horizon_s is not None else planner_cfg.get('history_horizon_s', 2.0))
    future_horizon_s = float(args.future_horizon_s if args.future_horizon_s is not None else planner_cfg.get('output_horizon_s', 8.0))
    prefix_stride_s = float(args.prefix_stride_s if args.prefix_stride_s is not None else training_cfg.get('prefix_stride_s', 0.5))
    interactive_stride_s = float(args.interactive_stride_s if args.interactive_stride_s is not None else training_cfg.get('interactive_stride_s', 0.2))

    scene_records = load_jsonl(args.scene_manifest)
    train_records, val_records = build_prefix_records(
        scene_records=scene_records,
        history_horizon_s=history_horizon_s,
        future_horizon_s=future_horizon_s,
        prefix_stride_s=prefix_stride_s,
        interactive_stride_s=interactive_stride_s,
    )

    output_root = ensure_dir(args.output_root)
    save_jsonl(train_records, output_root / 'prefix_index_train.jsonl')
    save_jsonl(val_records, output_root / 'prefix_index_val.jsonl')
    print(f'[INFO] wrote {len(train_records)} train prefixes', flush=True)
    print(f'[INFO] wrote {len(val_records)} val prefixes', flush=True)


if __name__ == '__main__':
    main()
