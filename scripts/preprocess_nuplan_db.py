from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tqdm import tqdm

from planner.common.config import load_yaml
from planner.common.io import ensure_dir
from planner.common.nuplan_compat import PlannerInitialization, PlannerInput, NuPlanScenarioBuilder, ScenarioFilter
from planner.preprocess.io_utils import load_jsonl
from planner.teacher.cache_writer import CacheWriter
from planner.teacher.teacher_runner import TeacherRunner, TeacherActionResult
from planner.runtime.context_builder import RuntimeContextBuilder
from planner.training.feature_utils import build_scene_action_atom_tensors

try:
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
except Exception:  # pragma: no cover
    Sequential = None


@dataclass
class SimpleIteration:
    index: int


@dataclass
class SimpleHistoryBuffer:
    ego_states: List[Any]
    observations: List[Any]
    scenario: Any


def infer_map_version(maps_root: str | Path) -> str:
    root = Path(maps_root)
    json_files = sorted(root.glob('nuplan-maps-v*.json'))
    if json_files:
        return json_files[0].stem
    return 'nuplan-maps-v1.0'


def build_scenarios(
    data_root: str,
    maps_root: str,
    db_files: list[str],
    map_version: str,
    scenario_tokens: list[str] | None = None,
):
    if NuPlanScenarioBuilder is None or Sequential is None:
        raise RuntimeError('nuPlan devkit is required to preprocess .db files. Please install nuplan-devkit first.')

    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=maps_root,
        sensor_root=data_root,
        db_files=db_files,
        map_version=map_version,
        include_cameras=False,
        max_workers=None,
        verbose=False,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=scenario_tokens,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=1.0,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        ego_route_radius=30.0,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


class DatasetPreprocessor:
    def __init__(self, config: dict, output_root: str | Path):
        self.config = config
        self.output_root = Path(output_root)
        self.context_builder = RuntimeContextBuilder(config)
        self.teacher = TeacherRunner(config)
        self.writer = CacheWriter(self.output_root)

    def process_prefixes(self, split: str, prefixes: Sequence[dict], data_root: str, maps_root: str, map_version: str) -> None:
        if not prefixes:
            self.writer.write_index(split, [])
            return

        grouped_prefixes: dict[str, list[dict]] = defaultdict(list)
        for prefix in prefixes:
            grouped_prefixes[str(Path(prefix['db_path']).resolve())].append(prefix)

        all_records: list[dict] = []
        for db_path, db_prefixes in tqdm(grouped_prefixes.items(), desc=f'preprocess[{split}]'):
            scenario_tokens = sorted({item['scenario_token'] for item in db_prefixes})
            scenarios = build_scenarios(
                data_root=data_root,
                maps_root=maps_root,
                db_files=[db_path],
                map_version=map_version,
                scenario_tokens=scenario_tokens,
            )
            scenario_by_token = {str(s.token): s for s in scenarios}
            db_prefixes = sorted(db_prefixes, key=lambda item: (item['scenario_token'], int(item['iteration_index'])))
            for prefix in db_prefixes:
                scenario_token = str(prefix['scenario_token'])
                scenario = scenario_by_token.get(scenario_token)
                if scenario is None:
                    print(f'[WARN] scenario token {scenario_token} not found in db {db_path}', flush=True)
                    continue
                record = self._process_single_prefix(split=split, scenario=scenario, prefix=prefix)
                if record is not None:
                    all_records.append(record)
        self.writer.write_index(split, all_records)

    def _process_single_prefix(self, split: str, scenario: Any, prefix: dict) -> dict | None:
        iteration = int(prefix['iteration_index'])
        hist_frames = int(prefix['history_frames'])
        planner_input, initialization = self._build_pseudo_planner_input(scenario, iteration, hist_frames)
        try:
            ctx = self.context_builder.build(planner_input, initialization)
            teacher_results = self.teacher.evaluate(ctx)
        except Exception as exc:
            print(
                f'[ERROR] split={split}, scenario={getattr(scenario, "token", "unknown")}, '
                f'iter={iteration}, exc={type(exc).__name__}: {exc}',
                flush=True,
            )
            return None
        if not teacher_results:
            return None

        winner = teacher_results[0]
        margin = teacher_results[1].J - winner.J if len(teacher_results) > 1 else 999.0
        interaction_active = (
            len(ctx.agents_interaction) > 0
            and any(a.anchor_type in {'conflict', 'merge', 'PED_CROSS', 'stop', 'ONCOMING_TURN', 'YIELD_ZONE'} for a in winner.support.anchors)
        )

        prefix_samples: list[dict] = []
        for action_result in teacher_results:
            support = action_result.support
            for atom_idx, atom in enumerate(support.atoms):
                tensors = build_scene_action_atom_tensors(ctx, action_result.action, atom, support, self.config)
                sample_id = f'{scenario.token}_{iteration}_{action_result.action.action_id}_{atom_idx}'
                meta = {
                    'sample_id': sample_id,
                    'scenario_token': str(scenario.token),
                    'scenario_type': str(getattr(scenario, 'scenario_type', 'unknown')),
                    'iteration_index': iteration,
                    'action_id': action_result.action.action_id,
                    'atom_id': atom.atom_id,
                    'rho_target': float(action_result.rho[atom_idx]),
                    'damage_target': float(action_result.omission_damage[atom_idx]),
                    'mu_target': float(action_result.mu[atom_idx]),
                    'sample_weight': float(
                        1.0
                        + self.config['training']['rare_reweight_lambda']
                        * (action_result.omission_damage[atom_idx] > self.config['training']['rare_damage_threshold'])
                    ),
                    'interactive': bool(
                        interaction_active
                        and margin < self.config['training']['interactive_margin_threshold']
                    ),
                    'margin': float(margin),
                    'residual_bucket': 0,
                    'split': split,
                    'split_dir': prefix['split_dir'],
                    'db_path': prefix['db_path'],
                    'is_interactive_candidate': bool(prefix.get('is_interactive_candidate', False)),
                }
                prefix_samples.append(
                    {
                        'sample_id': sample_id,
                        'tensors': tensors,
                        'meta': meta,
                    }
                )

        if not prefix_samples:
            return None

        batch_id = f'{scenario.token}_{iteration}'
        path = self.writer.write_batch(split, batch_id, prefix_samples)
        return {
            'file': path.name,
            'path': str(path),
            'scenario_token': str(scenario.token),
            'iteration_index': iteration,
            'num_samples': len(prefix_samples),
            'split': split,
            'split_dir': prefix['split_dir'],
            'db_path': prefix['db_path'],
        }

    def _build_pseudo_planner_input(self, scenario: Any, iteration: int, hist_frames: int):
        start_idx = max(0, iteration - hist_frames + 1)
        ego_states = [scenario.get_ego_state_at_iteration(i) for i in range(start_idx, iteration + 1)]
        observations = [scenario.get_tracked_objects_at_iteration(i) for i in range(start_idx, iteration + 1)]
        history = SimpleHistoryBuffer(ego_states=ego_states, observations=observations, scenario=scenario)
        tl = list(scenario.get_traffic_light_status_at_iteration(iteration))
        planner_input = PlannerInput(iteration=SimpleIteration(index=iteration), history=history, traffic_light_data=tl)
        mission_goal = scenario.get_mission_goal() or scenario.get_expert_goal_state()
        initialization = PlannerInitialization(
            route_roadblock_ids=list(scenario.get_route_roadblock_ids()),
            mission_goal=mission_goal,
            map_api=scenario.map_api,
        )
        return planner_input, initialization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--maps-root', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--map-version', type=str, default=None)
    parser.add_argument('--prefix-index-train', type=str, required=True)
    parser.add_argument('--prefix-index-val', type=str, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_root = ensure_dir(args.output_root)
    map_version = args.map_version or infer_map_version(args.maps_root)

    train_prefixes = load_jsonl(args.prefix_index_train)
    val_prefixes = load_jsonl(args.prefix_index_val)

    print(f'[INFO] loaded {len(train_prefixes)} train prefixes', flush=True)
    print(f'[INFO] loaded {len(val_prefixes)} val prefixes', flush=True)

    processor = DatasetPreprocessor(config, output_root)
    processor.process_prefixes(
        split='train',
        prefixes=train_prefixes,
        data_root=args.data_root,
        maps_root=args.maps_root,
        map_version=map_version,
    )
    processor.process_prefixes(
        split='val',
        prefixes=val_prefixes,
        data_root=args.data_root,
        maps_root=args.maps_root,
        map_version=map_version,
    )


if __name__ == '__main__':
    main()
