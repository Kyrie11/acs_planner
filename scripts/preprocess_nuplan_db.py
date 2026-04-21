from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from planner.common.config import load_yaml
from planner.common.io import ensure_dir, save_pickle
from planner.common.nuplan_compat import PlannerInitialization, PlannerInput, NuPlanScenarioBuilder, ScenarioFilter
from planner.runtime.context_builder import RuntimeContextBuilder
from planner.teacher.cache_writer import CacheWriter
from planner.teacher.teacher_runner import TeacherRunner
from planner.training.feature_utils import build_scene_action_atom_tensors

try:
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
except Exception as exc:  # pragma: no cover
    Sequential = None


@dataclass
class SimpleIteration:
    index: int


@dataclass
class SimpleHistoryBuffer:
    ego_states: List[Any]
    observations: List[Any]
    scenario: Any


class DatasetPreprocessor:
    def __init__(self, config: dict, output_root: str | Path):
        self.config = config
        self.output_root = Path(output_root)
        self.context_builder = RuntimeContextBuilder(config)
        self.teacher = TeacherRunner(config)
        self.writer = CacheWriter(self.output_root)

    def process_scenarios(self, scenarios: Sequence[Any], split: str, max_prefixes_per_scenario: int | None = None) -> None:
        records: List[dict] = []
        prefix_stride_s = float(self.config["training"]["prefix_stride_s"])
        interactive_stride_s = float(self.config["training"]["interactive_stride_s"])
        db_interval = float(getattr(scenarios[0], "database_interval", 0.1)) if scenarios else 0.1
        default_step = max(1, int(round(prefix_stride_s / db_interval)))
        interactive_step = max(1, int(round(interactive_stride_s / db_interval)))

        for scenario in tqdm(scenarios, desc=f"preprocess[{split}]"):
            num_iter = int(scenario.get_number_of_iterations())
            history_horizon_s = float(self.config["planner"]["history_horizon_s"])
            hist_frames = max(1, int(round(history_horizon_s / db_interval)) + 1)
            start_iter = hist_frames - 1
            current_iter = start_iter
            written = 0
            while current_iter < num_iter:
                planner_input, initialization = self._build_pseudo_planner_input(scenario, current_iter, hist_frames)
                try:
                    ctx = self.context_builder.build(planner_input, initialization)
                    teacher_results = self.teacher.evaluate(ctx)
                except Exception as exc:
                    current_iter += default_step
                    continue
                if not teacher_results:
                    current_iter += default_step
                    continue
                winner = teacher_results[0]
                margin = teacher_results[1].J - winner.J if len(teacher_results) > 1 else 999.0
                interaction_active = len(ctx.agents_interaction) > 0 and any(a.anchor_type in {"conflict", "merge", "PED_CROSS", "stop", "ONCOMING_TURN", "YIELD_ZONE"} for a in winner.support.anchors)
                for action_result in teacher_results:
                    support = action_result.support
                    for atom_idx, atom in enumerate(support.atoms):
                        tensors = build_scene_action_atom_tensors(ctx, action_result.action, atom, support, self.config)
                        sample_id = f"{scenario.token}_{current_iter}_{action_result.action.action_id}_{atom_idx}"
                        meta = {
                            "scenario_token": scenario.token,
                            "scenario_type": getattr(scenario, "scenario_type", "unknown"),
                            "iteration_index": current_iter,
                            "action_id": action_result.action.action_id,
                            "atom_id": atom.atom_id,
                            "rho_target": float(action_result.rho[atom_idx]),
                            "damage_target": float(action_result.omission_damage[atom_idx]),
                            "mu_target": float(action_result.mu[atom_idx]),
                            "sample_weight": float(1.0 + self.config["training"]["rare_reweight_lambda"] * (action_result.omission_damage[atom_idx] > self.config["training"]["rare_damage_threshold"])),
                            "interactive": bool(interaction_active and margin < self.config["training"]["interactive_margin_threshold"]),
                            "margin": float(margin),
                            "residual_bucket": 0,
                        }
                        path = self.writer.write_sample(split, sample_id, tensors, meta)
                        records.append({"file": path.name, "path": str(path), "scenario_token": scenario.token, "split": split})
                written += 1
                if max_prefixes_per_scenario is not None and written >= max_prefixes_per_scenario:
                    break
                current_iter += interactive_step if interaction_active and margin < self.config["training"]["interactive_margin_threshold"] else default_step
        self.writer.write_index(split, records)

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


def discover_db_files(data_root: str | Path, split_dirs: Sequence[str] | None = None) -> List[str]:
    root = Path(data_root)
    files: List[str] = []
    if split_dirs:
        for split_dir in split_dirs:
            files.extend([str(p) for p in sorted((root / split_dir).rglob("*.db"))])
    else:
        files.extend([str(p) for p in sorted(root.rglob("*.db"))])
    return files


def infer_map_version(maps_root: str | Path) -> str:
    root = Path(maps_root)
    json_files = sorted(root.glob("nuplan-maps-v*.json"))
    if json_files:
        return json_files[0].stem
    return "nuplan-maps-v1.0"


def build_scenarios(data_root: str, maps_root: str, db_files: List[str], map_version: str, max_scenarios: int | None = None):
    if NuPlanScenarioBuilder is None or Sequential is None:
        raise RuntimeError("nuPlan devkit is required to preprocess .db files. Please install nuplan-devkit first.")

    print(f"[INFO] build_scenarios: data_root={data_root}", flush=True)
    print(f"[INFO] build_scenarios: maps_root={maps_root}", flush=True)
    print(f"[INFO] build_scenarios: map_version={map_version}", flush=True)
    print(f"[INFO] build_scenarios: num_db_files={len(db_files)}", flush=True)
    for p in db_files[:10]:
        print(f"  db_file={p}", flush=True)

    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=maps_root,
        sensor_root=data_root,
        db_files=db_files,
        map_version=map_version,
        include_cameras=False,
        max_workers=None,
        verbose=True,
    )

    print("[INFO] NuPlanScenarioBuilder created", flush=True)

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

    print("[INFO] ScenarioFilter created", flush=True)

    worker = Sequential()
    print("[INFO] worker created, calling get_scenarios...", flush=True)

    scenarios = builder.get_scenarios(scenario_filter, worker)

    print(f"[INFO] get_scenarios returned {len(scenarios)} scenarios", flush=True)
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--maps-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--map-version", type=str, default=None)
    parser.add_argument("--train-splits", nargs="*", default=["train_boston", "train_vegas_1", "train_pittsburgh", "train_singapore"])
    parser.add_argument("--val-splits", nargs="*", default=["val"])
    parser.add_argument("--max-train-scenarios", type=int, default=None)
    parser.add_argument("--max-val-scenarios", type=int, default=None)
    parser.add_argument("--max-prefixes-per-scenario", type=int, default=None)
    args = parser.parse_args()

    print("[INFO] loading config...", flush=True)
    config = load_yaml(args.config)

    print("[INFO] loading config...", flush=True)
    output_root = ensure_dir(args.output_root)

    map_version = args.map_version or infer_map_version(args.maps_root)
    print("[INFO] creating output root...", flush=True)

    print("[INFO] discovering train db files...", flush=True)
    train_db_files = discover_db_files(args.data_root, args.train_splits)
    print(f"[INFO] found {len(train_db_files)} train db files", flush=True)
    for p in train_db_files[:5]:
        print(f"  train_db: {p}", flush=True)

    print("[INFO] discovering val db files...", flush=True)
    val_db_files = discover_db_files(args.data_root, args.val_splits)
    print(f"[INFO] found {len(val_db_files)} val db files", flush=True)
    for p in val_db_files[:5]:
        print(f"  val_db: {p}", flush=True)

    print("[INFO] building train scenarios...", flush=True)
    train_scenarios = build_scenarios(args.data_root, args.maps_root, train_db_files, map_version, args.max_train_scenarios)
    print(f"[INFO] built {len(train_scenarios)} train scenarios", flush=True)

    print("[INFO] building val scenarios...", flush=True)
    val_scenarios = build_scenarios(args.data_root, args.maps_root, val_db_files, map_version, args.max_val_scenarios)
    print("[INFO] building val scenarios...", flush=True)

    print("[INFO] creating preprocessor...", flush=True)
    processor = DatasetPreprocessor(config, output_root)

    print("[INFO] start processing train scenarios...", flush=True)
    processor.process_scenarios(train_scenarios, split="train", max_prefixes_per_scenario=args.max_prefixes_per_scenario)

    print("[INFO] start processing val scenarios...", flush=True)
    processor.process_scenarios(val_scenarios, split="val", max_prefixes_per_scenario=args.max_prefixes_per_scenario)


if __name__ == "__main__":
    main()
