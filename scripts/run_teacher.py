from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from planner.common.config import load_yaml
from planner.common.io import ensure_dir, save_pickle
from planner.common.nuplan_compat import PlannerInitialization, PlannerInput, NuPlanScenarioBuilder, ScenarioFilter
from planner.runtime.context_builder import RuntimeContextBuilder
from planner.teacher.teacher_runner import TeacherRunner

try:
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
except Exception:
    Sequential = None


class SimpleIteration:
    def __init__(self, index: int):
        self.index = index


class SimpleHistoryBuffer:
    def __init__(self, ego_states, observations, scenario):
        self.ego_states = ego_states
        self.observations = observations
        self.scenario = scenario


def infer_map_version(maps_root: str) -> str:
    root = Path(maps_root)
    json_files = sorted(root.glob("nuplan-maps-v*.json"))
    return json_files[0].stem if json_files else "nuplan-maps-v1.0"


def discover_db_files(data_root: str, split_dirs: list[str]) -> list[str]:
    root = Path(data_root)
    out = []
    for split in split_dirs:
        out.extend(str(p) for p in sorted((root / split).rglob("*.db")))
    return out


def build_scenarios(data_root: str, maps_root: str, db_files: list[str], map_version: str, limit: int | None):
    if NuPlanScenarioBuilder is None or Sequential is None:
        raise RuntimeError("nuPlan devkit is required to run teacher cache generation.")
    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=maps_root,
        sensor_root=data_root,
        db_files=db_files,
        map_version=map_version,
        include_cameras=False,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=limit,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=1.0,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        ego_route_radius=30.0,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--maps-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="*", default=["val"])
    parser.add_argument("--max-scenarios", type=int, default=64)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    runner = TeacherRunner(cfg)
    ctx_builder = RuntimeContextBuilder(cfg)
    map_version = infer_map_version(args.maps_root)
    db_files = discover_db_files(args.data_root, args.splits)
    scenarios = build_scenarios(args.data_root, args.maps_root, db_files, map_version, args.max_scenarios)
    output_root = ensure_dir(args.output_root)

    all_records = []
    for scenario in tqdm(scenarios, desc="teacher"):
        db_interval = float(scenario.database_interval)
        hist_frames = max(1, int(round(cfg["planner"]["history_horizon_s"] / db_interval)) + 1)
        stride = max(1, int(round(cfg["training"]["prefix_stride_s"] / db_interval)))
        for iteration in range(hist_frames - 1, scenario.get_number_of_iterations(), stride):
            ego_states = [scenario.get_ego_state_at_iteration(i) for i in range(iteration - hist_frames + 1, iteration + 1)]
            observations = [scenario.get_tracked_objects_at_iteration(i) for i in range(iteration - hist_frames + 1, iteration + 1)]
            history = SimpleHistoryBuffer(ego_states=ego_states, observations=observations, scenario=scenario)
            planner_input = PlannerInput(iteration=SimpleIteration(iteration), history=history, traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(iteration)))
            initialization = PlannerInitialization(route_roadblock_ids=list(scenario.get_route_roadblock_ids()), mission_goal=scenario.get_mission_goal() or scenario.get_expert_goal_state(), map_api=scenario.map_api)
            try:
                ctx = ctx_builder.build(planner_input, initialization)
                results = runner.evaluate(ctx)
            except Exception:
                continue
            serializable = []
            for result in results:
                serializable.append({
                    "action_id": result.action.action_id,
                    "token": {"path_mode": result.action.token.path_mode, "speed_mode": result.action.token.speed_mode},
                    "refinement": result.action.refinement,
                    "J": result.J,
                    "rho": result.rho.tolist(),
                    "mu": result.mu.tolist(),
                    "omission_damage": result.omission_damage.tolist(),
                    "anchors": [asdict(a) for a in result.support.anchors],
                    "atoms": [{"atom_id": atom.atom_id, "assignments": {k: v.as_dict() for k, v in atom.assignments.items()}, "prior_logit": atom.prior_logit} for atom in result.support.atoms],
                })
            out_path = output_root / f"{scenario.token}_{iteration}.pkl"
            save_pickle(serializable, out_path)
            all_records.append({"scenario_token": scenario.token, "iteration": iteration, "file": out_path.name})
    save_pickle(all_records, output_root / "index.pkl")


if __name__ == "__main__":
    main()
