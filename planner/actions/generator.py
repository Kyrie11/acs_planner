from __future__ import annotations

from typing import List

import numpy as np

from planner.actions.action_types import ActionCandidate, ActionToken, RefinementDomain
from planner.actions.conservative_actions import is_conservative
from planner.actions.path_templates import PathTemplateGenerator
from planner.actions.speed_profiles import SpeedProfileGenerator
from planner.common.geometry import cumulative_arc_length, resample_trajectory
from planner.runtime.types import RuntimeContext


class ActionLibraryGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.path_generator = PathTemplateGenerator(config)
        self.speed_generator = SpeedProfileGenerator(config)
        self.output_horizon_s = float(config["planner"]["output_horizon_s"])
        self.output_dt_s = float(config["planner"]["output_dt_s"])

    def generate(self, ctx: RuntimeContext) -> List[ActionCandidate]:
        path_modes = self._candidate_path_modes(ctx)
        speed_modes = self._candidate_speed_modes(ctx)
        actions: List[ActionCandidate] = []
        for path_mode in path_modes:
            nominal_path, path_meta = self.path_generator.generate(ctx, path_mode)
            for speed_mode in speed_modes:
                nominal_speed = self.speed_generator.generate(ctx, speed_mode, path_mode)
                nominal_traj = self._combine_path_and_speed(nominal_path, nominal_speed.speed_profile)
                action_id = f"{path_mode}__{speed_mode}"
                candidate = ActionCandidate(
                    action_id=action_id,
                    token=ActionToken(path_mode=path_mode, speed_mode=speed_mode),
                    nominal_path=nominal_path,
                    nominal_speed_profile=nominal_speed.speed_profile,
                    nominal_traj=nominal_traj,
                    refine_domain=self._build_refine_domain(path_mode, speed_mode),
                    is_conservative=is_conservative(path_mode, speed_mode),
                    metadata={**path_meta, **nominal_speed.metadata, "stop_point_s": nominal_speed.stop_point_s},
                )
                if self._valid_nominal(candidate):
                    actions.append(candidate)
        return actions

    def _candidate_path_modes(self, ctx: RuntimeContext) -> List[str]:
        modes = ["KEEP_ROUTE"]
        modes.extend(f"BRANCH_{i}" for i in range(min(4, len(ctx.route_info.route_branches))))
        if self.config["actions"].get("enable_lane_change", True):
            modes.extend(["LC_LEFT", "LC_RIGHT"])
        if self.config["actions"].get("enable_merge", True):
            modes.extend(["MERGE_LEFT", "MERGE_RIGHT"])
        if self.config["actions"].get("enable_nudge", True):
            modes.extend(["NUDGE_LEFT", "NUDGE_RIGHT"])
        return modes

    def _candidate_speed_modes(self, ctx: RuntimeContext) -> List[str]:
        modes = ["FOLLOW", "CRUISE", "DECEL", "STOP", "CREEP"]
        return modes

    def _build_refine_domain(self, path_mode: str, speed_mode: str) -> RefinementDomain:
        terminal_speed_deltas = list(self.config["actions"]["terminal_speed_deltas_mps"])
        if path_mode.startswith("LC") or path_mode.startswith("MERGE"):
            return RefinementDomain(
                lc_start_delay=[0.0, 0.5, 1.0],
                lc_duration=list(self.config["actions"]["lane_change_duration_s"]),
                delta_v_terminal=terminal_speed_deltas,
            )
        if path_mode.startswith("NUDGE") or speed_mode == "CREEP":
            return RefinementDomain(
                lateral_offset=list(self.config["actions"]["nudge_offsets_m"]),
                creep_distance=list(self.config["actions"]["creep_distances_m"]),
                creep_speed_cap=[0.5, 1.0, 1.5],
            )
        return RefinementDomain(
            delta_v_terminal=terminal_speed_deltas,
            delta_stop_offset=[-3.0, -1.0, 0.0, 1.0],
            delta_time_shift=[-0.5, 0.0, 0.5],
        )

    def _combine_path_and_speed(self, path_xy: np.ndarray, speed_profile: np.ndarray):
        if len(path_xy) == 0:
            return []
        arc = cumulative_arc_length(path_xy)
        distance = np.cumsum(speed_profile) * self.output_dt_s
        distance = np.clip(distance, 0.0, arc[-1] if len(arc) else 0.0)
        interp_x = np.interp(distance, arc, path_xy[:, 0])
        interp_y = np.interp(distance, arc, path_xy[:, 1])
        xy = np.stack([interp_x, interp_y], axis=-1)
        return resample_trajectory(xy, dt=self.output_dt_s, horizon_s=self.output_horizon_s)

    def _valid_nominal(self, action: ActionCandidate) -> bool:
        if len(action.nominal_traj) < 2:
            return False
        curvature = np.array([abs(p.curvature) for p in action.nominal_traj], dtype=np.float64)
        lateral_acc = curvature * (np.array([p.speed for p in action.nominal_traj], dtype=np.float64) ** 2)
        if float(np.max(lateral_acc)) > 5.5:
            return False
        return True
