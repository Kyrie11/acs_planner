from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from planner.common.geometry import interpolate_polyline, project_point_to_polyline, smooth_lateral_transition, frenet_to_cartesian
from planner.runtime.types import RuntimeContext


class PathTemplateGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.output_horizon_s = float(config["planner"]["output_horizon_s"])
        self.output_dt_s = float(config["planner"]["output_dt_s"])

    def generate(self, ctx: RuntimeContext, path_mode: str) -> tuple[np.ndarray, dict]:
        route_centerline = ctx.route_info.route_centerline
        if len(route_centerline) < 2:
            route_centerline = self._fallback_centerline(ctx)
        num = int(round(self.output_horizon_s / self.output_dt_s)) + 1
        base = interpolate_polyline(route_centerline, num=num)
        meta: Dict[str, float | str | int] = {}
        if path_mode == "KEEP_ROUTE":
            return base, meta
        if path_mode.startswith("BRANCH_"):
            branch_index = int(path_mode.split("_")[-1])
            if branch_index < len(ctx.route_info.route_branches):
                branch = ctx.route_info.route_branches[branch_index]
                return interpolate_polyline(branch.centerline, num=num), {"branch_id": branch.branch_id}
            return base, meta
        if path_mode in {"LC_LEFT", "LC_RIGHT", "MERGE_LEFT", "MERGE_RIGHT"}:
            sign = 1.0 if path_mode.endswith("LEFT") else -1.0
            duration = float(np.mean(self.config["actions"]["lane_change_duration_s"]))
            arc = np.linspace(0.0, max(20.0, ctx.route_info.goal_progress_s), num=num)
            offset = smooth_lateral_transition(arc, start_s=5.0, duration_s=duration * 8.0, target_offset=sign * 3.5)
            path = frenet_to_cartesian(base, np.linspace(0.0, max(20.0, ctx.route_info.goal_progress_s), num=num), offset)
            return path, {"lateral_target": sign * 3.5, "maneuver_duration": duration}
        if path_mode in {"NUDGE_LEFT", "NUDGE_RIGHT"}:
            sign = 1.0 if path_mode.endswith("LEFT") else -1.0
            target = float(max(self.config["actions"]["nudge_offsets_m"])) * sign
            arc = np.linspace(0.0, max(20.0, ctx.route_info.goal_progress_s), num=num)
            offset = smooth_lateral_transition(arc, start_s=0.0, duration_s=15.0, target_offset=target)
            path = frenet_to_cartesian(base, np.linspace(0.0, max(20.0, ctx.route_info.goal_progress_s), num=num), offset)
            return path, {"lateral_target": target, "maneuver_duration": 1.5}
        return base, meta

    def _fallback_centerline(self, ctx: RuntimeContext) -> np.ndarray:
        num = int(round(self.output_horizon_s / self.output_dt_s)) + 1
        dist = np.linspace(0.0, 90.0, num=num)
        return np.stack(
            [
                ctx.ego_state.pose.x + np.cos(ctx.ego_state.pose.heading) * dist,
                ctx.ego_state.pose.y + np.sin(ctx.ego_state.pose.heading) * dist,
            ],
            axis=-1,
        )
