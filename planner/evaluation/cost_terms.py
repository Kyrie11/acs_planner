from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

from planner.actions.action_types import ActionToken
from planner.common.geometry import clip01, exceed, min_distance_between_polylines, project_point_to_polyline, straight_line_projection, ttc_along_tracks
from planner.runtime.types import RuntimeContext, SceneAgent


@dataclass
class CostBreakdown:
    progress: float
    route: float
    comfort: float
    rule: float
    safe: float
    interaction: float
    total: float
    metadata: Dict[str, float] = field(default_factory=dict)


class PlannerCost:
    def __init__(self, config: dict):
        self.config = config
        cost_cfg = config["cost"]
        self.lambda_prog = float(cost_cfg["lambda_prog"])
        self.lambda_route = float(cost_cfg["lambda_route"])
        self.lambda_comfort = float(cost_cfg["lambda_comfort"])
        self.lambda_rule = float(cost_cfg["lambda_rule"])
        self.lambda_safe = float(cost_cfg["lambda_safe"])
        self.lambda_int = float(cost_cfg["lambda_int"])
        self.output_dt_s = float(config["planner"]["output_dt_s"])
        self.output_horizon_s = float(config["planner"]["output_horizon_s"])

    def evaluate(self, ctx: RuntimeContext, token: ActionToken, traj: List, deterministic_only: bool = False) -> CostBreakdown:
        traj_xy = np.asarray([[p.x, p.y] for p in traj], dtype=np.float64)
        speeds = np.asarray([p.speed for p in traj], dtype=np.float64)
        accel = np.asarray([p.accel for p in traj], dtype=np.float64)
        curvature = np.asarray([p.curvature for p in traj], dtype=np.float64)
        headings = np.asarray([p.heading for p in traj], dtype=np.float64)
        route_centerline = ctx.route_info.route_centerline

        progress = self._progress_cost(ctx, traj_xy)
        route = self._route_cost(route_centerline, traj_xy, token)
        comfort = self._comfort_cost(speeds, accel, curvature, headings)
        rule = self._rule_cost(ctx, token, traj_xy, speeds)
        safe = self._safety_cost(ctx, traj_xy)
        interaction = 0.0 if deterministic_only else self._interaction_cost(ctx, token, traj_xy, speeds)
        total = (
            self.lambda_prog * progress
            + self.lambda_route * route
            + self.lambda_comfort * comfort
            + self.lambda_rule * rule
            + self.lambda_safe * safe
            + self.lambda_int * interaction
        )
        return CostBreakdown(progress, route, comfort, rule, safe, interaction, total)

    def _progress_cost(self, ctx: RuntimeContext, traj_xy: np.ndarray) -> float:
        if len(ctx.route_info.route_centerline) < 2 or len(traj_xy) == 0:
            return 1.0
        fr0 = project_point_to_polyline(traj_xy[0], ctx.route_info.route_centerline)
        fr1 = project_point_to_polyline(traj_xy[-1], ctx.route_info.route_centerline)
        delta_s = max(0.0, fr1.s - fr0.s)
        target = min(ctx.route_info.goal_progress_s, max(10.0, ctx.route_info.reference_speed_limit_mps * self.output_horizon_s))
        return clip01(1.0 - min(delta_s / max(target, 1e-3), 1.0))

    def _route_cost(self, route_centerline: np.ndarray, traj_xy: np.ndarray, token: ActionToken) -> float:
        if len(route_centerline) < 2 or len(traj_xy) == 0:
            return 1.0
        dists = np.asarray([abs(project_point_to_polyline(p, route_centerline).d) for p in traj_xy], dtype=np.float64)
        off_route_ratio = float(np.mean(dists > 3.5))
        branch_mismatch_flag = float(token.path_mode.startswith("BRANCH") and np.mean(dists) > 4.0)
        return clip01(0.5 * float(np.mean((dists / 1.75) ** 2)) + 0.3 * off_route_ratio + 0.2 * branch_mismatch_flag)

    def _comfort_cost(self, speeds: np.ndarray, accel: np.ndarray, curvature: np.ndarray, headings: np.ndarray) -> float:
        jerk = np.gradient(accel, self.output_dt_s) if len(accel) > 1 else np.zeros_like(accel)
        a_lat = curvature * speeds**2
        yaw_rate = np.gradient(headings, self.output_dt_s) if len(headings) > 1 else np.zeros_like(headings)
        steer_rate = np.gradient(curvature, self.output_dt_s) if len(curvature) > 1 else np.zeros_like(curvature)
        return clip01(
            0.35 * float(np.mean(exceed(jerk, 4.0) ** 2))
            + 0.35 * float(np.mean(exceed(a_lat, 3.5) ** 2))
            + 0.15 * float(np.mean(exceed(yaw_rate, 0.6) ** 2))
            + 0.15 * float(np.mean(exceed(steer_rate, 0.5) ** 2))
        )

    def _rule_cost(self, ctx: RuntimeContext, token: ActionToken, traj_xy: np.ndarray, speeds: np.ndarray) -> float:
        red_light_violation = 0.0
        for _, tl in ctx.traffic_lights.items():
            if tl.status.upper() == "RED" and token.speed_mode not in {"STOP", "CREEP", "DECEL"}:
                red_light_violation = 1.0
                break
        stop_violation = float(token.speed_mode == "STOP" and np.min(speeds) > 0.5)
        crosswalk_violation = 0.0
        drivable_area_violation = float(np.mean(np.abs([project_point_to_polyline(p, ctx.route_info.route_centerline).d for p in traj_xy]) > 5.0))
        wrong_way_violation = float(token.path_mode.startswith("BRANCH") and len(ctx.route_info.route_branches) == 0)
        speed_limit_violation = float(np.mean(speeds > (ctx.route_info.reference_speed_limit_mps + 2.0)))
        return clip01(
            0.30 * red_light_violation
            + 0.15 * stop_violation
            + 0.15 * crosswalk_violation
            + 0.20 * drivable_area_violation
            + 0.10 * wrong_way_violation
            + 0.10 * speed_limit_violation
        )

    def _safety_cost(self, ctx: RuntimeContext, traj_xy: np.ndarray) -> float:
        if len(traj_xy) == 0:
            return 1.0
        collision = 0.0
        ttc_penalties: List[float] = []
        min_dist_penalties: List[float] = []
        for agent in ctx.agents_interaction:
            other_track = straight_line_projection(agent.pose, max(agent.dynamic.vx, 0.0), dt=self.output_dt_s, horizon_s=self.output_horizon_s)
            min_dist = min_distance_between_polylines(traj_xy, other_track)
            min_dist_penalties.append(clip01((3.0 - min_dist) / 3.0))
            ttc = ttc_along_tracks(traj_xy, other_track, dt=self.output_dt_s, threshold_m=3.0)
            threshold = 2.0 if agent.object_type in {"PEDESTRIAN", "BICYCLE", "CYCLIST"} else 1.5
            ttc_penalties.append(clip01((threshold - ttc) / threshold) if np.isfinite(ttc) else 0.0)
            if min_dist < 1.5:
                collision = 1.0
        ttc_penalty = max(ttc_penalties) if ttc_penalties else 0.0
        min_distance_penalty = max(min_dist_penalties) if min_dist_penalties else 0.0
        static_clearance_penalty = float(np.mean(np.abs([project_point_to_polyline(p, ctx.route_info.route_centerline).d for p in traj_xy]) > 6.0))
        return clip01(0.55 * collision + 0.20 * ttc_penalty + 0.15 * min_distance_penalty + 0.10 * static_clearance_penalty)

    def _interaction_cost(self, ctx: RuntimeContext, token: ActionToken, traj_xy: np.ndarray, speeds: np.ndarray) -> float:
        precedence_violation = 0.0
        forced_brake_penalty = 0.0
        assertive_merge_penalty = 0.0
        deadlock_penalty = 0.0
        for agent in ctx.agents_interaction:
            other_track = straight_line_projection(agent.pose, max(agent.dynamic.vx, 0.0), dt=self.output_dt_s, horizon_s=self.output_horizon_s)
            ttc = ttc_along_tracks(traj_xy, other_track, dt=self.output_dt_s, threshold_m=3.0)
            if np.isfinite(ttc):
                if token.path_mode.startswith(("LC", "MERGE", "NUDGE")):
                    precedence_violation = max(precedence_violation, clip01((2.0 - ttc) / 2.0))
                if token.path_mode.startswith("MERGE") and agent.object_type == "VEHICLE":
                    assertive_merge_penalty = max(assertive_merge_penalty, clip01((2.5 - ttc) / 2.5))
                if speeds[min(int(ttc / self.output_dt_s), len(speeds) - 1)] > agent.dynamic.vx + 2.0:
                    forced_brake_penalty = max(forced_brake_penalty, 0.8)
        if token.speed_mode in {"STOP", "CREEP"} and np.mean(speeds) < 0.5 and self._progress_cost(ctx, traj_xy) > 0.7:
            deadlock_penalty = 0.7
        return clip01(
            0.45 * precedence_violation
            + 0.25 * forced_brake_penalty
            + 0.20 * assertive_merge_penalty
            + 0.10 * deadlock_penalty
        )
