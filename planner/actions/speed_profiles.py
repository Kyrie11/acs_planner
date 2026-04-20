from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from planner.runtime.types import RuntimeContext


@dataclass
class SpeedProfileResult:
    speed_profile: np.ndarray
    stop_point_s: Optional[float]
    target_speed: float
    metadata: Dict[str, float]


class SpeedProfileGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.output_horizon_s = float(config["planner"]["output_horizon_s"])
        self.output_dt_s = float(config["planner"]["output_dt_s"])
        self.creep_speed_mps = float(config["planner"]["creep_speed_mps"])
        self.max_comfort_brake_mps2 = float(config["planner"]["max_comfort_brake_mps2"])

    def generate(self, ctx: RuntimeContext, speed_mode: str, path_mode: str) -> SpeedProfileResult:
        num = int(round(self.output_horizon_s / self.output_dt_s)) + 1
        v0 = max(ctx.ego_state.dynamic.vx, 0.0)
        v_ref = float(ctx.route_info.reference_speed_limit_mps)
        stop_point_s = self._infer_stop_point_s(ctx, speed_mode)

        if speed_mode == "FOLLOW":
            lead = self._closest_lead_agent(ctx)
            if lead is not None:
                gap = max(4.0, np.hypot(lead.pose.x - ctx.ego_state.pose.x, lead.pose.y - ctx.ego_state.pose.y) - 6.0)
                target = min(v_ref, max(0.0, lead.dynamic.vx + 0.25 * gap))
            else:
                target = min(v_ref, max(v0, 0.7 * v_ref))
            return SpeedProfileResult(self._jerk_limited(v0, target, num), stop_point_s, target, {"gap_control": 1.0})
        if speed_mode == "CRUISE":
            target = v_ref
            return SpeedProfileResult(self._jerk_limited(v0, target, num), None, target, {"cruise": 1.0})
        if speed_mode == "DECEL":
            target = max(0.0, 0.5 * v_ref)
            return SpeedProfileResult(self._jerk_limited(v0, target, num), stop_point_s, target, {"decel": 1.0})
        if speed_mode == "STOP":
            stop_point_s = stop_point_s if stop_point_s is not None else max(8.0, 1.5 * v0 * v0 / max(self.max_comfort_brake_mps2, 1e-3))
            return SpeedProfileResult(self._stop_profile(v0, stop_point_s, num), stop_point_s, 0.0, {"stop": 1.0})
        if speed_mode == "CREEP":
            distance = float(np.mean(self.config["actions"]["creep_distances_m"]))
            target = self.creep_speed_mps
            return SpeedProfileResult(self._creep_profile(v0, target, distance, num), distance, target, {"creep": 1.0})
        return SpeedProfileResult(self._jerk_limited(v0, v_ref, num), None, v_ref, {})

    def _closest_lead_agent(self, ctx: RuntimeContext):
        if not ctx.agents_interaction:
            return None
        ego_heading = ctx.ego_state.pose.heading
        front_agents = []
        for agent in ctx.agents_interaction:
            rel = np.array([agent.pose.x - ctx.ego_state.pose.x, agent.pose.y - ctx.ego_state.pose.y], dtype=np.float64)
            longitudinal = rel[0] * np.cos(ego_heading) + rel[1] * np.sin(ego_heading)
            lateral = -rel[0] * np.sin(ego_heading) + rel[1] * np.cos(ego_heading)
            if longitudinal > 0.0 and abs(lateral) < 4.0:
                front_agents.append((longitudinal, agent))
        if not front_agents:
            return None
        front_agents.sort(key=lambda item: item[0])
        return front_agents[0][1]

    def _infer_stop_point_s(self, ctx: RuntimeContext, speed_mode: str) -> Optional[float]:
        if speed_mode not in {"STOP", "DECEL", "CREEP"}:
            return None
        stop_candidates: List[float] = []
        # traffic light induced stop
        for branch in ctx.route_info.route_branches:
            if branch.connector_id and branch.connector_id in ctx.traffic_lights:
                status = ctx.traffic_lights[branch.connector_id].status.upper()
                if status in {"RED", "YELLOW"}:
                    stop_candidates.append(15.0)
        lead = self._closest_lead_agent(ctx)
        if lead is not None:
            gap = max(4.0, np.hypot(lead.pose.x - ctx.ego_state.pose.x, lead.pose.y - ctx.ego_state.pose.y) - 6.0)
            stop_candidates.append(gap)
        return min(stop_candidates) if stop_candidates else None

    def _jerk_limited(self, v0: float, target: float, num: int) -> np.ndarray:
        alpha = np.linspace(0.0, 1.0, num=num)
        return v0 + (target - v0) * (3 * alpha**2 - 2 * alpha**3)

    def _stop_profile(self, v0: float, stop_point_s: float, num: int) -> np.ndarray:
        times = np.arange(num, dtype=np.float64) * self.output_dt_s
        total_t = max(2.0, stop_point_s / max(v0, 1.0))
        alpha = np.clip(times / total_t, 0.0, 1.0)
        profile = v0 * (1.0 - (3 * alpha**2 - 2 * alpha**3))
        profile[profile < 0.2] = 0.0
        return profile

    def _creep_profile(self, v0: float, target: float, distance: float, num: int) -> np.ndarray:
        ramp = self._jerk_limited(v0, target, num)
        s = np.cumsum(ramp) * self.output_dt_s
        ramp[s > distance] = 0.0
        return ramp
