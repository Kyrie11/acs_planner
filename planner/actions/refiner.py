from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List

import numpy as np

from planner.actions.action_types import ActionCandidate, RefinedAction
from planner.common.geometry import cumulative_arc_length, frenet_to_cartesian, resample_trajectory, smooth_lateral_transition
from planner.evaluation.coarse_planner import CoarsePlanner
from planner.runtime.types import RuntimeContext


class ActionRefiner:
    def __init__(self, config: dict, coarse_planner: CoarsePlanner):
        self.config = config
        self.output_horizon_s = float(config["planner"]["output_horizon_s"])
        self.output_dt_s = float(config["planner"]["output_dt_s"])
        self.coarse_planner = coarse_planner

    def refine_actions(self, ctx: RuntimeContext, actions: Iterable[ActionCandidate]) -> List[RefinedAction]:
        refined: List[RefinedAction] = []
        for action in actions:
            refined.append(self.refine_single(ctx, action))
        refined.sort(key=lambda a: a.coarse_score)
        return refined

    def refine_single(self, ctx: RuntimeContext, action: ActionCandidate) -> RefinedAction:
        candidates = self._enumerate_refinements(action)
        best: RefinedAction | None = None
        for params in candidates:
            path, speed = self._apply_refinement(action, params)
            traj = self._combine_path_and_speed(path, speed)
            coarse_score = self.coarse_planner.score_trajectory(ctx, action.token, traj)
            current = RefinedAction(action=action, refinement=params, refined_path=path, refined_speed_profile=speed, refined_traj=traj, coarse_score=coarse_score)
            if best is None or current.coarse_score < best.coarse_score:
                best = current
        assert best is not None
        return best

    def _enumerate_refinements(self, action: ActionCandidate) -> List[Dict[str, float]]:
        domain = action.refine_domain
        if action.token.path_mode.startswith(("LC", "MERGE")):
            grid = product(domain.lc_start_delay or [0.0], domain.lc_duration or [3.5], domain.delta_v_terminal or [0.0])
            return [
                {"lc_start_delay": float(a), "lc_duration": float(b), "delta_v_terminal": float(c)}
                for a, b, c in grid
            ]
        if action.token.path_mode.startswith("NUDGE") or action.token.speed_mode == "CREEP":
            grid = product(domain.lateral_offset or [0.3], domain.creep_distance or [2.0], domain.creep_speed_cap or [1.0])
            signed = 1.0 if action.token.path_mode.endswith("LEFT") else -1.0
            return [
                {"lateral_offset": signed * float(a), "creep_distance": float(b), "creep_speed_cap": float(c)}
                for a, b, c in grid
            ]
        grid = product(domain.delta_v_terminal or [0.0], domain.delta_stop_offset or [0.0], domain.delta_time_shift or [0.0])
        return [
            {"delta_v_terminal": float(a), "delta_stop_offset": float(b), "delta_time_shift": float(c)}
            for a, b, c in grid
        ]

    def _apply_refinement(self, action: ActionCandidate, params: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        path = np.asarray(action.nominal_path, dtype=np.float64).copy()
        speed = np.asarray(action.nominal_speed_profile, dtype=np.float64).copy()
        if "delta_v_terminal" in params:
            delta = params["delta_v_terminal"]
            alpha = np.linspace(0.0, 1.0, num=len(speed))
            speed = np.maximum(0.0, speed + delta * (3 * alpha**2 - 2 * alpha**3))
        if "delta_time_shift" in params and abs(params["delta_time_shift"]) > 1e-6:
            shift = int(round(params["delta_time_shift"] / self.output_dt_s))
            speed = np.roll(speed, shift)
            if shift > 0:
                speed[:shift] = speed[0]
            elif shift < 0:
                speed[shift:] = speed[-1]
        if "delta_stop_offset" in params and abs(params["delta_stop_offset"]) > 1e-6:
            stop_idx = np.argmax(speed <= 0.05)
            if stop_idx > 0:
                offset = int(round(params["delta_stop_offset"] / max(np.mean(speed[:max(1, stop_idx)]), 1.0) / max(self.output_dt_s, 1e-3)))
                speed = np.roll(speed, offset)
                if offset > 0:
                    speed[:offset] = action.nominal_speed_profile[0]
        if "lc_duration" in params or "lc_start_delay" in params:
            sign = 1.0 if action.token.path_mode.endswith("LEFT") else -1.0
            arc = np.linspace(0.0, float(cumulative_arc_length(path)[-1]) if len(path) > 1 else 1.0, num=len(path))
            start_s = 5.0 + 8.0 * params.get("lc_start_delay", 0.0)
            duration_s = max(10.0, 8.0 * params.get("lc_duration", 3.5))
            offset = smooth_lateral_transition(arc, start_s=start_s, duration_s=duration_s, target_offset=sign * 3.5)
            path = frenet_to_cartesian(path, arc, offset)
        if "lateral_offset" in params:
            arc = np.linspace(0.0, float(cumulative_arc_length(path)[-1]) if len(path) > 1 else 1.0, num=len(path))
            offset = smooth_lateral_transition(arc, 0.0, 12.0, params["lateral_offset"])
            path = frenet_to_cartesian(path, arc, offset)
        if "creep_speed_cap" in params:
            speed = np.minimum(speed, params["creep_speed_cap"])
        if "creep_distance" in params:
            s = np.cumsum(speed) * self.output_dt_s
            speed[s > params["creep_distance"]] = 0.0
        return path, speed

    def _combine_path_and_speed(self, path_xy: np.ndarray, speed_profile: np.ndarray):
        arc = cumulative_arc_length(path_xy)
        s = np.cumsum(speed_profile) * self.output_dt_s
        s = np.clip(s, 0.0, arc[-1] if len(arc) else 0.0)
        x = np.interp(s, arc, path_xy[:, 0])
        y = np.interp(s, arc, path_xy[:, 1])
        return resample_trajectory(np.stack([x, y], axis=-1), dt=self.output_dt_s, horizon_s=self.output_horizon_s)
