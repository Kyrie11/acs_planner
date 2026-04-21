from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from acs_planner.actions.action_types import Action, PathMode, RefinementDomain, SpeedMode
from acs_planner.common.types import RuntimeContext


@dataclass(slots=True)
class ActionLibraryConfig:
    enable_keep: bool = True
    enable_branch: bool = True
    enable_lane_change: bool = True
    enable_merge: bool = True
    enable_nudge: bool = True
    lane_change_duration_s: tuple[float, ...] = (2.5, 3.5, 4.5)
    nudge_offsets_m: tuple[float, ...] = (0.3, 0.6)
    creep_distances_m: tuple[float, ...] = (1.0, 2.0, 3.0)
    terminal_speed_deltas_mps: tuple[float, ...] = (-2.0, 0.0, 2.0)


def _default_refinement(path_mode: PathMode, speed_mode: SpeedMode, cfg: ActionLibraryConfig) -> RefinementDomain:
    if path_mode in {PathMode.LC_LEFT, PathMode.LC_RIGHT, PathMode.MERGE_LEFT, PathMode.MERGE_RIGHT}:
        return RefinementDomain(
            {
                "lc_start_delay": [0.0, 0.5, 1.0],
                "lc_duration": list(cfg.lane_change_duration_s),
                "delta_v_terminal": list(cfg.terminal_speed_deltas_mps),
            }
        )
    if path_mode in {PathMode.NUDGE_LEFT, PathMode.NUDGE_RIGHT} or speed_mode == SpeedMode.CREEP:
        return RefinementDomain(
            {
                "lateral_offset": [0.3, 0.6],
                "creep_distance": list(cfg.creep_distances_m),
                "creep_speed_cap": [0.5, 1.0, 1.5],
            }
        )
    return RefinementDomain(
        {
            "delta_v_terminal": list(cfg.terminal_speed_deltas_mps),
            "delta_stop_offset": [-3.0, -1.0, 0.0, 1.0],
            "delta_time_shift": [-0.5, 0.0, 0.5],
        }
    )


def _is_conservative(path_mode: PathMode, speed_mode: SpeedMode) -> bool:
    if path_mode == PathMode.KEEP_ROUTE and speed_mode in {SpeedMode.FOLLOW, SpeedMode.DECEL, SpeedMode.STOP, SpeedMode.CREEP}:
        return True
    if path_mode == PathMode.BRANCH and speed_mode in {SpeedMode.DECEL, SpeedMode.STOP, SpeedMode.CREEP}:
        return True
    return False


def _dummy_traj(route_points: list[tuple[float, float]], speed: float) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for i, (x, y) in enumerate(route_points[:81]):
        out.append({"t": 0.1 * i, "x": x, "y": y, "v": speed})
    return out


def generate_action_library(ctx: RuntimeContext, cfg: ActionLibraryConfig) -> list[Action]:
    ctx.validate()
    actions: list[Action] = []
    route_points = list(ctx.route_centerline.points)
    action_idx = 0

    def add(path_mode: PathMode, speed_mode: SpeedMode, nominal_path: list[tuple[float, float]] | None = None) -> None:
        nonlocal action_idx
        path = nominal_path if nominal_path is not None else route_points
        speed = 0.5 if speed_mode == SpeedMode.CREEP else 5.0
        actions.append(
            Action(
                action_id=f"a{action_idx}",
                path_mode=path_mode,
                speed_mode=speed_mode,
                nominal_path=path,
                nominal_speed_profile=[speed] * min(81, max(len(path), 1)),
                nominal_traj=_dummy_traj(path, speed),
                refine_domain=_default_refinement(path_mode, speed_mode, cfg),
                is_conservative=_is_conservative(path_mode, speed_mode),
            )
        )
        action_idx += 1

    if cfg.enable_keep:
        for sm in (SpeedMode.FOLLOW, SpeedMode.CRUISE, SpeedMode.DECEL, SpeedMode.STOP, SpeedMode.CREEP):
            add(PathMode.KEEP_ROUTE, sm)

    if cfg.enable_branch:
        for branch in ctx.route_branches:
            for sm in (SpeedMode.CRUISE, SpeedMode.DECEL, SpeedMode.STOP, SpeedMode.CREEP):
                add(PathMode.BRANCH, sm, nominal_path=list(branch.centerline.points))

    if cfg.enable_lane_change:
        add(PathMode.LC_LEFT, SpeedMode.CRUISE)
        add(PathMode.LC_RIGHT, SpeedMode.CRUISE)

    if cfg.enable_merge:
        add(PathMode.MERGE_LEFT, SpeedMode.FOLLOW)
        add(PathMode.MERGE_RIGHT, SpeedMode.FOLLOW)

    if cfg.enable_nudge:
        add(PathMode.NUDGE_LEFT, SpeedMode.CREEP)
        add(PathMode.NUDGE_RIGHT, SpeedMode.CREEP)

    return actions


def conservative_actions(actions: Iterable[Action]) -> list[Action]:
    return [a for a in actions if a.is_conservative]
