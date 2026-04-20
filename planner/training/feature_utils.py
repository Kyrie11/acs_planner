from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from planner.actions.action_types import RefinedAction
from planner.runtime.types import MapPolyline, RuntimeContext, SceneAgent
from planner.support.support_types import Anchor, Atom

OBJECT_TYPE_TO_ID = {
    "VEHICLE": 1,
    "PEDESTRIAN": 2,
    "BICYCLE": 3,
    "CYCLIST": 4,
    "GENERIC_OBJECT": 5,
    "UNKNOWN": 0,
}

LAYER_TO_ID = {
    "LANE": 1,
    "LANE_CONNECTOR": 2,
    "ROADBLOCK": 3,
    "ROADBLOCK_CONNECTOR": 4,
    "STOP_LINE": 5,
    "CROSSWALK": 6,
    "INTERSECTION": 7,
    "DRIVABLE_AREA": 8,
}

PATH_MODE_TO_ID = {
    "KEEP_ROUTE": 0,
    "BRANCH_0": 1,
    "BRANCH_1": 2,
    "BRANCH_2": 3,
    "BRANCH_3": 4,
    "LC_LEFT": 5,
    "LC_RIGHT": 6,
    "MERGE_LEFT": 7,
    "MERGE_RIGHT": 8,
    "NUDGE_LEFT": 9,
    "NUDGE_RIGHT": 10,
}

SPEED_MODE_TO_ID = {"FOLLOW": 0, "CRUISE": 1, "DECEL": 2, "STOP": 3, "CREEP": 4}
ANCHOR_TYPE_TO_ID = {
    "branch": 1,
    "conflict": 2,
    "merge": 3,
    "stop": 4,
    "PED_CROSS": 5,
    "ONCOMING_TURN": 6,
    "PARKED_BYPASS": 7,
    "YIELD_ZONE": 8,
}
PRECEDENCE_TO_ID = {None: 0, "EGO_FIRST": 1, "OTHER_FIRST": 2}
GAP_TO_ID = {None: 0, "OPEN": 1, "TIGHT": 2, "CLOSED": 3}
BRANCH_TO_ID = {None: 0, "CONFLICTING_BRANCH": 1, "NONCONFLICTING_BRANCH": 2, "UNKNOWN_BRANCH": 3}


def _type_id(name: str) -> int:
    return OBJECT_TYPE_TO_ID.get(str(name).upper(), 0)


def _polyline_feature(ctx: RuntimeContext, polyline: MapPolyline) -> np.ndarray:
    start = polyline.points[0]
    end = polyline.points[-1]
    ego = ctx.ego_state.pose
    vec = end - start
    heading = float(np.arctan2(vec[1], vec[0])) if np.linalg.norm(vec) > 1e-3 else 0.0
    length = float(np.linalg.norm(vec))
    return np.array(
        [
            float(start[0] - ego.x),
            float(start[1] - ego.y),
            float(end[0] - ego.x),
            float(end[1] - ego.y),
            length,
            heading - ego.heading,
            float(LAYER_TO_ID.get(polyline.layer, 0)),
            float(polyline.layer in {"LANE", "LANE_CONNECTOR"}),
        ],
        dtype=np.float32,
    )


def _agent_feature(ctx: RuntimeContext, agent: SceneAgent) -> np.ndarray:
    ego = ctx.ego_state.pose
    return np.array(
        [
            float(agent.pose.x - ego.x),
            float(agent.pose.y - ego.y),
            float(agent.pose.heading - ego.heading),
            float(agent.dynamic.vx),
            float(agent.dynamic.vy),
            float(agent.dynamic.ax),
            float(agent.dynamic.ay),
            float(agent.size[0]),
            float(agent.size[1]),
            float(min(agent.ttc, 10.0)),
            float(agent.route_corridor),
            float(_type_id(agent.object_type)),
        ],
        dtype=np.float32,
    )


def _ego_history_feature(ctx: RuntimeContext) -> np.ndarray:
    rows = []
    ref = ctx.ego_state.pose
    for state in ctx.ego_history:
        rows.append(
            [
                float(state.pose.x - ref.x),
                float(state.pose.y - ref.y),
                float(state.pose.heading - ref.heading),
                float(state.dynamic.vx),
                float(state.dynamic.vy),
                float(state.dynamic.ax),
                float(state.dynamic.ay),
                float(state.tire_steering_angle),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _action_feature(action: RefinedAction) -> np.ndarray:
    return np.array(
        [
            float(PATH_MODE_TO_ID.get(action.token.path_mode, 0)),
            float(SPEED_MODE_TO_ID.get(action.token.speed_mode, 0)),
            float(action.is_conservative),
            float(action.coarse_score if np.isfinite(action.coarse_score) else 0.0),
            float(action.refinement.get("delta_v_terminal", 0.0)),
            float(action.refinement.get("delta_stop_offset", 0.0)),
            float(action.refinement.get("delta_time_shift", 0.0)),
            float(action.refinement.get("lc_start_delay", 0.0)),
            float(action.refinement.get("lc_duration", 0.0)),
            float(action.refinement.get("lateral_offset", 0.0)),
            float(action.refinement.get("creep_distance", 0.0)),
            float(action.refinement.get("creep_speed_cap", 0.0)),
        ],
        dtype=np.float32,
    )


def _anchor_feature(anchor: Anchor, atom: Atom) -> np.ndarray:
    state = atom.assignments.get(anchor.anchor_id)
    return np.array(
        [
            float(ANCHOR_TYPE_TO_ID.get(anchor.anchor_type, 0)),
            float(anchor.ego_s),
            float(anchor.ego_t_nominal),
            float(anchor.criticality),
            float(_type_id(anchor.metadata.get("agent_type", "UNKNOWN"))),
            float(PRECEDENCE_TO_ID.get(getattr(state, "precedence", None), 0)),
            float(GAP_TO_ID.get(getattr(state, "gap_state", None), 0)),
            float(BRANCH_TO_ID.get(getattr(state, "branch", None), 0)),
            float(_release_id(getattr(state, "release", None))),
            float(_occupancy_id(getattr(state, "occupancy", None))),
            float(getattr(state, "active", True)),
            float(len(anchor.agent_ids)),
            float(anchor.metadata.get("min_dist", 0.0)),
            float(anchor.metadata.get("ttc", 10.0) if np.isfinite(anchor.metadata.get("ttc", 10.0)) else 10.0),
            float(anchor.metadata.get("distance", 0.0)),
            1.0,
        ],
        dtype=np.float32,
    )


def _release_id(value: Any) -> int:
    if value is None:
        return 0
    if value == "NEVER":
        return 7
    if isinstance(value, str) and value.startswith("BIN_"):
        return int(value.split("_")[-1]) + 1
    return 0


def _occupancy_id(value: Any) -> int:
    if value is None or value == "NONE":
        return 0
    if isinstance(value, str) and value.startswith("BIN_"):
        return int(value.split("_")[-1]) + 1
    return 0


def build_scene_action_atom_tensors(ctx: RuntimeContext, action: RefinedAction, atom: Atom, support, config: dict) -> Dict[str, torch.Tensor]:
    model_cfg = config.get("model", {})
    max_agents = int(model_cfg.get("max_agents", config["context"]["max_agents_total"]))
    max_map = int(model_cfg.get("max_map_polylines", 128))
    max_anchors = int(model_cfg.get("max_atom_anchors", config["support"]["max_total_anchors_per_action"]))
    ego_feat = _ego_history_feature(ctx)
    ego_feat, ego_mask = _pad_rows(ego_feat, int(model_cfg.get("max_ego_history", len(ego_feat))), ego_feat.shape[-1] if ego_feat.size else 8)
    agent_feat = np.asarray([_agent_feature(ctx, a) for a in ctx.agents_all[:max_agents]], dtype=np.float32) if ctx.agents_all else np.zeros((0, 12), dtype=np.float32)
    agent_feat, agent_mask = _pad_rows(agent_feat, max_agents, 12)
    map_feat = np.asarray([_polyline_feature(ctx, p) for p in ctx.map_polylines[:max_map]], dtype=np.float32) if ctx.map_polylines else np.zeros((0, 8), dtype=np.float32)
    map_feat, map_mask = _pad_rows(map_feat, max_map, 8)
    anchors = support.anchors[:max_anchors]
    anchor_feat = np.asarray([_anchor_feature(anchor, atom) for anchor in anchors], dtype=np.float32) if anchors else np.zeros((0, 16), dtype=np.float32)
    anchor_feat, anchor_mask = _pad_rows(anchor_feat, max_anchors, 16)
    return {
        "ego_history": torch.from_numpy(ego_feat),
        "ego_mask": torch.from_numpy(ego_mask),
        "agents": torch.from_numpy(agent_feat),
        "agent_mask": torch.from_numpy(agent_mask),
        "map_polylines": torch.from_numpy(map_feat),
        "map_mask": torch.from_numpy(map_mask),
        "action_features": torch.from_numpy(_action_feature(action)),
        "atom_anchor_features": torch.from_numpy(anchor_feat),
        "atom_anchor_mask": torch.from_numpy(anchor_mask),
    }


def _pad_rows(arr: np.ndarray, max_rows: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_rows, dim), dtype=np.float32)
    mask = np.zeros((max_rows,), dtype=np.bool_)
    rows = min(max_rows, len(arr))
    if rows > 0:
        out[:rows] = arr[:rows]
        mask[:rows] = True
    return out, mask


def collate_tensor_dict(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out
