from __future__ import annotations

from typing import List

import numpy as np

from planner.actions.action_types import RefinedAction
from planner.common.geometry import cumulative_arc_length, min_distance_between_polylines, project_point_to_polyline, straight_line_projection, ttc_along_tracks
from planner.runtime.types import RuntimeContext, SceneAgent
from planner.support.support_types import Anchor


class AnchorExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.dt = float(config["planner"]["output_dt_s"])
        self.horizon_s = float(config["planner"]["support_eval_horizon_s"])
        self.max_conflict = int(config["support"]["max_conflict_anchors"])
        self.max_merge = int(config["support"]["max_merge_anchors"])
        self.max_stop = int(config["support"]["max_stop_anchors"])
        self.max_interact = int(config["support"]["max_interact_anchors"])
        self.max_total = int(config["support"]["max_total_anchors_per_action"])

    def extract(self, ctx: RuntimeContext, action: RefinedAction) -> List[Anchor]:
        anchors: List[Anchor] = []
        anchors.extend(self._extract_branch_anchors(ctx, action))
        anchors.extend(self._extract_conflict_anchors(ctx, action))
        anchors.extend(self._extract_merge_anchors(ctx, action))
        anchors.extend(self._extract_stop_anchors(ctx, action))
        anchors.extend(self._extract_interact_anchors(ctx, action))
        anchors.sort(key=lambda a: (-a.criticality, a.ego_s))
        return anchors[: self.max_total]

    def _extract_branch_anchors(self, ctx: RuntimeContext, action: RefinedAction) -> List[Anchor]:
        anchors: List[Anchor] = []
        if not ctx.route_info.route_branches:
            return anchors
        for idx, agent in enumerate(ctx.agents_interaction[: self.max_interact]):
            if agent.ttc >= 6.0:
                continue
            anchors.append(
                Anchor(
                    anchor_id=f"branch_{action.action_id}_{idx}",
                    base_anchor_id=f"branch_{agent.track_token}",
                    anchor_type="branch",
                    ego_s=15.0 + idx,
                    ego_t_nominal=2.0,
                    geometry=np.asarray([[agent.pose.x, agent.pose.y]], dtype=np.float64),
                    agent_ids=[agent.track_token],
                    metadata={"agent_type": agent.object_type},
                    criticality=max(0.1, 1.0 / (1.0 + agent.ttc)),
                )
            )
        return anchors[:1]

    def _extract_conflict_anchors(self, ctx: RuntimeContext, action: RefinedAction) -> List[Anchor]:
        anchors: List[Anchor] = []
        ego_track = np.asarray([[p.x, p.y] for p in action.refined_traj], dtype=np.float64)
        ego_arc = cumulative_arc_length(ego_track)
        for agent in ctx.agents_interaction:
            other_track = straight_line_projection(agent.pose, max(agent.dynamic.vx, 0.0), dt=self.dt, horizon_s=self.horizon_s)
            min_dist = min_distance_between_polylines(ego_track, other_track)
            ttc = ttc_along_tracks(ego_track, other_track, dt=self.dt, threshold_m=2.5)
            if min_dist > 2.5 and (not np.isfinite(ttc) or ttc > 3.0):
                continue
            idx = self._closest_pair_index(ego_track, other_track)
            if idx is None:
                continue
            ego_idx, other_idx = idx
            anchors.append(
                Anchor(
                    anchor_id=f"conflict_{action.action_id}_{agent.track_token}_{len(anchors)}",
                    base_anchor_id=f"conflict_{agent.track_token}",
                    anchor_type="conflict",
                    ego_s=float(ego_arc[min(ego_idx, len(ego_arc) - 1)]),
                    ego_t_nominal=float(ego_idx * self.dt),
                    geometry=np.vstack([ego_track[ego_idx], other_track[other_idx]]),
                    agent_ids=[agent.track_token],
                    metadata={"agent_type": agent.object_type, "min_dist": min_dist, "ttc": ttc},
                    criticality=max(0.1, 1.0 / (1.0 + min(min_dist, 10.0))) + (0.5 if np.isfinite(ttc) else 0.0),
                )
            )
        anchors.sort(key=lambda a: (a.ego_t_nominal, -a.criticality))
        return anchors[: self.max_conflict]

    def _extract_merge_anchors(self, ctx: RuntimeContext, action: RefinedAction) -> List[Anchor]:
        if not action.token.path_mode.startswith(("LC", "MERGE")):
            return []
        anchors: List[Anchor] = []
        ego_track = np.asarray([[p.x, p.y] for p in action.refined_traj], dtype=np.float64)
        ego_arc = cumulative_arc_length(ego_track)
        for agent in ctx.agents_interaction:
            if agent.object_type != "VEHICLE":
                continue
            other_track = straight_line_projection(agent.pose, max(agent.dynamic.vx, 0.0), dt=self.dt, horizon_s=self.horizon_s)
            ttc = ttc_along_tracks(ego_track, other_track, dt=self.dt, threshold_m=3.5)
            if not np.isfinite(ttc) or ttc > 4.0:
                continue
            ego_idx = min(int(round(ttc / self.dt)), len(ego_track) - 1)
            anchors.append(
                Anchor(
                    anchor_id=f"merge_{action.action_id}_{agent.track_token}_{len(anchors)}",
                    base_anchor_id=f"merge_{agent.track_token}",
                    anchor_type="merge",
                    ego_s=float(ego_arc[ego_idx]),
                    ego_t_nominal=float(ttc),
                    geometry=np.vstack([ego_track[ego_idx], other_track[min(ego_idx, len(other_track)-1)]]),
                    agent_ids=[agent.track_token],
                    metadata={"agent_type": agent.object_type, "ttc": ttc},
                    criticality=max(0.2, 1.2 / (1.0 + ttc)),
                )
            )
        anchors.sort(key=lambda a: a.ego_t_nominal)
        return anchors[: self.max_merge]

    def _extract_stop_anchors(self, ctx: RuntimeContext, action: RefinedAction) -> List[Anchor]:
        anchors: List[Anchor] = []
        # Red/yellow light stop.
        if any(entry.status.upper() in {"RED", "YELLOW"} for entry in ctx.traffic_lights.values()):
            anchors.append(
                Anchor(
                    anchor_id=f"stop_light_{action.action_id}",
                    base_anchor_id="stop_light",
                    anchor_type="stop",
                    ego_s=15.0,
                    ego_t_nominal=2.0,
                    geometry=np.asarray([[ctx.ego_state.pose.x, ctx.ego_state.pose.y]], dtype=np.float64),
                    agent_ids=[],
                    metadata={"source": "traffic_light"},
                    criticality=2.0,
                )
            )
        # Lead vehicle stop.
        if ctx.agents_interaction:
            lead = min(
                ctx.agents_interaction,
                key=lambda a: np.hypot(a.pose.x - ctx.ego_state.pose.x, a.pose.y - ctx.ego_state.pose.y),
            )
            dist = float(np.hypot(lead.pose.x - ctx.ego_state.pose.x, lead.pose.y - ctx.ego_state.pose.y))
            if dist < 35.0:
                anchors.append(
                    Anchor(
                        anchor_id=f"stop_lead_{action.action_id}",
                        base_anchor_id=f"stop_{lead.track_token}",
                        anchor_type="stop",
                        ego_s=max(5.0, dist - 6.0),
                        ego_t_nominal=max(1.0, dist / max(ctx.ego_state.dynamic.vx + 1e-2, 1.0)),
                        geometry=np.asarray([[lead.pose.x, lead.pose.y]], dtype=np.float64),
                        agent_ids=[lead.track_token],
                        metadata={"source": "lead_vehicle", "distance": dist},
                        criticality=max(0.4, 2.0 / (1.0 + dist)),
                    )
                )
        anchors.sort(key=lambda a: a.ego_s)
        deduped: List[Anchor] = []
        for anchor in anchors:
            if not deduped or abs(anchor.ego_s - deduped[-1].ego_s) > 5.0:
                deduped.append(anchor)
        return deduped[: self.max_stop]

    def _extract_interact_anchors(self, ctx: RuntimeContext, action: RefinedAction) -> List[Anchor]:
        anchors: List[Anchor] = []
        ego_track = np.asarray([[p.x, p.y] for p in action.refined_traj], dtype=np.float64)
        ego_arc = cumulative_arc_length(ego_track)
        for agent in ctx.agents_interaction:
            if agent.object_type in {"PEDESTRIAN", "BICYCLE", "CYCLIST"}:
                other_track = straight_line_projection(agent.pose, max(agent.dynamic.vx, 0.2), dt=self.dt, horizon_s=self.horizon_s)
                ttc = ttc_along_tracks(ego_track, other_track, dt=self.dt, threshold_m=2.0)
                if np.isfinite(ttc) and ttc < 6.0:
                    ego_idx = min(int(round(ttc / self.dt)), len(ego_track) - 1)
                    anchors.append(
                        Anchor(
                            anchor_id=f"ped_cross_{action.action_id}_{agent.track_token}",
                            base_anchor_id=f"ped_cross_{agent.track_token}",
                            anchor_type="PED_CROSS",
                            ego_s=float(ego_arc[ego_idx]),
                            ego_t_nominal=float(ttc),
                            geometry=np.vstack([ego_track[ego_idx], other_track[min(ego_idx, len(other_track)-1)]]),
                            agent_ids=[agent.track_token],
                            metadata={"agent_type": agent.object_type, "ttc": ttc},
                            criticality=max(0.4, 1.5 / (1.0 + ttc)),
                        )
                    )
            elif action.token.path_mode.startswith("NUDGE") and agent.object_type == "VEHICLE":
                dist = float(np.hypot(agent.pose.x - ctx.ego_state.pose.x, agent.pose.y - ctx.ego_state.pose.y))
                if dist < 18.0:
                    anchors.append(
                        Anchor(
                            anchor_id=f"parked_bypass_{action.action_id}_{agent.track_token}",
                            base_anchor_id=f"parked_bypass_{agent.track_token}",
                            anchor_type="PARKED_BYPASS",
                            ego_s=max(3.0, dist),
                            ego_t_nominal=max(1.0, dist / max(ctx.ego_state.dynamic.vx + 1e-2, 1.0)),
                            geometry=np.asarray([[agent.pose.x, agent.pose.y]], dtype=np.float64),
                            agent_ids=[agent.track_token],
                            metadata={"distance": dist},
                            criticality=max(0.3, 1.0 / (1.0 + dist)),
                        )
                    )
        anchors.sort(key=lambda a: a.ego_t_nominal)
        return anchors[: self.max_interact]

    def _closest_pair_index(self, a: np.ndarray, b: np.ndarray):
        if len(a) == 0 or len(b) == 0:
            return None
        diff = a[:, None, :] - b[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        idx = np.unravel_index(int(np.argmin(dist)), dist.shape)
        return idx
