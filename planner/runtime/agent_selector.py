from __future__ import annotations

from typing import Iterable, List

import numpy as np

from planner.common.geometry import SE2, clamp, straight_line_projection, ttc_along_tracks
from planner.runtime.types import SceneAgent


class AgentSelector:
    def __init__(self, config: dict):
        self.config = config
        self.max_agents_total = int(config["context"]["max_agents_total"])
        self.max_agents_interaction = int(config["context"]["max_agents_interaction"])
        self.agent_radius_m = float(config["context"]["agent_radius_m"])
        self.w_d = float(config.get("agent_ranking", {}).get("w_d", 1.0))
        self.w_t = float(config.get("agent_ranking", {}).get("w_t", 2.0))
        self.w_r = float(config.get("agent_ranking", {}).get("w_r", 1.5))
        self.w_i = float(config.get("agent_ranking", {}).get("w_i", 2.0))

    def score_agents(
        self,
        ego_pose: SE2,
        ego_speed: float,
        agents: Iterable[SceneAgent],
        route_centerline: np.ndarray,
        dt: float,
    ) -> List[SceneAgent]:
        ego_track = straight_line_projection(ego_pose, ego_speed, dt=dt, horizon_s=6.0)
        scored: List[SceneAgent] = []
        for agent in agents:
            dist = float(np.hypot(agent.pose.x - ego_pose.x, agent.pose.y - ego_pose.y))
            if dist > self.agent_radius_m + 20.0:
                continue
            agent_track = straight_line_projection(agent.pose, max(agent.dynamic.vx, 0.0), dt=dt, horizon_s=6.0)
            ttc = ttc_along_tracks(ego_track, agent_track, dt=dt, threshold_m=3.5)
            on_route_corridor = False
            if len(route_centerline) > 0:
                on_route_corridor = float(np.min(np.linalg.norm(route_centerline - np.array([[agent.pose.x, agent.pose.y]]), axis=-1))) < 8.0
            branch_interacts = ttc < 6.0 or on_route_corridor
            score = (
                self.w_d * (1.0 / (1.0 + dist))
                + self.w_t * (1.0 / (1.0 + ttc))
                + self.w_r * float(on_route_corridor)
                + self.w_i * float(branch_interacts)
            )
            agent.ttc = ttc
            agent.route_corridor = on_route_corridor
            agent.score = float(score)
            scored.append(agent)
        scored.sort(key=lambda a: a.score, reverse=True)
        return scored

    def select(self, ego_pose: SE2, ego_speed: float, agents: Iterable[SceneAgent], route_centerline: np.ndarray, dt: float) -> tuple[List[SceneAgent], List[SceneAgent]]:
        ranked = self.score_agents(ego_pose, ego_speed, agents, route_centerline, dt=dt)
        total = ranked[: self.max_agents_total]
        interaction: List[SceneAgent] = []
        for agent in total:
            if len(interaction) >= self.max_agents_interaction:
                break
            if agent.ttc < 6.0 or agent.route_corridor:
                interaction.append(agent)
        return total, interaction
