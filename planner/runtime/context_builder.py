from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from planner.common.geometry import DynamicState, SE2, straight_line_projection
from planner.common.logging_utils import setup_logger
from planner.runtime.agent_selector import AgentSelector
from planner.runtime.map_cache import MapCache
from planner.runtime.route_builder import RouteBuilder
from planner.runtime.types import EgoHistoryFrame, RuntimeContext, SceneAgent, TrafficLightEntry

LOGGER = setup_logger(__name__)


class RuntimeContextBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.route_builder = RouteBuilder(config)
        self.agent_selector = AgentSelector(config)
        self.map_cache = MapCache()

    def build(self, planner_input: Any, initialization: Any) -> RuntimeContext:
        current_ego = self._extract_current_ego_state(planner_input.history)
        ego_history = self._extract_ego_history(planner_input.history)
        iteration_index = int(getattr(getattr(planner_input, "iteration", None), "index", len(ego_history) - 1))
        scenario_token = self._extract_scenario_token(planner_input, iteration_index)
        map_cache_key = (scenario_token, iteration_index, "default")
        cached = self.map_cache.get(map_cache_key)
        if cached is None:
            ego_xy = np.array([current_ego.pose.x, current_ego.pose.y], dtype=np.float64)
            polylines = self.map_cache.extract_polylines(initialization.map_api, ego_xy, self.config)
            self.map_cache.put(map_cache_key, polylines)
            map_polylines = polylines
        else:
            map_polylines = cached.polylines

        route_info = self.route_builder.build(
            ego_pose=current_ego.pose,
            map_api=initialization.map_api,
            route_roadblock_ids=list(getattr(initialization, "route_roadblock_ids", [])),
            map_polylines=map_polylines,
            mission_goal=getattr(initialization, "mission_goal", None),
        )
        raw_agents = self._extract_agents(planner_input.history)
        agents_all, agents_interaction = self.agent_selector.select(
            ego_pose=current_ego.pose,
            ego_speed=max(current_ego.dynamic.vx, 0.0),
            agents=raw_agents,
            route_centerline=route_info.route_centerline,
            dt=float(self.config["planner"]["output_dt_s"]),
        )
        traffic_lights = self._extract_traffic_lights(getattr(planner_input, "traffic_light_data", None))
        return RuntimeContext(
            scenario_token=scenario_token,
            iteration_index=iteration_index,
            ego_state=current_ego,
            ego_history=ego_history,
            agents_all=agents_all,
            agents_interaction=agents_interaction,
            map_polylines=map_polylines,
            route_info=route_info,
            traffic_lights=traffic_lights,
            mission_goal=getattr(initialization, "mission_goal", None),
            route_roadblock_ids=list(getattr(initialization, "route_roadblock_ids", [])),
            map_api=getattr(initialization, "map_api", None),
            history_buffer=getattr(planner_input, "history", None),
            raw_planner_input=planner_input,
            raw_initialization=initialization,
            config=self.config,
        )

    def _extract_current_ego_state(self, history: Any) -> EgoHistoryFrame:
        ego_states = getattr(history, "ego_states", None)
        if ego_states is None or len(ego_states) == 0:
            raise RuntimeError("Planner history does not expose ego_states")
        return self._convert_ego_state(ego_states[-1])

    def _extract_ego_history(self, history: Any) -> List[EgoHistoryFrame]:
        ego_states = list(getattr(history, "ego_states", []))
        if not ego_states:
            return []
        horizon_s = float(self.config["planner"]["history_horizon_s"])
        dt = float(self.config["planner"]["history_dt_s"])
        max_frames = int(round(horizon_s / dt)) + 1
        selected = ego_states[-max_frames:]
        return [self._convert_ego_state(state) for state in selected]

    def _convert_ego_state(self, ego_state: Any) -> EgoHistoryFrame:
        rear_axle = getattr(ego_state, "rear_axle", None)
        dyn = getattr(ego_state, "dynamic_car_state", None)
        vel = getattr(dyn, "rear_axle_velocity_2d", None)
        acc = getattr(dyn, "rear_axle_acceleration_2d", None)
        pose = SE2(
            x=float(getattr(rear_axle, "x", 0.0)),
            y=float(getattr(rear_axle, "y", 0.0)),
            heading=float(getattr(rear_axle, "heading", 0.0)),
        )
        dynamic = DynamicState(
            vx=float(getattr(vel, "x", 0.0)),
            vy=float(getattr(vel, "y", 0.0)),
            ax=float(getattr(acc, "x", 0.0)),
            ay=float(getattr(acc, "y", 0.0)),
            yaw_rate=float(getattr(dyn, "angular_velocity", 0.0)),
        )
        time_us = int(getattr(ego_state, "time_us", 0))
        return EgoHistoryFrame(
            pose=pose,
            dynamic=dynamic,
            tire_steering_angle=float(getattr(ego_state, "tire_steering_angle", 0.0)),
            time_s=time_us * 1e-6,
        )

    def _extract_agents(self, history: Any) -> List[SceneAgent]:
        current_obs = None
        observations = getattr(history, "observations", None)
        if observations is not None and len(observations) > 0:
            current_obs = observations[-1]
        if current_obs is None:
            return []
        tracked_objects = getattr(current_obs, "tracked_objects", None)
        tracked_objects_iter = getattr(tracked_objects, "tracked_objects", tracked_objects)
        if tracked_objects_iter is None:
            return []
        agents: List[SceneAgent] = []
        for obj in list(tracked_objects_iter):
            pose = getattr(obj, "center", None)
            velocity = getattr(obj, "velocity", None)
            box = getattr(obj, "box", getattr(obj, "oriented_box", None))
            metadata = getattr(obj, "metadata", None)
            tracked_object_type = getattr(getattr(obj, "tracked_object_type", None), "name", str(getattr(obj, "tracked_object_type", "UNKNOWN")))
            if pose is None:
                continue
            length = float(getattr(box, "length", 4.8)) if box is not None else 4.8
            width = float(getattr(box, "width", 1.8)) if box is not None else 1.8
            agents.append(
                SceneAgent(
                    track_token=str(getattr(metadata, "track_token", getattr(obj, "token", len(agents)))),
                    track_id=int(getattr(metadata, "track_id", len(agents))),
                    object_type=str(tracked_object_type),
                    pose=SE2(float(getattr(pose, "x", 0.0)), float(getattr(pose, "y", 0.0)), float(getattr(pose, "heading", 0.0))),
                    dynamic=DynamicState(
                        vx=float(getattr(velocity, "x", 0.0)),
                        vy=float(getattr(velocity, "y", 0.0)),
                        ax=0.0,
                        ay=0.0,
                        yaw_rate=0.0,
                    ),
                    size=np.array([length, width], dtype=np.float64),
                )
            )
        return agents

    def _extract_traffic_lights(self, traffic_light_data: Optional[Iterable[Any]]) -> Dict[str, TrafficLightEntry]:
        result: Dict[str, TrafficLightEntry] = {}
        if traffic_light_data is None:
            return result
        for entry in traffic_light_data:
            lane_connector_id = str(getattr(entry, "lane_connector_id", getattr(entry, "lane_connector", "unknown")))
            status = getattr(getattr(entry, "status", None), "name", str(getattr(entry, "status", "UNKNOWN")))
            result[lane_connector_id] = TrafficLightEntry(
                lane_connector_id=lane_connector_id,
                status=status,
                timestamp_us=int(getattr(entry, "timestamp", getattr(entry, "timestamp_us", 0))),
                extra={"raw": entry},
            )
        return result

    def _extract_scenario_token(self, planner_input: Any, iteration_index: int) -> str:
        history = getattr(planner_input, "history", None)
        scenario = getattr(history, "scenario", None)
        if scenario is not None:
            token = getattr(scenario, "token", None)
            if token is not None:
                return str(token)
        return f"tick_{iteration_index}"
