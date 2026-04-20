from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from planner.common.geometry import DynamicState, SE2


@dataclass
class EgoHistoryFrame:
    pose: SE2
    dynamic: DynamicState
    tire_steering_angle: float
    time_s: float


@dataclass
class SceneAgent:
    track_token: str
    track_id: int
    object_type: str
    pose: SE2
    dynamic: DynamicState
    size: np.ndarray
    ttc: float = float("inf")
    score: float = 0.0
    lane_id: Optional[str] = None
    route_corridor: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapPolyline:
    polyline_id: str
    layer: str
    points: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteBranch:
    branch_id: str
    connector_id: Optional[str]
    centerline: np.ndarray
    route_consistent: bool = True


@dataclass
class RouteInfo:
    route_centerline: np.ndarray
    route_lane_graph: Dict[str, List[str]]
    route_branches: List[RouteBranch]
    goal_progress_s: float
    current_lane_id: Optional[str] = None
    reference_speed_limit_mps: float = 13.0


@dataclass
class TrafficLightEntry:
    lane_connector_id: str
    status: str
    timestamp_us: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeContext:
    scenario_token: str
    iteration_index: int
    ego_state: EgoHistoryFrame
    ego_history: List[EgoHistoryFrame]
    agents_all: List[SceneAgent]
    agents_interaction: List[SceneAgent]
    map_polylines: List[MapPolyline]
    route_info: RouteInfo
    traffic_lights: Dict[str, TrafficLightEntry]
    mission_goal: Any
    route_roadblock_ids: List[str]
    map_api: Any
    history_buffer: Any
    raw_planner_input: Any
    raw_initialization: Any
    config: Dict[str, Any]

    @property
    def ego_pose_array(self) -> np.ndarray:
        return np.array([self.ego_state.pose.x, self.ego_state.pose.y, self.ego_state.pose.heading], dtype=np.float64)
