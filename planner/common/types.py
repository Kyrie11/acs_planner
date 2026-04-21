from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class AgentType(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    GENERIC = "generic"
    STATIC = "static"


@dataclass(slots=True)
class EgoStateLite:
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    ax: float
    ay: float
    yaw_rate: float
    steering_angle: float
    timestamp_s: float


@dataclass(slots=True)
class AgentStateLite:
    track_id: str
    agent_type: AgentType
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    length: float = 4.5
    width: float = 2.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def speed(self) -> float:
        return float((self.vx ** 2 + self.vy ** 2) ** 0.5)


@dataclass(slots=True)
class Polyline:
    points: List[Tuple[float, float]]

    def is_empty(self) -> bool:
        return not self.points


@dataclass(slots=True)
class RouteBranch:
    branch_id: str
    centerline: Polyline
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrafficLightPermission:
    lane_connector_id: str
    status: str
    timestamp_s: float


@dataclass(slots=True)
class RuntimeContext:
    ego: EgoStateLite
    ego_history_01s: List[EgoStateLite]
    ego_history_02s: List[EgoStateLite]
    agents_all: List[AgentStateLite]
    interaction_agents: List[AgentStateLite]
    route_centerline: Polyline
    route_branches: List[RouteBranch]
    goal_progress_s: float
    traffic_lights: Dict[str, TrafficLightPermission]
    map_polylines: Dict[str, List[Polyline]]
    map_window: Tuple[float, float, float, float]
    raw: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.ego_history_01s:
            raise ValueError("ego_history_01s cannot be empty")
        if self.route_centerline.is_empty():
            raise ValueError("route_centerline cannot be empty")


def safe_slice(sequence: Sequence[Any], every: int) -> List[Any]:
    return list(sequence[::every])
