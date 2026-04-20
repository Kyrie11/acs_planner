from __future__ import annotations

"""Compatibility wrappers so the code remains importable without nuPlan installed.

The real implementation is used when nuPlan is present. Otherwise lightweight stubs
are injected to keep preprocessing/training modules import-safe during static checks.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

try:  # pragma: no cover - exercised in real nuPlan environment.
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
    from nuplan.common.maps.abstract_map import AbstractMap
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
    from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
except Exception:  # pragma: no cover - fallback for environments without nuPlan.
    EgoState = Any
    StateSE2 = Any
    StateVector2D = Any
    TimePoint = Any
    VehicleParameters = Any
    AbstractMap = Any
    TrafficLightStatusData = Any
    AbstractScenario = Any
    NuPlanScenarioBuilder = None
    ScenarioFilter = None
    SimulationHistoryBuffer = Any
    DetectionsTracks = Any
    AbstractPlanner = object

    @dataclass(frozen=True)
    class PlannerInitialization:
        route_roadblock_ids: List[str]
        mission_goal: Any
        map_api: Any

    @dataclass(frozen=True)
    class PlannerInput:
        iteration: Any
        history: Any
        traffic_light_data: Optional[List[Any]] = None

    class InterpolatedTrajectory(list):
        def __init__(self, trajectory: List[Any]):
            super().__init__(trajectory)

    def get_pacifica_parameters() -> Any:
        return None
