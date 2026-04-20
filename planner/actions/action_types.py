from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from planner.common.geometry import TrajectorySample

PATH_MODES = [
    "KEEP_ROUTE",
    "BRANCH_0",
    "BRANCH_1",
    "BRANCH_2",
    "BRANCH_3",
    "LC_LEFT",
    "LC_RIGHT",
    "MERGE_LEFT",
    "MERGE_RIGHT",
    "NUDGE_LEFT",
    "NUDGE_RIGHT",
]

SPEED_MODES = ["FOLLOW", "CRUISE", "DECEL", "STOP", "CREEP"]


@dataclass(frozen=True)
class ActionToken:
    path_mode: str
    speed_mode: str

    def signature(self) -> str:
        return f"{self.path_mode}__{self.speed_mode}"


@dataclass
class RefinementDomain:
    delta_v_terminal: List[float] = field(default_factory=list)
    delta_stop_offset: List[float] = field(default_factory=list)
    delta_time_shift: List[float] = field(default_factory=list)
    lc_start_delay: List[float] = field(default_factory=list)
    lc_duration: List[float] = field(default_factory=list)
    lateral_offset: List[float] = field(default_factory=list)
    creep_distance: List[float] = field(default_factory=list)
    creep_speed_cap: List[float] = field(default_factory=list)


@dataclass
class ActionCandidate:
    action_id: str
    token: ActionToken
    nominal_path: np.ndarray
    nominal_speed_profile: np.ndarray
    nominal_traj: List[TrajectorySample]
    refine_domain: RefinementDomain
    is_conservative: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinedAction:
    action: ActionCandidate
    refinement: Dict[str, float]
    refined_path: np.ndarray
    refined_speed_profile: np.ndarray
    refined_traj: List[TrajectorySample]
    coarse_score: float = float("inf")
    full_score: float = float("inf")
    support: Optional[Any] = None
    support_mass: Optional[np.ndarray] = None
    omission_score: Optional[np.ndarray] = None

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def token(self) -> ActionToken:
        return self.action.token

    @property
    def is_conservative(self) -> bool:
        return self.action.is_conservative

    @property
    def signature(self) -> str:
        return self.action.token.signature() + "::" + ",".join(f"{k}={v}" for k, v in sorted(self.refinement.items()))
