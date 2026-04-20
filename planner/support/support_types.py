from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ANCHOR_TYPES = {
    "branch",
    "conflict",
    "merge",
    "stop",
    "PED_CROSS",
    "ONCOMING_TURN",
    "PARKED_BYPASS",
    "YIELD_ZONE",
}

PRECEDENCE_STATES = ["EGO_FIRST", "OTHER_FIRST"]
GAP_STATES = ["OPEN", "TIGHT", "CLOSED"]
TIME_BINS = ["BIN_0", "BIN_1", "BIN_2", "BIN_3", "BIN_4", "BIN_5"]
RELEASE_STATES = TIME_BINS + ["NEVER"]
OCCUPANCY_STATES = ["NONE"] + TIME_BINS
BRANCH_STATES = ["CONFLICTING_BRANCH", "NONCONFLICTING_BRANCH", "UNKNOWN_BRANCH"]


@dataclass(frozen=True)
class LocalState:
    precedence: Optional[str] = None
    gap_state: Optional[str] = None
    release: Optional[str] = None
    occupancy: Optional[str] = None
    branch: Optional[str] = None
    active: bool = True

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "precedence": self.precedence,
            "gap_state": self.gap_state,
            "release": self.release,
            "occupancy": self.occupancy,
            "branch": self.branch,
            "active": self.active,
        }

    def label(self) -> str:
        parts = [
            self.precedence or "-",
            self.gap_state or "-",
            self.release or "-",
            self.occupancy or "-",
            self.branch or "-",
            "A" if self.active else "I",
        ]
        return "|".join(parts)


@dataclass
class Anchor:
    anchor_id: str
    base_anchor_id: str
    anchor_type: str
    ego_s: float
    ego_t_nominal: float
    geometry: np.ndarray
    agent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    criticality: float = 0.0


@dataclass
class Atom:
    atom_id: str
    assignments: Dict[str, LocalState]
    active_anchor_ids: List[str]
    prior_logit: float = 0.0

    def label(self) -> str:
        return ";".join(f"{aid}:{state.label()}" for aid, state in sorted(self.assignments.items()))


@dataclass
class SupportFactorGraph:
    edges: List[Tuple[str, str]] = field(default_factory=list)
    edge_types: Dict[Tuple[str, str], str] = field(default_factory=dict)


@dataclass
class CompiledSupport:
    anchors: List[Anchor]
    local_domains: Dict[str, List[LocalState]]
    atoms: List[Atom]
    factor_graph: SupportFactorGraph


@dataclass
class SupportCompilerConfig:
    anchor_bin_dt_s: float
    max_total_anchors_per_action: int
    max_conflict_anchors: int
    max_merge_anchors: int
    max_stop_anchors: int
    max_interact_anchors: int
    compiler_beam_width_online: int
    max_atoms_per_action_online: int
