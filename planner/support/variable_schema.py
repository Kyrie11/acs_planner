from __future__ import annotations

from typing import Dict, List

from planner.support.support_types import (
    BRANCH_STATES,
    GAP_STATES,
    OCCUPANCY_STATES,
    PRECEDENCE_STATES,
    RELEASE_STATES,
    Anchor,
    LocalState,
)


class VariableSchema:
    def domain_for_anchor(self, anchor: Anchor, conservative: bool = False) -> List[LocalState]:
        t = anchor.anchor_type
        if t == "branch":
            return [LocalState(branch=b) for b in BRANCH_STATES]
        if t == "conflict":
            return [LocalState(precedence=p, occupancy=o) for p in PRECEDENCE_STATES for o in OCCUPANCY_STATES[1:]]
        if t == "merge":
            states = [LocalState(precedence=p, gap_state=g, occupancy=o) for p in PRECEDENCE_STATES for g in GAP_STATES for o in OCCUPANCY_STATES[1:]]
            if conservative:
                states = [s for s in states if s.precedence != "EGO_FIRST" or s.gap_state == "OPEN"]
            return states
        if t == "stop":
            return [LocalState(release=r) for r in RELEASE_STATES]
        if t == "PED_CROSS":
            return [LocalState(precedence=p, occupancy=o, release=r) for p in PRECEDENCE_STATES for o in OCCUPANCY_STATES[1:] for r in RELEASE_STATES]
        if t == "ONCOMING_TURN":
            return [LocalState(precedence=p, occupancy=o) for p in PRECEDENCE_STATES for o in OCCUPANCY_STATES[1:]]
        if t == "PARKED_BYPASS":
            return [LocalState(gap_state=g, release=r) for g in GAP_STATES for r in RELEASE_STATES]
        if t == "YIELD_ZONE":
            return [LocalState(precedence=p, gap_state=g, release=r) for p in PRECEDENCE_STATES for g in GAP_STATES for r in RELEASE_STATES]
        return [LocalState(active=False)]

    def conservative_default(self, anchor: Anchor) -> LocalState:
        if anchor.anchor_type == "branch":
            return LocalState(branch="CONFLICTING_BRANCH")
        if anchor.anchor_type == "stop":
            return LocalState(release="NEVER")
        if anchor.anchor_type in {"conflict", "ONCOMING_TURN"}:
            return LocalState(precedence="OTHER_FIRST", occupancy="BIN_0")
        if anchor.anchor_type in {"merge", "PARKED_BYPASS", "YIELD_ZONE"}:
            return LocalState(precedence="OTHER_FIRST", gap_state="CLOSED", occupancy="BIN_0", release="NEVER")
        if anchor.anchor_type == "PED_CROSS":
            return LocalState(precedence="OTHER_FIRST", occupancy="BIN_0", release="NEVER")
        return LocalState(active=False)
