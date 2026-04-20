from __future__ import annotations

from typing import Dict, Iterable, List

from planner.support.support_types import Anchor, LocalState


class ConsistencyChecker:
    def __init__(self, config: dict):
        self.config = config

    def valid_partial(self, anchors: Dict[str, Anchor], assignments: Dict[str, LocalState], conservative: bool) -> bool:
        return (
            self._precedence_consistency(anchors, assignments)
            and self._merge_gap_consistency(anchors, assignments, conservative)
            and self._release_consistency(anchors, assignments)
            and self._branch_consistency(anchors, assignments)
            and self._same_agent_monotonicity(anchors, assignments)
        )

    def _precedence_consistency(self, anchors: Dict[str, Anchor], assignments: Dict[str, LocalState]) -> bool:
        for aid, state in assignments.items():
            anchor = anchors[aid]
            if anchor.anchor_type not in {"conflict", "merge", "PED_CROSS", "ONCOMING_TURN", "YIELD_ZONE"}:
                continue
            if state.precedence is None:
                continue
            occ = state.occupancy
            if occ is None:
                continue
            try:
                occ_bin = int(occ.split("_")[-1])
            except Exception:
                occ_bin = 0
            ego_bin = int(anchor.ego_t_nominal)
            if state.precedence == "OTHER_FIRST" and ego_bin <= occ_bin:
                return False
            if state.precedence == "EGO_FIRST" and ego_bin > occ_bin + 1:
                return False
        return True

    def _merge_gap_consistency(self, anchors: Dict[str, Anchor], assignments: Dict[str, LocalState], conservative: bool) -> bool:
        for aid, state in assignments.items():
            anchor = anchors[aid]
            if anchor.anchor_type != "merge":
                continue
            if state.gap_state == "CLOSED" and state.precedence == "EGO_FIRST":
                return False
            if conservative and state.gap_state == "TIGHT" and state.precedence == "EGO_FIRST":
                return False
        return True

    def _release_consistency(self, anchors: Dict[str, Anchor], assignments: Dict[str, LocalState]) -> bool:
        for aid, state in assignments.items():
            anchor = anchors[aid]
            if anchor.anchor_type != "stop":
                continue
            if state.release is None:
                continue
            if state.release == "NEVER":
                continue
            try:
                release_bin = int(state.release.split("_")[-1])
            except Exception:
                release_bin = 0
            if anchor.ego_t_nominal < release_bin:
                # If nominal arrival is before release, the trajectory must plausibly stop first.
                continue
        return True

    def _branch_consistency(self, anchors: Dict[str, Anchor], assignments: Dict[str, LocalState]) -> bool:
        for aid, state in assignments.items():
            anchor = anchors[aid]
            if anchor.anchor_type != "branch" or state.branch is None:
                continue
            if state.branch == "NONCONFLICTING_BRANCH":
                agent_ids = set(anchor.agent_ids)
                for other_aid, other_state in assignments.items():
                    if other_aid == aid:
                        continue
                    other_anchor = anchors[other_aid]
                    if set(other_anchor.agent_ids) & agent_ids and other_anchor.ego_s > anchor.ego_s:
                        if other_anchor.anchor_type in {"conflict", "merge", "PED_CROSS", "ONCOMING_TURN", "YIELD_ZONE"}:
                            if other_state.active:
                                return False
        return True

    def _same_agent_monotonicity(self, anchors: Dict[str, Anchor], assignments: Dict[str, LocalState]) -> bool:
        grouped: Dict[str, List[tuple[Anchor, LocalState]]] = {}
        for aid, state in assignments.items():
            anchor = anchors[aid]
            for agent_id in anchor.agent_ids:
                grouped.setdefault(agent_id, []).append((anchor, state))
        for _, pairs in grouped.items():
            pairs.sort(key=lambda item: item[0].ego_s)
            last_occ = -1
            for anchor, state in pairs:
                if state.occupancy and state.occupancy.startswith("BIN_"):
                    occ = int(state.occupancy.split("_")[-1])
                    if occ < last_occ:
                        return False
                    last_occ = occ
        return True
