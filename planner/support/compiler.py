from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp
from typing import Callable, Iterable, List, Sequence

from acs_planner.actions.action_types import Action
from acs_planner.common.types import RuntimeContext
from acs_planner.support.consistency import ConsistencyConfig, is_consistent
from acs_planner.support.support_types import (
    Anchor,
    AnchorType,
    Atom,
    BranchChoice,
    CompiledSupport,
    GapState,
    LocalState,
    OccupancyBin,
    Precedence,
    ReleaseBin,
)


@dataclass(slots=True)
class CompilerConfig:
    max_total_anchors_per_action: int = 12
    max_conflict_anchors: int = 6
    max_merge_anchors: int = 2
    max_stop_anchors: int = 2
    max_interact_anchors: int = 2
    compiler_beam_width_online: int = 32
    max_atoms_per_action_online: int = 24


def enumerate_local_states(anchor: Anchor) -> list[LocalState]:
    state_space = anchor.state_space()
    keys = list(state_space.keys())
    values = [list(v) for v in state_space.values()]
    out: list[LocalState] = []
    for combo in product(*values):
        out.append(LocalState(values={k: str(v.value) for k, v in zip(keys, combo)}))
    return out


def default_local_state(anchor: Anchor) -> LocalState:
    if anchor.anchor_type == AnchorType.CONFLICT:
        return LocalState({"precedence": Precedence.OTHER_FIRST.value, "occupancy": OccupancyBin.BIN_0.value})
    if anchor.anchor_type == AnchorType.MERGE:
        return LocalState(
            {
                "precedence": Precedence.OTHER_FIRST.value,
                "gap_state": GapState.CLOSED.value,
                "occupancy": OccupancyBin.BIN_0.value,
            }
        )
    if anchor.anchor_type == AnchorType.STOP:
        return LocalState({"release": ReleaseBin.NEVER.value})
    if anchor.anchor_type == AnchorType.BRANCH:
        return LocalState({"branch": BranchChoice.CONFLICTING_BRANCH.value})
    if anchor.anchor_type == AnchorType.PED_CROSS:
        return LocalState(
            {
                "precedence": Precedence.OTHER_FIRST.value,
                "occupancy": OccupancyBin.BIN_0.value,
                "release": ReleaseBin.NEVER.value,
            }
        )
    if anchor.anchor_type == AnchorType.ONCOMING_TURN:
        return LocalState({"precedence": Precedence.OTHER_FIRST.value, "occupancy": OccupancyBin.BIN_0.value})
    if anchor.anchor_type == AnchorType.PARKED_BYPASS:
        return LocalState({"gap_state": GapState.CLOSED.value, "release": ReleaseBin.NEVER.value})
    if anchor.anchor_type == AnchorType.YIELD_ZONE:
        return LocalState(
            {
                "precedence": Precedence.OTHER_FIRST.value,
                "gap_state": GapState.TIGHT.value,
                "release": ReleaseBin.NEVER.value,
            }
        )
    raise KeyError(anchor.anchor_type)


def default_atom(anchors: Sequence[Anchor]) -> Atom:
    assignments = {anchor.anchor_id: default_local_state(anchor) for anchor in anchors}
    return Atom(atom_id="default", assignments=assignments, active_anchor_ids=list(assignments.keys()), prior_log_score=-5.0)


def sort_anchors_by_criticality(anchors: Sequence[Anchor]) -> list[Anchor]:
    priority = {
        AnchorType.STOP: 0,
        AnchorType.CONFLICT: 1,
        AnchorType.MERGE: 2,
        AnchorType.PED_CROSS: 3,
        AnchorType.BRANCH: 4,
        AnchorType.ONCOMING_TURN: 5,
        AnchorType.PARKED_BYPASS: 6,
        AnchorType.YIELD_ZONE: 7,
    }
    return sorted(anchors, key=lambda a: (priority.get(a.anchor_type, 99), a.ego_t_nominal, a.ego_s))


def _state_prior(anchor: Anchor, local_state: LocalState) -> float:
    score = 0.0
    if local_state.get("precedence") == Precedence.OTHER_FIRST.value:
        score += 0.25
    if local_state.get("release") == ReleaseBin.NEVER.value:
        score -= 0.5
    if local_state.get("gap_state") == GapState.CLOSED.value:
        score -= 0.25
    return score


class SupportCompiler:
    def __init__(
        self,
        cfg: CompilerConfig | None = None,
        consistency_cfg: ConsistencyConfig | None = None,
        prior_scorer: Callable[[Anchor, LocalState], float] | None = None,
    ) -> None:
        self.cfg = cfg or CompilerConfig()
        self.consistency_cfg = consistency_cfg or ConsistencyConfig()
        self.prior_scorer = prior_scorer or _state_prior

    def compile(
        self,
        ctx: RuntimeContext,
        action: Action,
        anchors: Sequence[Anchor],
    ) -> CompiledSupport:
        ordered = sort_anchors_by_criticality(list(anchors))[: self.cfg.max_total_anchors_per_action]
        local_domains = {anchor.anchor_id: enumerate_local_states(anchor) for anchor in ordered}
        beams: list[tuple[dict[str, LocalState], float]] = [({}, 0.0)]

        for anchor in ordered:
            next_beams: list[tuple[dict[str, LocalState], float]] = []
            for assignments, score in beams:
                for local_state in local_domains[anchor.anchor_id]:
                    new_assignments = dict(assignments)
                    new_assignments[anchor.anchor_id] = local_state
                    if not is_consistent(
                        ordered,
                        new_assignments,
                        is_conservative_action=action.is_conservative,
                        action_family=action.speed_mode.value,
                        cfg=self.consistency_cfg,
                    ):
                        continue
                    local_score = score + self.prior_scorer(anchor, local_state)
                    next_beams.append((new_assignments, local_score))
            next_beams.sort(key=lambda item: item[1], reverse=True)
            beams = next_beams[: self.cfg.compiler_beam_width_online]
            if not beams:
                break

        atoms: list[Atom] = []
        for i, (assignments, score) in enumerate(beams[: self.cfg.max_atoms_per_action_online]):
            atoms.append(
                Atom(
                    atom_id=f"{action.action_id}_tau_{i}",
                    assignments=assignments,
                    active_anchor_ids=list(assignments.keys()),
                    prior_log_score=score,
                )
            )
        if not atoms:
            atoms = [default_atom(ordered)]

        factor_graph = self._build_factor_graph(ordered)
        return CompiledSupport(
            anchors=list(ordered),
            local_domains=local_domains,
            atoms=atoms,
            factor_graph=factor_graph,
            metadata={"action_id": action.action_id, "num_atoms": len(atoms)},
        )

    @staticmethod
    def _build_factor_graph(anchors: Sequence[Anchor]) -> dict[str, list[str]]:
        edges: dict[str, list[str]] = {anchor.anchor_id: [] for anchor in anchors}
        for i, anchor in enumerate(anchors):
            for j in range(i + 1, len(anchors)):
                other = anchors[j]
                if set(anchor.agent_ids) & set(other.agent_ids):
                    edges[anchor.anchor_id].append(other.anchor_id)
                    edges[other.anchor_id].append(anchor.anchor_id)
                    continue
                if anchor.metadata.get("zone_id") == other.metadata.get("zone_id"):
                    edges[anchor.anchor_id].append(other.anchor_id)
                    edges[other.anchor_id].append(anchor.anchor_id)
        return edges
