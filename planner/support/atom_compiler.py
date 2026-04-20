from __future__ import annotations

from typing import Dict, List, Tuple

from planner.actions.action_types import RefinedAction
from planner.runtime.types import RuntimeContext
from planner.support.anchor_extractor import AnchorExtractor
from planner.support.consistency import ConsistencyChecker
from planner.support.support_types import Anchor, Atom, CompiledSupport, SupportFactorGraph
from planner.support.variable_schema import VariableSchema


class AtomCompiler:
    def __init__(self, config: dict):
        self.config = config
        self.anchor_extractor = AnchorExtractor(config)
        self.variable_schema = VariableSchema()
        self.consistency = ConsistencyChecker(config)

    def compile(self, ctx: RuntimeContext, action: RefinedAction, mode: str = "online") -> CompiledSupport:
        beam_width, max_atoms = self._budget(mode)
        anchors = self.anchor_extractor.extract(ctx, action)
        anchor_dict = {a.anchor_id: a for a in anchors}
        local_domains = {
            a.anchor_id: self.variable_schema.domain_for_anchor(a, conservative=action.is_conservative)
            for a in anchors
        }
        atoms = self._enumerate_atoms(anchor_dict, local_domains, conservative=action.is_conservative, beam_width=beam_width, max_atoms=max_atoms)
        factor_graph = self._build_factor_graph(anchors)
        if not atoms:
            default_atom = Atom(
                atom_id=f"default_{action.action_id}",
                assignments={a.anchor_id: self.variable_schema.conservative_default(a) for a in anchors},
                active_anchor_ids=[a.anchor_id for a in anchors],
                prior_logit=-1.0,
            )
            atoms = [default_atom]
        return CompiledSupport(anchors=anchors, local_domains=local_domains, atoms=atoms, factor_graph=factor_graph)

    def _budget(self, mode: str) -> Tuple[int, int]:
        support_cfg = self.config["support"]
        if mode == "teacher":
            return int(support_cfg.get("compiler_beam_width_teacher", support_cfg["compiler_beam_width_online"])), int(support_cfg.get("max_atoms_per_action_teacher", support_cfg["max_atoms_per_action_online"]))
        return int(support_cfg["compiler_beam_width_online"]), int(support_cfg["max_atoms_per_action_online"])

    def _enumerate_atoms(self, anchors: Dict[str, Anchor], local_domains: Dict[str, List], conservative: bool, beam_width: int, max_atoms: int) -> List[Atom]:
        ordered = sorted(anchors.values(), key=lambda a: (-a.criticality, a.ego_s))
        beam: List[Tuple[Dict[str, object], float]] = [({}, 0.0)]
        for anchor in ordered:
            next_beam: List[Tuple[Dict[str, object], float]] = []
            domain = local_domains[anchor.anchor_id]
            for assignments, score in beam:
                for local_state in domain:
                    candidate = dict(assignments)
                    candidate[anchor.anchor_id] = local_state
                    if not self.consistency.valid_partial(anchors, candidate, conservative=conservative):
                        continue
                    local_score = score + self._prior_logit(anchor, local_state)
                    next_beam.append((candidate, local_score))
            next_beam.sort(key=lambda item: item[1], reverse=True)
            beam = next_beam[:beam_width]
            if not beam:
                break
        atoms: List[Atom] = []
        for idx, (assignments, score) in enumerate(beam[:max_atoms]):
            atoms.append(
                Atom(
                    atom_id=f"atom_{idx}",
                    assignments=assignments,
                    active_anchor_ids=[aid for aid, state in assignments.items() if state.active],
                    prior_logit=float(score),
                )
            )
        return atoms

    def _prior_logit(self, anchor: Anchor, local_state) -> float:
        score = 0.0
        if anchor.anchor_type == "stop" and local_state.release == "NEVER":
            score += 0.5
        if anchor.anchor_type in {"conflict", "merge", "PED_CROSS", "ONCOMING_TURN"}:
            if getattr(local_state, "precedence", None) == "OTHER_FIRST":
                score += 0.2
            if getattr(local_state, "precedence", None) == "EGO_FIRST":
                score += max(-0.3, 0.2 - 0.1 * anchor.criticality)
        if anchor.anchor_type == "merge":
            gap_state = getattr(local_state, "gap_state", None)
            score += {"OPEN": 0.2, "TIGHT": 0.0, "CLOSED": 0.1}.get(gap_state, 0.0)
        if anchor.anchor_type == "branch":
            score += {"CONFLICTING_BRANCH": 0.1, "NONCONFLICTING_BRANCH": -0.05, "UNKNOWN_BRANCH": 0.0}.get(getattr(local_state, "branch", None), 0.0)
        return score

    def _build_factor_graph(self, anchors: List[Anchor]) -> SupportFactorGraph:
        edges: List[Tuple[str, str]] = []
        edge_types: Dict[Tuple[str, str], str] = {}
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                if set(a.agent_ids) & set(b.agent_ids):
                    edges.append((a.anchor_id, b.anchor_id))
                    edge_types[(a.anchor_id, b.anchor_id)] = "same_agent"
                elif a.base_anchor_id == b.base_anchor_id:
                    edges.append((a.anchor_id, b.anchor_id))
                    edge_types[(a.anchor_id, b.anchor_id)] = "same_zone"
        return SupportFactorGraph(edges=edges, edge_types=edge_types)
