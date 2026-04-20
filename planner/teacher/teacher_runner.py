from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from planner.actions.generator import ActionLibraryGenerator
from planner.actions.refiner import ActionRefiner
from planner.actions.action_types import RefinedAction
from planner.evaluation.coarse_planner import CoarsePlanner
from planner.evaluation.cost_terms import PlannerCost
from planner.runtime.types import RuntimeContext
from planner.support.atom_compiler import AtomCompiler
from planner.teacher.omission_targets import expected_cost, omission_damage_targets


@dataclass
class TeacherActionResult:
    action: RefinedAction
    support: any
    rho: np.ndarray
    mu: np.ndarray
    J: float
    omission_damage: np.ndarray


class TeacherRunner:
    def __init__(self, config: dict):
        self.config = config
        self.coarse = CoarsePlanner(config)
        self.action_generator = ActionLibraryGenerator(config)
        self.refiner = ActionRefiner(config, self.coarse)
        self.compiler = AtomCompiler(config)
        self.cost = PlannerCost(config)

    def build_candidates(self, ctx: RuntimeContext) -> List[RefinedAction]:
        actions = self.action_generator.generate(ctx)
        refined = self.refiner.refine_actions(ctx, actions)
        return self._screen_rivals(refined)

    def evaluate(self, ctx: RuntimeContext) -> List[TeacherActionResult]:
        candidates = self.build_candidates(ctx)
        if not candidates:
            return []
        results: List[TeacherActionResult] = []
        for action in candidates:
            support = self.compiler.compile(ctx, action, mode="teacher")
            rho, mu = self._teacher_posterior_and_cost(ctx, action, support)
            J = expected_cost(rho, mu)
            results.append(TeacherActionResult(action=action, support=support, rho=rho, mu=mu, J=J, omission_damage=np.zeros_like(rho)))
        results.sort(key=lambda r: r.J)
        if len(results) >= 2:
            rival_cost = results[1].J
            results[0].omission_damage = omission_damage_targets(results[0].J, rival_cost, results[0].rho, results[0].mu)
        for i in range(1, len(results)):
            best_other = min(r.J for j, r in enumerate(results) if j != i)
            results[i].omission_damage = omission_damage_targets(results[i].J, best_other, results[i].rho, results[i].mu)
        return results

    def _teacher_posterior_and_cost(self, ctx: RuntimeContext, action: RefinedAction, support) -> tuple[np.ndarray, np.ndarray]:
        base_cost = self.cost.evaluate(ctx, action.token, action.refined_traj).total
        atom_penalties = []
        logits = []
        anchor_by_id = {anchor.anchor_id: anchor for anchor in support.anchors}
        for atom in support.atoms:
            penalty = 0.0
            logit = atom.prior_logit
            for aid, state in atom.assignments.items():
                anchor = anchor_by_id[aid]
                if anchor.anchor_type == "stop":
                    if state.release == "NEVER":
                        penalty += 2.0
                        logit += 0.4
                    elif state.release and state.release.startswith("BIN_"):
                        penalty += 0.2 * int(state.release.split("_")[-1])
                if anchor.anchor_type in {"conflict", "merge", "PED_CROSS", "ONCOMING_TURN", "YIELD_ZONE"}:
                    if state.precedence == "EGO_FIRST":
                        penalty += 0.8 * anchor.criticality
                        logit -= 0.2 * anchor.criticality
                    else:
                        penalty += 0.2 * anchor.criticality
                        logit += 0.1 * anchor.criticality
                    if state.gap_state == "CLOSED":
                        penalty += 1.0
                        logit += 0.2
                    elif state.gap_state == "TIGHT":
                        penalty += 0.5
                if anchor.anchor_type == "branch":
                    if state.branch == "NONCONFLICTING_BRANCH":
                        penalty -= 0.2
                        logit -= 0.05
                    elif state.branch == "CONFLICTING_BRANCH":
                        penalty += 0.2
                        logit += 0.05
            atom_penalties.append(base_cost + penalty)
            logits.append(logit - 0.15 * penalty)
        mu = np.asarray(atom_penalties, dtype=np.float64)
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - np.max(logits)
        rho = np.exp(logits)
        rho /= np.sum(rho) + 1e-8
        return rho, mu

    def _screen_rivals(self, refined: List[RefinedAction]) -> List[RefinedAction]:
        if not refined:
            return []
        best = refined[0]
        threshold = float(self.config["ranking"]["rival_gap_threshold"])
        max_rivals = int(self.config["ranking"].get("teacher_rival_limit", self.config["ranking"]["rivals_max"]))
        survivors = [a for a in refined if a.coarse_score - best.coarse_score <= threshold]
        survivors = survivors[: max_rivals + 1]
        conservative = [a for a in refined if a.is_conservative]
        if conservative and all(a.action_id != conservative[0].action_id for a in survivors):
            survivors.append(conservative[0])
        deduped: Dict[tuple[str, str], RefinedAction] = {}
        for action in survivors:
            key = (action.token.path_mode, action.token.speed_mode)
            if key not in deduped or action.coarse_score < deduped[key].coarse_score:
                deduped[key] = action
        return sorted(deduped.values(), key=lambda a: a.coarse_score)
