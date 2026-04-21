from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from acs_planner.actions.action_types import Action
from acs_planner.support.support_types import Atom, CompiledSupport


@dataclass(slots=True)
class SelectionConfig:
    global_topk_budget_online: int = 48
    min_atoms_for_best_action: int = 4
    min_atoms_for_best_conservative_action: int = 2
    min_atoms_for_other_rivals: int = 1


def retained_score(probability: float, omission_damage: float) -> float:
    return probability * omission_damage


def allocate_global_topk(
    actions: Sequence[Action],
    compiled: Mapping[str, CompiledSupport],
    rho_by_action: Mapping[str, Mapping[str, float]],
    damage_by_action: Mapping[str, Mapping[str, float]],
    best_action_id: str,
    best_conservative_action_id: str | None,
    cfg: SelectionConfig | None = None,
) -> dict[str, list[str]]:
    cfg = cfg or SelectionConfig()
    budget = cfg.global_topk_budget_online
    selected: dict[str, list[str]] = {a.action_id: [] for a in actions}

    def reserve(action_id: str | None, k: int) -> None:
        nonlocal budget
        if not action_id or budget <= 0:
            return
        atoms = compiled[action_id].atoms
        scored = sorted(
            atoms,
            key=lambda atom: retained_score(
                rho_by_action.get(action_id, {}).get(atom.atom_id, 0.0),
                damage_by_action.get(action_id, {}).get(atom.atom_id, 0.0),
            ),
            reverse=True,
        )
        for atom in scored[: min(k, budget)]:
            if atom.atom_id not in selected[action_id]:
                selected[action_id].append(atom.atom_id)
                budget -= 1

    reserve(best_action_id, cfg.min_atoms_for_best_action)
    reserve(best_conservative_action_id, cfg.min_atoms_for_best_conservative_action)

    for action in actions:
        if action.action_id in {best_action_id, best_conservative_action_id}:
            continue
        reserve(action.action_id, cfg.min_atoms_for_other_rivals)

    remaining = []
    for action in actions:
        for atom in compiled[action.action_id].atoms:
            if atom.atom_id in selected[action.action_id]:
                continue
            remaining.append(
                (
                    retained_score(
                        rho_by_action.get(action.action_id, {}).get(atom.atom_id, 0.0),
                        damage_by_action.get(action.action_id, {}).get(atom.atom_id, 0.0),
                    ),
                    action.action_id,
                    atom.atom_id,
                )
            )
    remaining.sort(reverse=True)
    for _, action_id, atom_id in remaining[:budget]:
        selected[action_id].append(atom_id)
    return selected
