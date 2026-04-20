from __future__ import annotations

from typing import Iterable, List

from planner.actions.action_types import ActionCandidate, RefinedAction


CONSERVATIVE_PATH_DISALLOWED = {
    "LC_LEFT",
    "LC_RIGHT",
    "MERGE_LEFT",
    "MERGE_RIGHT",
    "NUDGE_LEFT",
    "NUDGE_RIGHT",
}

CONSERVATIVE_SPEED_ALLOWED = {"FOLLOW", "DECEL", "STOP", "CREEP"}


def is_conservative(path_mode: str, speed_mode: str) -> bool:
    if path_mode in CONSERVATIVE_PATH_DISALLOWED:
        return False
    return speed_mode in CONSERVATIVE_SPEED_ALLOWED


def conservative_action_subset(actions: Iterable[RefinedAction]) -> List[RefinedAction]:
    return [a for a in actions if a.is_conservative]
