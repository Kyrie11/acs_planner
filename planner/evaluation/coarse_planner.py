from __future__ import annotations

from typing import List

from planner.actions.action_types import ActionToken
from planner.evaluation.cost_terms import PlannerCost
from planner.runtime.types import RuntimeContext


class CoarsePlanner:
    def __init__(self, config: dict):
        self.config = config
        self.cost = PlannerCost(config)

    def score_trajectory(self, ctx: RuntimeContext, token: ActionToken, traj: List) -> float:
        return self.cost.evaluate(ctx, token, traj, deterministic_only=True).total

    def score_action(self, ctx: RuntimeContext, refined_action) -> float:
        return self.score_trajectory(ctx, refined_action.token, refined_action.refined_traj)
