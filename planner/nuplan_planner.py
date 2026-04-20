from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from planner.actions.generator import ActionLibraryGenerator
from planner.actions.refiner import ActionRefiner
from planner.common.config import load_yaml
from planner.common.geometry import TrajectorySample
from planner.common.logging_utils import setup_logger
from planner.common.nuplan_compat import (
    AbstractPlanner,
    DetectionsTracks,
    EgoState,
    InterpolatedTrajectory,
    PlannerInitialization,
    PlannerInput,
    StateSE2,
    StateVector2D,
    TimePoint,
    get_pacifica_parameters,
)
from planner.evaluation.certification import ConformalCalibrator, certify_winner
from planner.evaluation.coarse_planner import CoarsePlanner
from planner.evaluation.cost_terms import PlannerCost
from planner.evaluation.retained_evaluator import RetainedEvaluator
from planner.models.acs_model import ACSModel
from planner.runtime.context_builder import RuntimeContextBuilder
from planner.support.atom_compiler import AtomCompiler
from planner.teacher.residual_bank import ResidualBank
from planner.training.calibrate import load_calibrator
from planner.training.feature_utils import build_scene_action_atom_tensors, collate_tensor_dict

LOGGER = setup_logger(__name__)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default.yaml"


class ACSNuPlanPlanner(AbstractPlanner):
    requires_scenario: bool = False

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        residual_bank_path: str | None = None,
        calibrator_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.config = load_yaml(config_path or _default_config_path())
        self.device = torch.device(device)
        self.initialization: PlannerInitialization | None = None
        self.context_builder = RuntimeContextBuilder(self.config)
        self.coarse_planner = CoarsePlanner(self.config)
        self.action_generator = ActionLibraryGenerator(self.config)
        self.refiner = ActionRefiner(self.config, self.coarse_planner)
        self.compiler = AtomCompiler(self.config)
        self.cost = PlannerCost(self.config)
        self.retained_evaluator = RetainedEvaluator(self.config)
        self.model: ACSModel | None = None
        self.residual_bank: ResidualBank | None = None
        self.calibrator = ConformalCalibrator(quantile=0.0)
        self.vehicle_params = get_pacifica_parameters()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        if residual_bank_path:
            self.residual_bank = ResidualBank.load(residual_bank_path, self.config)
        if calibrator_path:
            self.calibrator = load_calibrator(calibrator_path)

    def name(self) -> str:
        return "acs_nuplan_planner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self.initialization = initialization

    def load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model = ACSModel(ckpt.get("config", self.config))
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        LOGGER.info("Loaded checkpoint from %s", checkpoint_path)

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input: PlannerInput):
        if self.initialization is None:
            raise RuntimeError("Planner must be initialized before compute_planner_trajectory")
        ctx = self.context_builder.build(current_input, self.initialization)
        actions = self.action_generator.generate(ctx)
        if not actions:
            return self._build_emergency_stop_trajectory(ctx)
        refined = self.refiner.refine_actions(ctx, actions)
        candidate_actions = self._screen_rivals(refined)
        if not candidate_actions:
            return self._build_emergency_stop_trajectory(ctx)

        support_by_action = {a.action_id: self.compiler.compile(ctx, a, mode="online") for a in candidate_actions}
        stats = self._score_action_atoms(ctx, candidate_actions, support_by_action)
        retained = self._global_topk_select(candidate_actions, support_by_action, stats)
        fast_scores, omission_upper = self._aggregate_fast_scores(candidate_actions, stats, retained)

        ranked = sorted(candidate_actions, key=lambda a: fast_scores.get(a.action_id, float("inf")))
        winner = ranked[0]
        best_cons = self.retained_evaluator.best_conservative(ranked, fast_scores)
        cert_scores, cert_radii = self._aggregate_cert_scores(candidate_actions, stats, retained, winner, best_cons)

        cert_result = certify_winner(
            winner.action_id,
            cert_scores=cert_scores,
            cert_radii=cert_radii,
            omission_upper=omission_upper,
        )
        if cert_result.passed:
            return self._pack_trajectory(winner.refined_traj, current_input)

        conservative_actions = [a for a in candidate_actions if a.is_conservative]
        if conservative_actions:
            conservative_actions.sort(
                key=lambda a: cert_scores.get(a.action_id, fast_scores[a.action_id])
                + cert_radii.get(a.action_id, 0.0)
                + omission_upper.get(a.action_id, 0.0)
            )
            return self._pack_trajectory(conservative_actions[0].refined_traj, current_input)

        return self._build_emergency_stop_trajectory(ctx)

    def _screen_rivals(self, refined: Sequence) -> List:
        if not refined:
            return []
        best = refined[0]
        threshold = float(self.config["ranking"]["rival_gap_threshold"])
        max_rivals = int(self.config["ranking"]["rivals_max"])
        survivors = [a for a in refined if a.coarse_score - best.coarse_score <= threshold][: max_rivals + 1]
        conservative = [a for a in refined if a.is_conservative]
        if conservative and all(a.action_id != conservative[0].action_id for a in survivors):
            survivors.append(conservative[0])
        deduped = {}
        for action in survivors:
            key = (action.token.path_mode, action.token.speed_mode)
            if key not in deduped or action.coarse_score < deduped[key].coarse_score:
                deduped[key] = action
        return sorted(deduped.values(), key=lambda a: a.coarse_score)

    def _score_action_atoms(self, ctx, actions, support_by_action):
        stats: Dict[str, Dict[str, np.ndarray]] = {}
        if self.model is None:
            for action in actions:
                support = support_by_action[action.action_id]
                rho, damage, mu = self._heuristic_stats(ctx, action, support)
                stats[action.action_id] = {"rho": rho, "damage": damage, "mu_fast": mu, "mu_cert": mu.copy()}
            return stats

        batch_tensors = []
        mapping: List[Tuple[str, int]] = []
        for action in actions:
            support = support_by_action[action.action_id]
            for atom_idx, atom in enumerate(support.atoms):
                batch_tensors.append(build_scene_action_atom_tensors(ctx, action, atom, support, self.config))
                mapping.append((action.action_id, atom_idx))
        if not batch_tensors:
            return stats
        batch = collate_tensor_dict(batch_tensors)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        residual_bank_embeddings = None
        if self.residual_bank is not None:
            residual_bank_embeddings = self.residual_bank.bucket_embeddings("default").to(self.device)
        outputs = self.model(batch, residual_bank_embeddings=residual_bank_embeddings)
        rho_logits = outputs["rho_logits"].detach().cpu().numpy()
        damage = outputs["damage"].detach().cpu().numpy()
        mu_fast = outputs["mu_mean"].detach().cpu().numpy()
        mu_cert = mu_fast.copy()
        if residual_bank_embeddings is not None and "residual_logits" in outputs:
            residual_logits = outputs["residual_logits"].detach().cpu().numpy()
            topk = min(int(self.config["residual"]["mc_samples_for_certification"]), residual_logits.shape[-1])
            proto_vectors = residual_bank_embeddings.detach().cpu().numpy()
            for idx in range(len(mu_cert)):
                order = np.argsort(residual_logits[idx])[::-1][:topk]
                if len(order) > 0:
                    residual_penalty = np.mean(np.linalg.norm(proto_vectors[order, :4], axis=-1))
                    mu_cert[idx] = mu_fast[idx] + 0.1 * residual_penalty
        grouped: Dict[str, Dict[str, List[float]]] = {}
        for row_idx, (action_id, _atom_idx) in enumerate(mapping):
            grouped.setdefault(action_id, {"rho_logits": [], "damage": [], "mu_fast": [], "mu_cert": []})
            grouped[action_id]["rho_logits"].append(float(rho_logits[row_idx]))
            grouped[action_id]["damage"].append(float(damage[row_idx]))
            grouped[action_id]["mu_fast"].append(float(mu_fast[row_idx]))
            grouped[action_id]["mu_cert"].append(float(mu_cert[row_idx]))
        for action_id, values in grouped.items():
            logits = np.asarray(values["rho_logits"], dtype=np.float64)
            logits = logits - np.max(logits)
            rho = np.exp(logits)
            rho /= np.sum(rho) + 1e-8
            stats[action_id] = {
                "rho": rho,
                "damage": np.asarray(values["damage"], dtype=np.float64),
                "mu_fast": np.asarray(values["mu_fast"], dtype=np.float64),
                "mu_cert": np.asarray(values["mu_cert"], dtype=np.float64),
            }
        return stats

    def _heuristic_stats(self, ctx, action, support):
        base_cost = self.cost.evaluate(ctx, action.token, action.refined_traj).total
        rho_logits = []
        damage = []
        mu = []
        anchor_by_id = {anchor.anchor_id: anchor for anchor in support.anchors}
        for atom in support.atoms:
            penalty = 0.0
            critical = 0.0
            for aid, state in atom.assignments.items():
                anchor = anchor_by_id[aid]
                critical += anchor.criticality
                if anchor.anchor_type == "stop" and state.release == "NEVER":
                    penalty += 2.0
                if anchor.anchor_type in {"conflict", "merge", "PED_CROSS", "ONCOMING_TURN", "YIELD_ZONE"}:
                    if state.precedence == "EGO_FIRST":
                        penalty += 0.7 * anchor.criticality
                    else:
                        penalty += 0.2 * anchor.criticality
                    if state.gap_state == "CLOSED":
                        penalty += 0.8
                if anchor.anchor_type == "branch" and state.branch == "CONFLICTING_BRANCH":
                    penalty += 0.2
            mu.append(base_cost + penalty)
            rho_logits.append(atom.prior_logit - 0.1 * penalty)
            damage.append(max(0.05, 0.2 * critical))
        logits = np.asarray(rho_logits, dtype=np.float64)
        if logits.size == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        logits = logits - np.max(logits)
        rho = np.exp(logits)
        rho /= np.sum(rho) + 1e-8
        return rho, np.asarray(damage, dtype=np.float64), np.asarray(mu, dtype=np.float64)

    def _global_topk_select(self, actions, support_by_action, stats):
        budget = int(self.config["ranking"]["global_topk_budget_online"])
        min_best = int(self.config["ranking"]["min_atoms_for_best_action"])
        min_best_cons = int(self.config["ranking"]["min_atoms_for_best_conservative_action"])
        guaranteed = {a.action_id: [] for a in actions}
        sorted_actions = sorted(actions, key=lambda a: a.coarse_score)
        if sorted_actions:
            aid = sorted_actions[0].action_id
            guaranteed[aid] = list(range(min(min_best, len(support_by_action[aid].atoms))))
        conservative = [a for a in sorted_actions if a.is_conservative]
        if conservative:
            aid = conservative[0].action_id
            guaranteed[aid] = sorted(set(guaranteed[aid] + list(range(min(min_best_cons, len(support_by_action[aid].atoms))))))
        for action in sorted_actions[1:]:
            if len(support_by_action[action.action_id].atoms) > 0 and not guaranteed[action.action_id]:
                guaranteed[action.action_id] = [0]
        chosen = {aid: set(indices) for aid, indices in guaranteed.items()}
        remaining = []
        for action in actions:
            aid = action.action_id
            score = stats.get(aid, {})
            rho = score.get("rho", np.zeros((0,), dtype=np.float64))
            damage = score.get("damage", np.zeros((0,), dtype=np.float64))
            retain = rho * damage
            for idx in range(len(retain)):
                if idx in chosen[aid]:
                    continue
                remaining.append((float(retain[idx]), aid, idx))
        remaining.sort(key=lambda item: item[0], reverse=True)
        used = sum(len(v) for v in chosen.values())
        for _, aid, idx in remaining:
            if used >= budget:
                break
            chosen[aid].add(idx)
            used += 1
        return {aid: sorted(list(indices)) for aid, indices in chosen.items()}

    def _aggregate_fast_scores(self, actions, stats, retained):
        fast_scores = {}
        omission_upper = {}
        L_max = float(self.config["cost"]["L_max"])
        for action in actions:
            aid = action.action_id
            rho = stats[aid]["rho"]
            mu = stats[aid]["mu_fast"]
            keep = retained.get(aid, [])
            if keep:
                keep_rho = rho[keep]
                keep_mu = mu[keep]
                fast_scores[aid] = float(np.sum(keep_rho * keep_mu))
            else:
                fast_scores[aid] = float("inf")
            omitted_mask = np.ones(len(rho), dtype=bool)
            omitted_mask[keep] = False
            omitted_mass = float(np.sum(rho[omitted_mask])) if len(rho) else 0.0
            omission_pred = float(np.sum(rho[omitted_mask] * stats[aid]["damage"][omitted_mask])) if np.any(omitted_mask) else 0.0
            omission_upper[aid] = self.calibrator.upper_bound(omission_pred) + L_max * omitted_mass
        return fast_scores, omission_upper

    def _aggregate_cert_scores(self, actions, stats, retained, winner, best_cons):
        cert_scores = {}
        cert_radii = {}
        n_mc = int(self.config["residual"]["mc_samples_for_certification"])
        certify_ids = {winner.action_id}
        if best_cons is not None:
            certify_ids.add(best_cons.action_id)
        for action in actions:
            aid = action.action_id
            rho = stats[aid]["rho"]
            mu = stats[aid]["mu_cert"] if aid in certify_ids else stats[aid]["mu_fast"]
            keep = retained.get(aid, [])
            if keep:
                keep_rho = rho[keep]
                keep_mu = mu[keep]
                cert_scores[aid], cert_radii[aid] = self.retained_evaluator.aggregate_cert(keep_rho, keep_mu, n_mc=n_mc)
            else:
                cert_scores[aid] = float("inf")
                cert_radii[aid] = float(self.config["cost"]["L_max"])
        return cert_scores, cert_radii

    def _pack_trajectory(self, traj: Sequence[TrajectorySample], current_input: PlannerInput):
        ego_states = list(getattr(current_input.history, "ego_states", []))
        if not ego_states:
            return InterpolatedTrajectory(list(traj))
        current = ego_states[-1]
        t0 = int(getattr(current, "time_us", 0))
        states = []
        for idx, p in enumerate(traj):
            time_point = TimePoint(int(t0 + idx * self.config["planner"]["output_dt_s"] * 1e6))
            try:
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(p.x, p.y, p.heading),
                    rear_axle_velocity_2d=StateVector2D(p.speed * math.cos(p.heading), p.speed * math.sin(p.heading)),
                    rear_axle_acceleration_2d=StateVector2D(p.accel * math.cos(p.heading), p.accel * math.sin(p.heading)),
                    tire_steering_angle=float(np.clip(p.curvature * 2.7, -0.5, 0.5)),
                    time_point=time_point,
                    vehicle_parameters=self.vehicle_params,
                    angular_vel=0.0,
                    angular_accel=0.0,
                )
            except Exception:
                state = current
            states.append(state)
        return InterpolatedTrajectory(states)

    def _build_emergency_stop_trajectory(self, ctx):
        dt = float(self.config["planner"]["output_dt_s"])
        horizon = float(self.config["planner"]["output_horizon_s"])
        max_brake = float(self.config["planner"]["max_emergency_brake_mps2"])
        steps = int(round(horizon / dt)) + 1
        v0 = max(ctx.ego_state.dynamic.vx, 0.0)
        speed = np.maximum(0.0, v0 - max_brake * np.arange(steps) * dt)
        dist = np.cumsum(speed) * dt
        x = ctx.ego_state.pose.x + np.cos(ctx.ego_state.pose.heading) * dist
        y = ctx.ego_state.pose.y + np.sin(ctx.ego_state.pose.heading) * dist
        traj = [
            TrajectorySample(
                x=float(x[i]),
                y=float(y[i]),
                heading=float(ctx.ego_state.pose.heading),
                speed=float(speed[i]),
                accel=float(-max_brake if speed[i] > 0 else 0.0),
                curvature=0.0,
                time_s=float(i * dt),
            )
            for i in range(steps)
        ]
        return self._pack_trajectory(traj, ctx.raw_planner_input)
