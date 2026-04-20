from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from planner.common.geometry import SE2, compute_headings, cumulative_arc_length, interpolate_polyline, project_point_to_polyline
from planner.runtime.types import MapPolyline, RouteBranch, RouteInfo


class RouteBuilder:
    def __init__(self, config: dict):
        self.config = config

    def build(
        self,
        ego_pose: SE2,
        map_api: Any,
        route_roadblock_ids: List[str],
        map_polylines: List[MapPolyline],
        mission_goal: Any,
    ) -> RouteInfo:
        route_centerline = self._build_centerline_from_route(map_api, route_roadblock_ids)
        if len(route_centerline) < 2:
            route_centerline = self._build_centerline_from_map(ego_pose, map_polylines)
        if len(route_centerline) < 2:
            route_centerline = self._fallback_straight_centerline(ego_pose)
        current_lane_id = self._infer_current_lane_id(ego_pose, map_polylines)
        goal_progress_s = float(cumulative_arc_length(route_centerline)[-1]) if len(route_centerline) else 0.0
        route_branches = self._extract_branches(route_centerline, map_polylines)
        route_lane_graph = self._build_route_lane_graph(route_roadblock_ids)
        reference_speed_limit_mps = self._estimate_speed_limit(map_api, current_lane_id)
        return RouteInfo(
            route_centerline=route_centerline,
            route_lane_graph=route_lane_graph,
            route_branches=route_branches,
            goal_progress_s=goal_progress_s,
            current_lane_id=current_lane_id,
            reference_speed_limit_mps=reference_speed_limit_mps,
        )

    def _build_centerline_from_route(self, map_api: Any, route_roadblock_ids: List[str]) -> np.ndarray:
        if map_api is None or not route_roadblock_ids:
            return np.zeros((0, 2), dtype=np.float64)
        points: List[np.ndarray] = []
        for roadblock_id in route_roadblock_ids:
            for method_name in ["get_map_object", "get_map_object_by_id"]:
                method = getattr(map_api, method_name, None)
                if method is None:
                    continue
                try:
                    obj = method(roadblock_id, "ROADBLOCK")
                except Exception:
                    try:
                        obj = method(roadblock_id)
                    except Exception:
                        obj = None
                if obj is None:
                    continue
                poly = self._extract_centerline_from_object(obj)
                if poly is not None and len(poly) >= 2:
                    points.append(poly)
                    break
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        merged = np.concatenate(points, axis=0)
        return interpolate_polyline(merged, num=max(64, len(merged)))

    def _build_centerline_from_map(self, ego_pose: SE2, map_polylines: List[MapPolyline]) -> np.ndarray:
        if not map_polylines:
            return np.zeros((0, 2), dtype=np.float64)
        candidate_layers = {"LANE", "LANE_CONNECTOR"}
        candidates = [p for p in map_polylines if p.layer in candidate_layers and len(p.points) >= 2]
        if not candidates:
            return np.zeros((0, 2), dtype=np.float64)
        ego_xy = np.array([ego_pose.x, ego_pose.y], dtype=np.float64)
        candidates.sort(key=lambda p: float(np.min(np.linalg.norm(p.points - ego_xy[None, :], axis=-1))))
        return interpolate_polyline(candidates[0].points, num=96)

    def _fallback_straight_centerline(self, ego_pose: SE2) -> np.ndarray:
        dist = np.linspace(0.0, 90.0, num=96)
        pts = np.stack(
            [ego_pose.x + np.cos(ego_pose.heading) * dist, ego_pose.y + np.sin(ego_pose.heading) * dist],
            axis=-1,
        )
        return pts

    def _extract_centerline_from_object(self, obj: Any) -> np.ndarray | None:
        baseline = getattr(obj, "baseline_path", None)
        for candidate in [baseline, obj]:
            if candidate is None:
                continue
            if hasattr(candidate, "discrete_path"):
                pts = np.asarray([[p.x, p.y] for p in candidate.discrete_path], dtype=np.float64)
                if len(pts) >= 2:
                    return pts
            if hasattr(candidate, "coords"):
                pts = np.asarray(list(candidate.coords), dtype=np.float64)
                if len(pts) >= 2:
                    return pts[:, :2]
        return None

    def _infer_current_lane_id(self, ego_pose: SE2, map_polylines: List[MapPolyline]) -> Optional[str]:
        lane_polys = [p for p in map_polylines if p.layer in {"LANE", "LANE_CONNECTOR"}]
        if not lane_polys:
            return None
        ego_xy = np.array([ego_pose.x, ego_pose.y], dtype=np.float64)
        lane_polys.sort(key=lambda p: float(np.min(np.linalg.norm(p.points - ego_xy[None, :], axis=-1))))
        return lane_polys[0].polyline_id

    def _extract_branches(self, route_centerline: np.ndarray, map_polylines: List[MapPolyline]) -> List[RouteBranch]:
        branches: List[RouteBranch] = []
        lane_connectors = [p for p in map_polylines if p.layer == "LANE_CONNECTOR"]
        if not lane_connectors:
            return branches
        for i, poly in enumerate(lane_connectors[:4]):
            if len(route_centerline) == 0:
                continue
            d = float(np.min(np.linalg.norm(poly.points - route_centerline[-1][None, :], axis=-1)))
            if d < 35.0:
                branches.append(RouteBranch(branch_id=f"BRANCH_{i}", connector_id=poly.polyline_id, centerline=interpolate_polyline(poly.points, 64)))
        return branches

    def _build_route_lane_graph(self, route_roadblock_ids: List[str]) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for idx, rid in enumerate(route_roadblock_ids):
            if idx + 1 < len(route_roadblock_ids):
                graph.setdefault(rid, []).append(route_roadblock_ids[idx + 1])
            else:
                graph.setdefault(rid, [])
        return graph

    def _estimate_speed_limit(self, map_api: Any, lane_id: Optional[str]) -> float:
        if map_api is None or lane_id is None:
            return 13.0
        methods = ["get_map_object", "get_map_object_by_id"]
        for method_name in methods:
            method = getattr(map_api, method_name, None)
            if method is None:
                continue
            try:
                lane_obj = method(lane_id, "LANE")
            except Exception:
                lane_obj = None
            if lane_obj is None:
                continue
            speed = getattr(lane_obj, "speed_limit_mps", None)
            if speed is not None:
                return float(speed)
        return 13.0
