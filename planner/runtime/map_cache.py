from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from planner.runtime.types import MapPolyline


@dataclass
class CachedMapCrop:
    key: Tuple[str, int, str]
    polylines: List[MapPolyline]


class MapCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int, str], CachedMapCrop] = {}

    def get(self, key: Tuple[str, int, str]) -> CachedMapCrop | None:
        return self._cache.get(key)

    def put(self, key: Tuple[str, int, str], polylines: List[MapPolyline]) -> None:
        self._cache[key] = CachedMapCrop(key=key, polylines=polylines)

    def extract_polylines(self, map_api: Any, ego_xy: np.ndarray, config: dict) -> List[MapPolyline]:
        # The nuPlan map API differs slightly across versions; this extractor intentionally
        # prefers graceful degradation over brittle hard-coded method calls.
        layers = [
            "LANE",
            "LANE_CONNECTOR",
            "ROADBLOCK",
            "ROADBLOCK_CONNECTOR",
            "STOP_LINE",
            "CROSSWALK",
            "INTERSECTION",
            "DRIVABLE_AREA",
        ]
        front = float(config["context"]["map_front_m"])
        back = float(config["context"]["map_back_m"])
        side = float(config["context"]["map_side_m"])
        center_x, center_y = float(ego_xy[0]), float(ego_xy[1])
        polylines: List[MapPolyline] = []
        if map_api is None:
            return polylines

        objects_by_layer: Dict[str, List[Any]] = {}
        candidate_calls = [
            ("get_proximal_map_objects", (center_x, center_y, front, back, side), {}),
            ("get_proximal_map_objects", ((center_x, center_y), front), {}),
        ]
        for method_name, args, kwargs in candidate_calls:
            method = getattr(map_api, method_name, None)
            if method is None:
                continue
            try:
                raw = method(*args, layers=layers, **kwargs)
                if isinstance(raw, dict):
                    objects_by_layer = {str(k): list(v) for k, v in raw.items()}
                    break
            except Exception:
                continue

        for layer_name, objects in objects_by_layer.items():
            for obj in objects:
                pts = self._extract_object_centerline(obj)
                if pts is None or len(pts) < 2:
                    continue
                polylines.append(
                    MapPolyline(
                        polyline_id=str(getattr(obj, "id", getattr(obj, "token", f"{layer_name}_{len(polylines)}"))),
                        layer=layer_name,
                        points=pts,
                        metadata={"source_object": obj},
                    )
                )
        return polylines

    def _extract_object_centerline(self, map_obj: Any) -> np.ndarray | None:
        baseline = getattr(map_obj, "baseline_path", None)
        for candidate in [baseline, map_obj]:
            if candidate is None:
                continue
            if hasattr(candidate, "discrete_path"):
                path = getattr(candidate, "discrete_path")
                pts = [
                    [float(getattr(p, "x", p[0])), float(getattr(p, "y", p[1]))]
                    for p in path
                ]
                if pts:
                    return np.asarray(pts, dtype=np.float64)
            if hasattr(candidate, "xyz"):
                pts = np.asarray(getattr(candidate, "xyz"), dtype=np.float64)
                if pts.ndim == 2 and pts.shape[1] >= 2:
                    return pts[:, :2]
            if hasattr(candidate, "coords"):
                pts = np.asarray(list(candidate.coords), dtype=np.float64)
                if pts.ndim == 2 and pts.shape[1] >= 2:
                    return pts[:, :2]
        return None
