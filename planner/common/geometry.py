from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

EPS = 1e-6


@dataclass
class SE2:
    x: float
    y: float
    heading: float


@dataclass
class DynamicState:
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class Box2D:
    center_x: float
    center_y: float
    length: float
    width: float
    heading: float


@dataclass
class TrajectorySample:
    x: float
    y: float
    heading: float
    speed: float
    accel: float
    curvature: float
    time_s: float


@dataclass
class FrenetPoint:
    s: float
    d: float


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_to_local(origin: SE2, points_xy: np.ndarray) -> np.ndarray:
    rot = rotation_matrix(-origin.heading)
    shifted = points_xy - np.array([[origin.x, origin.y]], dtype=np.float64)
    return shifted @ rot.T


def transform_to_global(origin: SE2, local_xy: np.ndarray) -> np.ndarray:
    rot = rotation_matrix(origin.heading)
    return local_xy @ rot.T + np.array([[origin.x, origin.y]], dtype=np.float64)


def pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def cumulative_arc_length(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(points_xy, axis=0), axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])


def interpolate_polyline(points_xy: np.ndarray, num: int) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((num, 2), dtype=np.float64)
    if len(points_xy) == 1:
        return np.repeat(points_xy, repeats=num, axis=0)
    s = cumulative_arc_length(points_xy)
    target = np.linspace(0.0, float(s[-1]) if s[-1] > EPS else 1.0, num=num)
    x = np.interp(target, s if s[-1] > EPS else np.arange(len(points_xy)), points_xy[:, 0])
    y = np.interp(target, s if s[-1] > EPS else np.arange(len(points_xy)), points_xy[:, 1])
    return np.stack([x, y], axis=-1)


def compute_headings(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(points_xy) == 1:
        return np.zeros((1,), dtype=np.float64)
    diff = np.diff(points_xy, axis=0)
    headings = np.arctan2(diff[:, 1], diff[:, 0])
    headings = np.concatenate([headings, headings[-1:]], axis=0)
    return headings


def project_point_to_polyline(point_xy: np.ndarray, polyline_xy: np.ndarray) -> FrenetPoint:
    if len(polyline_xy) == 0:
        return FrenetPoint(0.0, 0.0)
    if len(polyline_xy) == 1:
        d = float(np.linalg.norm(point_xy - polyline_xy[0]))
        return FrenetPoint(0.0, d)
    best_s, best_d = 0.0, float("inf")
    arc = cumulative_arc_length(polyline_xy)
    for i in range(len(polyline_xy) - 1):
        a = polyline_xy[i]
        b = polyline_xy[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab)) + EPS
        t = float(np.clip(np.dot(point_xy - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        vec = point_xy - proj
        cross = ab[0] * (point_xy - a)[1] - ab[1] * (point_xy - a)[0]
        d = float(np.linalg.norm(vec)) * (1.0 if cross >= 0 else -1.0)
        s = float(arc[i] + t * np.linalg.norm(ab))
        if abs(d) < abs(best_d):
            best_s, best_d = s, d
    return FrenetPoint(best_s, best_d)


def frenet_to_cartesian(polyline_xy: np.ndarray, s_values: np.ndarray, d_values: np.ndarray) -> np.ndarray:
    if len(polyline_xy) == 0:
        return np.zeros((len(s_values), 2), dtype=np.float64)
    if len(polyline_xy) == 1:
        normal = np.array([[0.0, 1.0]], dtype=np.float64)
        return np.repeat(polyline_xy[:1], len(s_values), axis=0) + d_values[:, None] * normal
    arc = cumulative_arc_length(polyline_xy)
    headings = compute_headings(polyline_xy)
    xs = np.interp(s_values, arc, polyline_xy[:, 0])
    ys = np.interp(s_values, arc, polyline_xy[:, 1])
    hs = np.interp(s_values, arc, headings)
    normals = np.stack([-np.sin(hs), np.cos(hs)], axis=-1)
    return np.stack([xs, ys], axis=-1) + normals * d_values[:, None]


def quintic_blend(u: np.ndarray) -> np.ndarray:
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def smooth_lateral_transition(
    s_values: np.ndarray,
    start_s: float,
    duration_s: float,
    target_offset: float,
) -> np.ndarray:
    out = np.zeros_like(s_values, dtype=np.float64)
    if duration_s <= EPS:
        out[s_values >= start_s] = target_offset
        return out
    u = np.clip((s_values - start_s) / max(duration_s, EPS), 0.0, 1.0)
    out = target_offset * quintic_blend(u)
    out[s_values >= start_s + duration_s] = target_offset
    return out


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.zeros_like(values)
    grad = np.gradient(values, dt)
    return np.asarray(grad)


def curvature_from_xy(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 3:
        return np.zeros((len(points_xy),), dtype=np.float64)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy + EPS, 1.5)
    k = (dx * ddy - dy * ddx) / denom
    return np.nan_to_num(k)


def speed_from_xy(points_xy: np.ndarray, dt: float) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    vel = np.gradient(points_xy, dt, axis=0)
    return np.linalg.norm(vel, axis=-1)


def resample_trajectory(points_xy: np.ndarray, dt: float, horizon_s: float) -> List[TrajectorySample]:
    num = int(round(horizon_s / dt)) + 1
    interp = interpolate_polyline(points_xy, num)
    headings = compute_headings(interp)
    speeds = speed_from_xy(interp, dt)
    accel = finite_difference(speeds, dt)
    curvature = curvature_from_xy(interp)
    return [
        TrajectorySample(
            x=float(interp[i, 0]),
            y=float(interp[i, 1]),
            heading=float(headings[i]),
            speed=float(speeds[i]),
            accel=float(accel[i]),
            curvature=float(curvature[i]),
            time_s=float(i * dt),
        )
        for i in range(num)
    ]


def points_to_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def min_distance_between_polylines(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(np.min(pairwise_distances(a, b)))


def approximate_collision(
    ego_xy: np.ndarray,
    other_xy: np.ndarray,
    threshold_m: float = 2.5,
) -> bool:
    horizon = min(len(ego_xy), len(other_xy))
    if horizon == 0:
        return False
    distances = np.linalg.norm(ego_xy[:horizon] - other_xy[:horizon], axis=-1)
    return bool(np.min(distances) < threshold_m)


def ttc_along_tracks(ego_xy: np.ndarray, other_xy: np.ndarray, dt: float, threshold_m: float = 3.0) -> float:
    horizon = min(len(ego_xy), len(other_xy))
    if horizon == 0:
        return float("inf")
    distances = np.linalg.norm(ego_xy[:horizon] - other_xy[:horizon], axis=-1)
    below = np.where(distances < threshold_m)[0]
    return float(below[0] * dt) if len(below) else float("inf")


def straight_line_projection(origin: SE2, speed: float, dt: float, horizon_s: float) -> np.ndarray:
    steps = int(round(horizon_s / dt)) + 1
    times = np.arange(steps, dtype=np.float64) * dt
    dx = np.cos(origin.heading) * speed * times
    dy = np.sin(origin.heading) * speed * times
    return np.stack([origin.x + dx, origin.y + dy], axis=-1)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clip01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def batched_clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def exceed(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.abs(values)
    return np.maximum(values - threshold, 0.0) / max(threshold, EPS)


def first_index(values: Iterable[bool]) -> int | None:
    for i, v in enumerate(values):
        if v:
            return i
    return None
