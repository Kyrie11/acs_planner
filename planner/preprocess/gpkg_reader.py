from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from shapely import STRtree, from_wkb
from shapely.geometry.base import BaseGeometry

from acs_planner.preprocess.sqlite_reader import connect_readonly


@dataclass(slots=True)
class GPKGFeature:
    layer: str
    row_id: str
    geometry: BaseGeometry
    attrs: dict[str, Any]


def list_layers(gpkg_path: str | Path) -> list[str]:
    with connect_readonly(gpkg_path) as conn:
        rows = conn.execute("SELECT table_name FROM gpkg_contents ORDER BY table_name").fetchall()
        return [str(row["table_name"]) for row in rows]


def geometry_column(conn: sqlite3.Connection, layer: str) -> str:
    rows = conn.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
        (layer,),
    ).fetchall()
    if not rows:
        raise KeyError(f"Layer {layer} has no registered geometry column")
    return str(rows[0]["column_name"])


def layer_columns(conn: sqlite3.Connection, layer: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({layer})").fetchall()
    return [str(row["name"]) for row in rows]


def load_layer_features(
    gpkg_path: str | Path,
    layer: str,
    *,
    limit: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    extra_columns: Sequence[str] | None = None,
) -> list[GPKGFeature]:
    with connect_readonly(gpkg_path) as conn:
        geom_col = geometry_column(conn, layer)
        cols = layer_columns(conn, layer)
        extra_cols = [c for c in (extra_columns or cols) if c != geom_col]
        select_cols = ", ".join([geom_col] + extra_cols)
        sql = f"SELECT {select_cols} FROM {layer}"
        rows = conn.execute(sql).fetchmany(limit or 1_000_000)
        out: list[GPKGFeature] = []
        for row in rows:
            geom = from_wkb(bytes(row[geom_col]))
            if bbox is not None and not geom.bounds:
                continue
            if bbox is not None:
                minx, miny, maxx, maxy = geom.bounds
                if maxx < bbox[0] or maxy < bbox[1] or minx > bbox[2] or miny > bbox[3]:
                    continue
            attrs = {c: row[c] for c in extra_cols}
            row_id = str(attrs.get("fid", len(out)))
            out.append(GPKGFeature(layer=layer, row_id=row_id, geometry=geom, attrs=attrs))
        return out


@dataclass(slots=True)
class LayerIndex:
    layer: str
    features: list[GPKGFeature]
    tree: STRtree

    @classmethod
    def build(cls, features: list[GPKGFeature], layer: str) -> "LayerIndex":
        return cls(layer=layer, features=features, tree=STRtree([feat.geometry for feat in features]))

    def query(self, bbox_geometry: BaseGeometry) -> list[GPKGFeature]:
        hit_geoms = self.tree.query(bbox_geometry)
        geom_to_feature = {id(feat.geometry): feat for feat in self.features}
        out: list[GPKGFeature] = []
        for geom in hit_geoms:
            feat = geom_to_feature.get(id(geom))
            if feat is not None:
                out.append(feat)
        return out
