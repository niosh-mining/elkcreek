"""
Module for working with dxf (autocad map) files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import ezdxf
import numpy as np
import pandas as pd
from ezdxf.document import Drawing
from ezdxf.entities import Circle, DXFEntity, Line, LWPolyline, Point, Polyline, Text
from ezdxf.layouts import Layout
from ezdxf.query import EntityQuery
from matplotlib.path import Path as MPath
from scipy.interpolate import griddata

from elkcreek.grid import Grid  # This isn't going to cause a circular import, right?

DXF_CACHE = {}

dxf_entities_type = Drawing | DXFEntity | Path | Layout | EntityQuery
dxf_entity_containers_type = Path | Drawing | Layout | EntityQuery


def build_and_cache_topo(
    dxf, name, lower_left, upper_right, spacing, method="nearest", **kwargs
):
    topo = DXF_CACHE.get(name, None)
    if topo is None:
        # Grab the elevation points
        pts = extract_points(dxf)

        topo = Grid(
            label=name,
            lower_left=lower_left,
            upper_right=upper_right,
            spacing=spacing,
        )

        # Impose the topo points on a grid
        grid_x, grid_y = np.mgrid[  # The square bracket is correct?
            topo.grid_points[0][0] : topo.grid_points[0][-1] : topo.header["num_gps"][0]
            * 1j,
            topo.grid_points[1][0] : topo.grid_points[1][-1] : topo.header["num_gps"][1]
            * 1j,
        ]
        # Interpolate over the grid
        nearest_grid = griddata(
            np.array(pts[["x", "y"]]),
            np.array(pts["z"]),
            (grid_x, grid_y),
            method="nearest",
        )
        if method == "nearest":
            topo_grid = nearest_grid
        else:
            topo_grid = griddata(
                np.array(pts[["x", "y"]]),
                np.array(pts["z"]),
                (grid_x, grid_y),
                method=method,
            )
            # Fill in any missing information
            mask = np.isnan(topo_grid)
            topo_grid[mask] = nearest_grid[mask]

        topo.data = np.ndarray.astype(topo_grid, np.float64)

        DXF_CACHE[name] = topo
    return topo


def get_polylines(path, name, **kwargs):
    """
    Extract the polylines from dxf

    Parameters
    ----------
    path
        Path to the dxf file
    name
        Key to assign to in DXF_CACHE (or pull from, if the file has already
        been parsed)

    Other Parameters
    ----------------
    See make_paths
    """
    data = DXF_CACHE.get(name, None)
    if data is None:
        data = make_paths(path, **kwargs)
        DXF_CACHE[name] = data
    return data


def make_paths(
    dxf: dxf_entities_type,
    entity_types: Sequence = None,
    layers: Sequence = None,
    unsupported_entity_behavior: str = "raise",
    **kwargs,
):
    """
    Function for converting applicable entities into matplotlib Paths

    Currently supports LINE, LWPOLYLINE, and POLYLINE entities

    Parameters
    ----------
    dxf
        The dxf object from which to extract the point information
    entity_types
        Restrict the search to these entity types
    layers
        Restrict the search to only entities in these layers
    unsupported_entity_behavior
        Behavior to apply if an unsupported entity type is encountered.
        Supported values include "raise" to raise a TypeError if such an entity
        is encountered or "ignore" to ignore the entity and move on.

    Other Parameters
    ----------------
    See seistats.conversions.convert_coords for additional kwargs

    Warnings
    --------
    This is currently experimental and has not been thoroughly tested and could
    change/move without warning
    """
    match dxf:
        case Path() | str():
            return make_paths(
                ezdxf.readfile(dxf),
                entity_types,
                layers,
                unsupported_entity_behavior=unsupported_entity_behavior,
                **kwargs,
            )
        case Drawing():
            return make_paths(
                dxf.modelspace(),
                entity_types,
                layers,
                unsupported_entity_behavior=unsupported_entity_behavior,
                **kwargs,
            )
        case Layout() | EntityQuery():
            # First query on the entity types and layers
            if (entity_types is not None) or (layers is not None):
                dxf = query_entities(dxf, entity_types or [], layers or [])
            paths = []
            for e in dxf:
                paths.append(make_paths(e, **kwargs))
            return paths
        case Line() | LWPolyline() | Polyline():
            verts = extract_points(
                dxf, unsupported_entity_behavior=unsupported_entity_behavior
            )
            codes = [MPath.LINETO] * len(verts)
            codes[0] = MPath.MOVETO
            if (isinstance(dxf, LWPolyline) and dxf.closed) or (
                isinstance(dxf, Polyline) and dxf.is_closed
            ):
                verts.loc[len(verts)] = [0, 0, 0]
                codes.append(MPath.CLOSEPOLY)
            return MPath(vertices=verts[["x", "y"]], codes=codes)
        case _:
            raise TypeError(
                f"Unsupported type for creating matplotlib Paths: {type(dxf).__name__}"
            )


def extract_points(
    dxf: dxf_entities_type,
    entity_types: Sequence = None,
    layers: Sequence = None,
    unsupported_entity_behavior: str = "raise",
) -> pd.DataFrame:
    """
    Extracts points from provided dxf inputs

    Parameters
    ----------
    dxf
        The dxf object from which to extract the point information
    entity_types
        Restrict the search to these entity types
    layers
        Restrict the search to only entities in these layers
    unsupported_entity_behavior
        If "raise", raise a TypeError if an unknown entity type is encountered.
        If "ignore", return an empty DataFrame.

    Returns
    -------
    The point information
    """

    def _unsupported():
        match unsupported_entity_behavior:
            case "raise":
                raise TypeError(
                    f"Unsupported type for extracting points: {type(dxf).__name__}"
                )
            case "ignore":
                return pd.DataFrame(columns=["x", "y", "z"], dtype="float64")
            case _:
                raise ValueError(
                    "Invalid unsupported_entity_behavior. Must be 'raise' or 'ignore'."
                )

    match dxf:
        case Path() | str():
            return extract_points(ezdxf.readfile(dxf), entity_types, layers)
        case Drawing():
            return extract_points(dxf.modelspace(), entity_types, layers)
        case Layout() | EntityQuery():
            # First query on the entity types and layers
            if (entity_types is not None) or (layers is not None):
                dxf = query_entities(dxf, entity_types, layers)
            points = []
            for e in dxf:
                points.append(extract_points(e))
            return pd.concat(points).reset_index(drop=True)
        case DXFEntity():
            # Unfortunately, the point information is stored differently on each entity type...
            match dxf:
                case LWPolyline():
                    points = pd.DataFrame(
                        dxf.get_points(format="xy"), columns=["x", "y"]
                    )
                    points["z"] = dxf.dxf.elevation
                    return points
                case Text():
                    return pd.DataFrame(
                        [dxf.get_placement()[1].xyz], columns=["x", "y", "z"]
                    )
                case Line():
                    return pd.DataFrame(
                        [dxf.dxf.start, dxf.dxf.end], columns=["x", "y", "z"]
                    )
                case Polyline():
                    points = []
                    for v in dxf.vertices:
                        points.append(v.dxf.location)
                    return pd.DataFrame(points, columns=["x", "y", "z"])
                case Point():
                    return pd.DataFrame([dxf.dxf.location], columns=["x", "y", "z"])
                case Circle():
                    return pd.DataFrame([dxf.dxf.center], columns=["x", "y", "z"])
                case _:
                    return _unsupported()
        case _:
            return _unsupported()


def query_entities(
    entities: dxf_entity_containers_type,
    entity_types: Sequence = None,
    layers: Sequence = None,
) -> EntityQuery:
    """
    Query for entities matching a particular entity type or layer

    Parameters
    ----------
    entities
        The entities to query
    entity_types
        Restrict the search to these entity types
    layers
        Restrict the search to only entities in these layers

    Notes
    -----
    ezdxf can actually get much crazier than this, but I don't necessarily
    think that's appropriate for my purposes (many of the possible queries seem
    somewhat spurious?)
    """
    if isinstance(entities, (str, Path)):
        entities = ezdxf.readfile(entities)
    if not hasattr(entities, "query"):
        raise AttributeError(f"{type(entities).__name__} does not support queries")
    else:
        query = _build_query(entity_types, layers)
    return entities.query(query)


def _build_query(
    entity_types: Sequence | None = None, layers: Sequence | None = None
) -> str:
    """Builds a query for ezdxf"""
    entity_types = entity_types or []
    layers = layers or []

    if isinstance(entity_types, str):
        entity_types = [entity_types]
    if isinstance(layers, str):
        layers = [layers]
    if not len(entity_types) and not len(layers):
        raise ValueError(
            "Either entity_types or layers must be specified for an entity query"
        )

    # First build the layer query
    match len(layers):
        case 1:
            layers = f"[layer=='{layers[0]}']"
        case x if x > 1:
            lqs = []
            for lay in layers:
                lqs.append(f"layer=='{lay}'")
            layers = "|".join(lqs)
            layers = f"[{layers}]"
        case _:
            layers = ""

    # Now build the entity type query (and add in the layers)
    match len(entity_types):
        case 0:
            eq = "*"
        case 1:
            eq = f"{entity_types[0]}"
        case _:
            eqs = []
            for et in entity_types:
                eqs.append(et)
            eq = " ".join(eqs)

    return f"{eq}{layers}"
