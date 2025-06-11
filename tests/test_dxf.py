"""Tests for parsing info from DXF files"""

import pandas as pd
import pytest
from elkcreek.dxf import (
    DXF_CACHE,
    build_and_cache_topo,
    extract_points,
    get_polylines,
    make_paths,
    query_entities,
)
from elkcreek.grid import Grid
from matplotlib.path import Path as MPath
from numpy.testing import assert_allclose as np_assert


@pytest.fixture(scope="module")
def dxf_file(data_dir):
    return data_dir / "ug_instrumentation.dxf"


@pytest.fixture(scope="class")
def prep_cache():
    DXF_CACHE["stuff"] = "things"
    yield
    DXF_CACHE.pop("stuff")


class TestExtractPoints:
    """Make sure can extract point data from a dxf"""

    def test_extract_points(self, dxf_file):
        """Make sure it's possible to extract points from a file"""
        pts = extract_points(dxf_file)

        assert isinstance(pts, pd.DataFrame)
        assert set(pts.columns) == {"x", "y", "z"}
        assert len(pts) == 284

        min_pts = [10392.126201, 4208.036948, -110]
        max_pts = [11962.156765, 5122.337269, 1000]
        mean_pts = [11163.638857, 4673.510913, 6.267606]

        stats = pts.describe()
        np_assert(stats.loc["min"], min_pts)
        np_assert(stats.loc["max"], max_pts)
        np_assert(stats.loc["mean"], mean_pts)


class TestQueryEntities:
    def test_query_entities_by_layer(self, dxf_file):
        out = query_entities(dxf_file, layers=["instrument_sites"])
        assert len(out) == 4

    def test_query_entities_by_type(self, dxf_file):
        out = query_entities(dxf_file, entity_types=["CIRCLE"])
        assert len(out) == 60

    def test_query_by_both(self, dxf_file):
        out = query_entities(dxf_file, entity_types=["LINE"], layers=["bpcs"])
        assert len(out) == 27


class TestMakePaths:
    def test_make_paths(self, dxf_file):
        pths = make_paths(dxf_file, layers=["instrument_sites"])
        assert len(pths) == 4
        for p in pths:
            assert isinstance(p, MPath)
            assert len(p) == 5  # There's 1 extra point because it's a closed polyline


class TestGetPolylines:
    def test_get_polylines(self, dxf_file):
        stuff = get_polylines(dxf_file, "test", layers=["instrument_sites"])
        assert len(stuff) == 4

    def test_cached_polylines(self, dxf_file, prep_cache):
        stuff = get_polylines(dxf_file, "stuff")
        assert stuff == "things"


class TestBuildTopo:
    lower_left = (10389, 4550)
    upper_right = (10531, 4749)
    spacing = 50

    def test_build_and_cache_topo(self, dxf_file):
        stuff = build_and_cache_topo(
            dxf_file,
            "grid",
            self.lower_left,
            self.upper_right,
            self.spacing,
            method="cubic",
        )
        assert isinstance(stuff, Grid)
        np_assert(stuff.data.shape, [4, 5])
        np_assert(
            stuff.data.mean(), -4160, rtol=0.001
        )  # This was a terrible test dataset XD

    def test_cached_build_and_cache_topo(self, dxf_file, prep_cache):
        stuff = build_and_cache_topo(
            dxf_file, "stuff", self.lower_left, self.upper_right, self.spacing
        )
        assert stuff == "things"
