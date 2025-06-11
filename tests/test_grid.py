"""Tests for the Grid class"""

from collections.abc import Sequence

import pytest
from elkcreek.grid import Grid
from numpy.testing import assert_allclose as np_assert


def check_grid_bounds(
    grid: Grid, ll: Sequence[float], ur: Sequence[float], num_gps: Sequence[int]
):
    """
    Make sure the grid boundaries are correct
    """
    for i in range(
        2
    ):  # Because it's a two-dimensional grid and I don't want to repeat myself...
        # Grid points
        assert len(grid.grid_points[i]) == num_gps[i]
        np_assert(grid.grid_points[i][0], ll[i])
        np_assert(grid.grid_points[i][-1], ur[i])

        # Grid map
        np_assert(grid.grid_map[i].shape, num_gps)
        np_assert(grid.grid_map[i].min(), ll[i])
        np_assert(grid.grid_map[i].max(), ur[i])


class TestGrid:
    lower_left = [0, 0]
    upper_right = [1073, 2115]
    upper_right_adjusted = [1100, 2200]
    spacing = 100
    num_gps = [12, 23]
    val = 3.6

    @pytest.fixture(scope="class")
    def grid(self):
        return Grid("test", self.lower_left, self.upper_right, self.spacing, self.val)

    def test_geometry(self, grid):
        """
        Make sure the bounds and values of a velocity model are correct when
        defining using number of grid points.
        """
        check_grid_bounds(
            grid, self.lower_left, self.upper_right_adjusted, self.num_gps
        )

    def test_header(self, grid):
        """Make sure the header information is reasonable"""
        assert set(grid.header.keys()) == {
            "num_cells",
            "num_gps",
            "spacing",
            "lower_left",
            "upper_right",
            "label",
        }
        assert grid.header["label"] == "test"

    def test_data(self, grid):
        """Make sure the data were set properly"""
        np_assert(grid.data.shape, self.num_gps)
        np_assert(grid.data, self.val)

    def test_bad_data(self):
        """The shape of the data must match the shape of the grid"""
        with pytest.raises(ValueError, match="must match number of grid points"):
            Grid("test", self.lower_left, self.upper_right, self.spacing, [[1, 2]])
