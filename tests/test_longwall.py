"""Tests for reading longwall information"""

import numpy as np
import pandas as pd
import pytest
from elkcreek.longwall import (
    compile_daily_face_positions,
    get_date_from_face_position,
    get_longwall_positions,
    read_longwall_df,
)
from numpy.testing import assert_allclose as np_assert


@pytest.fixture
def longwall_df(data_dir) -> pd.DataFrame:
    """Read a longwall dataframe"""
    return read_longwall_df(data_dir / "face_positions.csv")


class TestReadLongwallDF:
    """Tests for reading a longwall dataframe"""

    def test_longwall_df(self, longwall_df):
        """Make sure required columns are present and have the correct dtype"""
        expected_dtypes = {
            "headgate_x": np.float64,
            "headgate_y": np.float64,
            "tailgate_x": np.float64,
            "tailgate_y": np.float64,
            "panel": "str",
            "local_time": "datetime64[us]",
        }
        for col, dtype in expected_dtypes.items():
            assert longwall_df[col].dtype == dtype


class TestFacePositionInterpolation:
    """Tests for interpolating face positions"""

    def test_get_longwall_positions(self, longwall_df):
        """Make sure you can get the face position for a date"""
        times = pd.Series(
            {
                "during panel": "2011-06-10",
                "end of panel": "2011-06-24",
                "between panels": "2011-07-03",
            }
        ).astype("datetime64[s]")
        positions = get_longwall_positions(times, longwall_df)

        cols = ["headgate_x", "headgate_y", "tailgate_x", "tailgate_y"]
        # In the middle of a panel
        np_assert(
            positions.loc["during panel", cols].values,
            [12095.376, 4904.2076, 12024.587, 4662.6024],
        )
        # Between panels (within buffer period)
        np_assert(
            positions.loc["end of panel", cols].values,
            [12127.69, 4895.79, 12056.88, 4654.19],
        )
        # Between panels (outside of buffer period)
        positions.loc["between panels"].isnull().all()

    def test_compile_daily_face_positions(self, longwall_df):
        """Make sure you can interpolate daily face positions"""
        start_date = "2011-06-10"
        end_date = "2011-06-15"
        daily_positions = compile_daily_face_positions(
            start_date, end_date, longwall_df
        )

        assert len(daily_positions) == 5  # End date is not inclusive
        # The face should be moving!!!
        assert (
            not daily_positions[
                ["headgate_x", "headgate_y", "tailgate_x", "tailgate_y"]
            ]
            .duplicated()
            .any()
        )

    def test_get_date_from_face_position(self, longwall_df):
        """Make sure you can estimate a date from a face position"""
        face_pos = pd.DataFrame(
            [
                [12090, 4900, 12020, 4660],
                [10080, 5822, 10010, 5580],
            ],
            columns=["headgate_x", "headgate_y", "tailgate_x", "tailgate_y"],
        )

        start_date = "2011-06-01"
        end_date = "2011-07-15"
        daily_positions = compile_daily_face_positions(
            start_date, end_date, longwall_df
        )
        dates = get_date_from_face_position(face_pos, daily_positions)

        for d, ref in zip(dates, ["2011-06-09", "2011-07-11"]):
            assert d == np.datetime64(ref)
