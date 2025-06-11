"""Tests for catalog-related things"""

import numpy as np
import pandas as pd
import pytest
from elkcreek.events import apparent_volume, filter_event_df, get_outliers
from numpy.testing import assert_allclose as np_assert


class TestApparentVolume:
    """Validate apparent volume calculations"""

    moments = np.array([4.43e9, 5.28e9])
    energies = np.array([46.8, 17.8])
    shear_mod = np.array([33.3e9] * 2)
    apparent_stresses = (
        np.array([3.49e-4, 1.11e-4]) * 10e5
    )  # just to piss me off... this *should* be 10e6, imo
    apparent_volumes = np.array([6.35e6, 2.38e7])

    def test_apparent_volume_from_apparent_stress(self):
        """Test apparent volumes are calculated correctly from apparent stress."""
        np_assert(
            apparent_volume(self.moments, apparent_stress=self.apparent_stresses),
            self.apparent_volumes,
            rtol=0.05,
        )

    def test_apparent_volume_from_energy_shear(self):
        """Test apparent volumes are calculated correctly shear energy."""
        np_assert(
            apparent_volume(
                moment=self.moments, energy=self.energies, shear_modulus=self.shear_mod
            ),
            self.apparent_volumes,
            rtol=0.05,
        )

    def test_from_apparent_stress_unnecessary_kwargs(self):
        """Test apparent stress with bad kwargs raises."""
        with pytest.raises(ValueError, match="Unused keyword arguments"):
            apparent_volume(
                moment=self.moments,
                apparent_stress=self.apparent_stresses,
                energies=self.energies,
                shear_mod=self.shear_mod,
            )

    def test_from_energy_shear_unnecessary_kwargs(self):
        """Test apparent stress with bad kwargs raises."""
        with pytest.raises(ValueError, match="Unused keyword arguments"):
            apparent_volume(
                moment=self.moments,
                energy=self.energies,
                shear_modulus=self.shear_mod,
                garbage="things",
            )

    def test_single_value_instead_of_iterable(self):
        """Ensure a single value works."""
        np_assert(
            apparent_volume(self.moments[0], apparent_stress=self.apparent_stresses[0]),
            self.apparent_volumes[0],
            rtol=0.05,
        )


class TestCatFiltering:
    """Make sure the catalog filter does what we think it does"""

    @pytest.fixture(scope="class")
    def bursts(self):
        return pd.DataFrame(
            [
                [np.datetime64("2010-02-03T07:23:56")],
            ],
            columns=["time"],
        )

    @pytest.fixture(scope="class")
    def filtered_catalog(self, data_dir, bursts):
        """A fixture to return the catalog once it has been filtered."""
        df = pd.read_csv(data_dir / "cat_for_filtering.csv", parse_dates=["time"])
        return filter_event_df(df, bursts)

    def test_cat_len(self, filtered_catalog):
        """There should be exactly 6 events that survived filtering"""
        assert len(filtered_catalog) == 6

    def test_surviving_events(self, filtered_catalog):
        """
        Four of the RockSigma events should have survived, and one each of the
        IMS events.
        """
        counts = filtered_catalog.event_status.value_counts()
        assert counts["RockSigma"] == 4
        assert counts["Auto"] == 1
        assert counts["Manual"] == 1


class TestOutliers:
    """Make sure outliers are correctly selected"""

    @pytest.fixture(scope="class")
    def outliers(self, data_dir):
        """A fixture to return the catalog once it has been filtered."""
        df = pd.read_csv(data_dir / "cat_for_filtering.csv")
        return get_outliers(df)

    def test_correct_number_of_outliers(self, outliers):
        """Make sure events got selected"""
        outlier_df = outliers[0]
        assert len(outlier_df) == 3

    def test_correct_outlier_categories(self, outliers):
        """Make sure the outliers are assigned the correct categories"""
        outlier_df = outliers[0]
        counts = outlier_df.event_status.value_counts()
        assert counts["High Apparent Stress"] == 1
        assert counts["High Apparent Volume"] == 1
        assert counts["Big Source Radius"] == 1

    def test_big_events(self, outliers):
        """
        There should be 3 big events
        """
        big_event_df = outliers[1]
        assert len(big_event_df) == 3
        assert (big_event_df["local_mag"] >= 2).all()
