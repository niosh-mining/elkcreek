"""Plotting functions"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mplstereonet  # noqa
import numpy as np
import pandas as pd
import pyrocko
import seaborn as sns
from matplotlib import dates as mdates
from matplotlib.contour import QuadContourSet
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.patches import PathPatch, Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyrocko.moment_tensor import MomentTensor
from pyrocko.plot import beachball, mpl_color

from elkcreek.dxf import DXF_CACHE, build_and_cache_topo, get_polylines, make_paths
from elkcreek.grid import Grid
from elkcreek.mt import crack_decomposition, project

LABEL_MAPPING = {
    "event_status": "Event Status",
    "local_mag": "Local Magnitude",
    "moment_mag": "Moment Magnitude",
    "location_residual": "Location Error (m)",
    "moment_total": "Seismic Moment (N-m)",
    "sp_moment_ratio": "S-to-P Moment Ratio",
    "energy_p": "P-Wave Energy (J)",
    "energy_s": "S-Wave Energy (J)",
    "energy_total": "Radiated Energy (J)",
    "sp_energy_ratio": "S-to-P Energy Ratio",
    "apparent_stress": "Apparent Stress (MPa)",
    "apparent_volume": "Apparent Volume ($m^3$)",
    "corner_frequency": "Corner Frequency (Hz)",
    "source_radius": "Source Radius (m)",
    "static_stress_drop": "Static Stress Drop (Pa)",
    "dynamic_stress_drop": "Dynamic Stress Drop (Pa)",
    "num_sensors_used": "Number of Sensors Used",
    "rupture_velocity": "Rupture Velocity (m/s?)",
}


def make_source_pairplot(
    cat: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "event_status",
    palette: sns.palettes._ColorPalette = None,
    hue_order: Iterable[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot two source parameters against each other

    Parameters
    ----------
    cat
        Event catalog
    x
        Parameter to plot on the x-axis
    y
        Parameter to plot on the y-axis
    hue
        Color the events according to this parameter
    palette
        Color palette to use
    hue_order
        The order that categories in hue should be plotted in

    Returns
    -------
    The matplotlib.Figure and matplotlib.Axes that the plot is created on
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    ax = sns.scatterplot(
        cat, x=x, y=y, hue=hue, alpha=0.1, palette=palette, hue_order=hue_order, ax=ax
    )
    # Grab the legend and fix the transparencies of all of the symbols to make
    # them more visible
    legend_fontsize = 8
    for child in ax.get_children():
        if isinstance(child, Legend):
            for line in child.get_lines():
                line.set_alpha(0.8)
            # Adjust the title of the legend
            txt = child.get_title()
            txt.set_text(LABEL_MAPPING[hue])
            txt.set_size(legend_fontsize)
            # Reduce the fontsize of the labels to 8 so they don't dwarf the
            # rest of the plot
            for txt in child.get_texts():
                txt.set_size(legend_fontsize)
            break
    # Set the axes labels to something nicer
    ax.set_xlabel(LABEL_MAPPING[ax.get_xlabel()])
    ax.set_ylabel(LABEL_MAPPING[ax.get_ylabel()])
    return fig, ax


# --- Plotting "grids" of data
def plot_overburden(
    dxf,
    contours,
    lower_left,
    upper_right,
    spacing,
    gridding_kwargs=None,
    plotting_kwargs=None,
):
    grid_params = {"method": "cubic"}
    grid_params.update(gridding_kwargs or {})
    ob = build_and_cache_topo(
        dxf, "overburden", lower_left, upper_right, spacing, **grid_params
    )
    plotting_kwargs = plotting_kwargs or {}
    return contourplot(grid=ob, levels=contours, **plotting_kwargs)


def contourplot(
    grid: Grid, levels: list[float] | list[int], **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a contour plot from a Grid of data
    """
    fig, ax = grid.plot(
        contour=True,
        colors=["darkolivegreen", "silver", "silver"],
        levels=levels,
        colorbar=False,
        **kwargs,
    )
    contours: QuadContourSet = [
        x for x in ax.get_children() if isinstance(x, QuadContourSet)
    ][0]
    for x in contours.labelTexts:
        x.set_text(int(float(x.get_text())))
    return fig, ax


# --- Plotting dxf elements
def _plot_patch(path, ax, legend_label=None, **kwargs):
    if legend_label is None:
        ppatch = PathPatch(path, **kwargs)
    else:
        ppatch = PathPatch(path, label=legend_label, **kwargs)
    ax.add_patch(ppatch)
    return


def _plot_paths(paths, ax, **kwargs):
    for path in paths:
        _plot_patch(path, ax=ax, **kwargs)
    return ax


def plot_workings(dxf, ax, facecolor="none", edgecolor="#1c1603", alpha=0.6, **kwargs):
    workings = get_polylines(dxf, "workings")
    _plot_paths(
        workings, ax=ax, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs
    )
    return ax


def plot_instrumentation_sites(
    dxf, ax, facecolor="none", edgecolor="#ffad01", lw=2, **kwargs
):
    sites = get_polylines(dxf, "instruments", layers=["instrument_sites"])
    _plot_paths(sites, ax=ax, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    return ax


def plot_faults(dxf, ax, facecolor="none", edgecolor="#224400", lw=1.5, **kwargs):
    faults = get_polylines(dxf, "faults")
    _plot_paths(
        faults, ax=ax, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs
    )
    return ax


def plot_burst(
    time,
    burst_event_dxf,
    burst_df,
    color,
    ax,
):
    def _get_cached_or_parse(key):
        data = DXF_CACHE.get(key, None)
        if data is None:
            data = make_paths(burst_event_dxf, layers=[key])
            DXF_CACHE[key] = data
        return data

    # Get & plot the face position
    date = time.split("T")[0]
    fp = _get_cached_or_parse(f"{date}_face")
    ax = _plot_paths(fp, ax=ax, edgecolor=color, facecolor="None", lw=3)

    # Get & plot the damaged area
    dz = _get_cached_or_parse(f"{date}_damage")
    ax = _plot_paths(dz, ax=ax, edgecolor=color, facecolor="None", lw=2.5)

    # Get and plot the event
    df = burst_df.loc[
        (burst_df["time"] > np.datetime64(time) - np.timedelta64(24, "h"))
        & (burst_df["time"] <= np.datetime64(time) + np.timedelta64(24, "h"))
    ]
    ax = plot_events(df, c=color, ax=ax, s=magnitude_scaling(df))

    return ax


# --- Plotting events and such


def plot_events(
    events: pd.DataFrame,
    ax: plt.Axes,
    s: float = 20,
    c: str = "red",
    marker: str = "o",
    edgecolors: str = "black",
    legend_label: str | None = None,
    colorbar: bool = False,
    colorbar_date: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Function for plotting event locations

    Parameters
    ----------
    events
        DataFrame containing the event locations. Required columns are "x" and
        "y"
    ax
        Axis to plot the events on
    s
        Symbol size
    c
        Symbol color
    marker
        Symbol type
    edgecolors
        Symbol edge color
    legend_label
      Label to assign the events in the legend
    colorbar
        Indicates whether to show a colorbar
    colorbar_date
        Indicates whether the colorbar is for datetimes

    Other Parameters
    ----------------
    See matplotlib.pyplot.scatter

    Returns
    -------
    Axes object containing the events
    """
    if colorbar_date:
        c = mdates.date2num(c)

    params = {
        "s": s,
        "c": c,
        "marker": marker,
        "edgecolors": edgecolors,
    }
    params.update(kwargs)

    if legend_label is not None:
        params["label"] = legend_label

    scat = ax.scatter(events.x, events.y, **params)

    if colorbar:
        cbar = plt.colorbar(scat, ax=ax)
        if colorbar_date:
            loc = mdates.AutoDateLocator()
            cbar.ax.yaxis.set_major_locator(loc)
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    return ax


def plot_stations(
    stations: pd.DataFrame,
    station_groups: dict[str, list],
    color_palette: sns.palettes._ColorPalette,
    ax: plt.Axes,
    s: float = 100,
    edgecolor: str = "black",
    legend: dict | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Function for plotting station locations

    Parameters
    ----------
    stations
        DataFrame containing station locations. Required columns are "x" and "y"
    station_groups
        Subsets of the stations to plot together
    color_palette
        Color palette to apply to the different station groupings
    ax
        Axis to plot the events on
    s
        Symbol size
    edgecolors
        Symbol edge color

    Other Parameters
    ----------------
    See matplotlib.pyplot.scatter

    Returns
    -------
    Axes object containing the stations
    """
    params = {
        "s": s,
        "edgecolor": edgecolor,
        "zorder": 20,
    }
    params.update(kwargs)

    for (label, sta_list), color in zip(station_groups.items(), color_palette.as_hex()):
        df = stations.loc[stations.name.isin(sta_list)]
        sur = df.loc[~df.underground]
        ax.scatter(sur["x"], sur["y"], c=color, marker="^", label=label, **params)
        ug = df.loc[df.underground]
        ax.scatter(ug["x"], ug["y"], c=color, marker="v", **params)

    if legend is not None:
        ax.legend(**legend)

    return ax


# --- More generic formatting stuff
def magnitude_scaling(events, col="local_mag"):
    """Create the dot size for plotting"""
    # Pretty arbitrary, but it seems to look nice at least for larger events
    return 2 ** (events[col] * 2) + 10


def plot_scale_bar(
    ax: plt.Axes,
    dist: float,
    unit: str,
    loc: str = "lower left",
    pad: float = 0.1,
    borderpad: float = 0.5,
    sep: float = 5,
    frameon: bool = False,
    font_properties: dict | None = None,
    label_override: bool = False,
    **kwargs,
):
    """
    Plot a scale bar

    Parameters
    ----------
    ax
        Matplotlib Axes to add the scale bar to
    dist
        The length of the scale bar in plot coordinates
    unit
        The units of the scale bar. If label_override is False, this will be
        tacked onto the distance specified by dist (ex., if dist=100 and
        unit="m" then the scale bar will say "100 m"). If label_override is
        True, then the scale bar will just show this value (ex., if dist=1000,
        unit="1 km", and label_override=True, the scale bar will just say
        "1 km").
    loc
        Location of the scale bar on the plot
    pad
        Padding of the bounding box around the scale bar in plot coordinates?
    borderpad
        The padding from the edge of the plot to the scale bar in plot coordinates?
    sep
        Vertical separation between the scale bar itself and the text label
    frameon
        If True, display a bbox around the scale bar
    font_properties
        Properties of the font for the text label
    label_override
        Indicates whether to override what is displayed in the text for the
        scale bar (see unit for more info)

    Other Parameters
    ----------------
    See mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar
    """
    if font_properties is not None:
        kwargs["fontproperties"] = FontProperties(**font_properties)
    asb = AnchoredSizeBar(
        ax.transData,
        dist,
        unit if label_override else f"{dist} {unit}",
        loc=loc,
        pad=pad,
        borderpad=borderpad,
        sep=sep,
        frameon=frameon,
        **kwargs,
    )
    ax.add_artist(asb)
    return asb


def set_extents(ax, extents):
    ax.set_xlim(extents["x"])
    ax.xaxis.set_visible(False)

    ax.set_ylim(extents["y"])
    ax.yaxis.set_visible(False)

    ax.set_aspect("equal")


@dataclass
class MomentTensorCDCPlotter:
    """Plot results of a model run."""

    best_event: pyrocko.model.Event
    event_ensemble: list[pyrocko.model.Event]
    crush_azimuth: float
    crush_plunge: float

    poisson_ratio = 0.25
    cmap: str = "viridis"
    figsize: tuple = (4, 4.5)

    def _get_fig_n_axis(self):
        """Get axis for plotting."""
        fig = plt.figure(figsize=self.figsize)
        ax_main = fig.add_subplot(1, 1, 1)
        ax_main.set_aspect("equal")

        return fig, ax_main

    def _draw_hudson_axes(
        self,
        ax: None | plt.Axes = None,
        color="black",
        fontsize=12,
        linewidth=1.5,
        alpha=0.75,
        label_source_type=True,
    ):
        """
        Plot axes and annotations of Hudson's MT decomposition diagram.
        """
        ax = plt.subplots(1, 1)[0] if ax is None else ax

        # Turn off bounding box and ticks but still allow labels.
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)

        ax.set_aspect(1.0)

        ax.set_xlim(-4.0 / 3.0 - 0.1, 4.0 / 3.0 + 0.1)
        ax.set_ylim(-1.14, 1.14)
        # Drawing outline
        x_args = [-4.0 / 3.0, 0.0, 4.0 / 3.0, 0.0, -4 / 3.0]
        y_args = [-1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0, -1.0 / 3.0]
        ax.plot(
            x_args,
            y_args,
            zorder=-1,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
        )

        ax.plot(
            [-1.0, 1.0],
            [0.0, 0.0],
            zorder=-1,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
        )

        ax.plot(
            [0.0, 0.0],
            [-1.0, 1.0],
            zorder=-1,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
        )

        d = fontsize / 3.0
        for txt, pos, off, va, ha in [
            ("+ISO", (0.0, 1.0), (-d, d), "bottom", "center"),
            ("-ISO", (0.0, -1.0), (d, -d), "top", "center"),
            ("-CLVD", (+1.0, 0.0), (d, -d), "top", "left"),
            ("+CLVD", (-1.0, 0.0), (-d, d), "bottom", "right"),
        ]:
            ax.plot(
                pos[0], pos[1], "o", color=color, alpha=alpha, markersize=fontsize / 4.0
            )
            if label_source_type:
                ax.annotate(
                    txt,
                    xy=pos,
                    xycoords="data",
                    xytext=off,
                    textcoords="offset points",
                    verticalalignment=va,
                    horizontalalignment=ha,
                    alpha=alpha,
                    rotation=0.0,
                )
        for txt, pos, off, va, ha in [
            ("-Dipole", (2.0 / 3.0, -1.0 / 3.0), (d, -d), "top", "left"),
            ("+Dipole", (-2.0 / 3.0, 1.0 / 3.0), (-d, d), "bottom", "right"),
            ("-Crack", (4.0 / 9.0, -5.0 / 9.0), (d, -d), "top", "left"),
            ("+Crack", (-4.0 / 9.0, 5.0 / 9.0), (-d, d), "bottom", "right"),
        ]:
            ax.plot(
                pos[0], pos[1], "o", color=color, alpha=alpha, markersize=fontsize / 4.0
            )
            if label_source_type:
                ax.annotate(
                    txt,
                    xy=pos,
                    xycoords="data",
                    xytext=off,
                    textcoords="offset points",
                    verticalalignment=va,
                    horizontalalignment=ha,
                    alpha=alpha,
                    rotation=0.0,
                )

    def _draw_cdc_region(self, ax):
        """Draw the region on the hudson plot were CDC decomp works."""
        v = self.poisson_ratio
        uk = (2 / 3) * (1 - 2 * v) / (1 - v)
        vk = (-1 / 3) * (1 + v) / (1 - v)
        u1 = -(4 / 3) * (1 - 2 * v) / (1 - v)
        v1 = -(1 / 3) * (1 + v) / (1 - v)
        u2 = -(2 / 3) * (1 - 2 * v)
        v2 = -(2 / 3) * (1 + v)
        points = [(u1, v1), (u2, v2), (uk, vk), (0, 0)]
        polygon = Polygon(
            points,
            closed=True,
            facecolor="skyblue",
            edgecolor="black",
            alpha=0.3,
            linewidth=1.5,  # thicker border
            linestyle="--",
        )
        ax.add_patch(polygon)

    def plot_events_on_hudson(self, ax, mt, color, alpha=1, size=4):
        """Plot the docs on the hudson."""
        if isinstance(mt, pyrocko.model.Event):
            mt = mt.moment_tensor
        u, v = project(mt.m_east_north_up())
        ax.plot(u, v, marker="o", ms=size, color=color, alpha=alpha)
        return ax

    def get_ensemble_mts(self):
        """Return the ensemble mts plus dc moment tensors."""
        ensemble_mts = [x.moment_tensor for x in self.event_ensemble]

        ensemble_decomp = [self.mt_decompose(x) for x in ensemble_mts]

        ensemble_crush = [x[0] for x in ensemble_decomp]
        ensemble_dc = [x[1] for x in ensemble_decomp]

        return ensemble_mts, ensemble_crush, ensemble_dc

    def plot_hudson(self, ax, best, ensemble):
        """Plot the hudson diagram."""
        for sub in ensemble:
            self.plot_events_on_hudson(
                ax,
                sub,
                color="black",
                alpha=0.015,
                size=3.0,
            )

        self.plot_events_on_hudson(
            ax,
            best,
            color=mpl_color("scarletred2"),
            size=8,
        )

    def plot_fuzzy_mt(
        self, ax, best, ensemble, scale=1.10, color="black", position=(0, 0)
    ):
        """Plot the fuzzy moment tensors."""
        beachball.plot_fuzzy_beachball_mpl_pixmap(
            ensemble,
            ax,
            best,
            beachball_type="full",
            size=scale,
            size_units="data",
            position=position,
            color_t=color,
            edgecolor="black",
            best_color=mpl_color("scarletred2"),
            method="contourf",
        )
        ax.set_axis_off()

        return ax

    def mt_decompose(self, mt):
        """Perform the simple crush decomposition."""
        best_crush, best_dc = crack_decomposition(
            mt,
            azimuth=self.crush_azimuth,
            plunge=self.crush_plunge,
        )

        return best_crush, best_dc

    def plot_strike_dip_rake_text(self, ax, best_dc, position):
        """Plot the strike dip and rake angles on the axis."""

        def _get_sdr_text_string(sdr1):
            out = []
            for num, val in enumerate(sdr1):
                if num == 1:
                    out.append(f"{val:02d}")
                else:
                    out.append(f"{val:03d}")
            return " / ".join(out)

        attrs = ("strike", "dip", "rake")
        sdr1 = [int(np.round(getattr(best_dc, f"{x}1"))) for x in attrs]
        sdr2 = [int(np.round(getattr(best_dc, f"{x}2"))) for x in attrs]
        sdr_text_1 = _get_sdr_text_string(sdr1)
        sdr_text_2 = _get_sdr_text_string(sdr2)
        text = sdr_text_1 + "\n" + sdr_text_2
        ax.text(position[0], position[1], text, ha="center", va="top")
        return ax

    def plot_decomp_percentages(self, ax, best_mt, best_crush, best_dc, position):
        """Plot the percentages of decomposition."""

        def per_str(value):
            """Return a percent string of a ratio."""
            out = f"{int(np.round(value * 100)):d}%"
            return out

        decomp = best_mt.standard_decomposition()
        ratio_iso = decomp[0][1]
        ratio_dc = decomp[1][1]
        ratio_clvd = decomp[2][1]
        stand_decomp_str = (
            f"ISO:{per_str(ratio_iso)} "
            f"DC: {per_str(ratio_dc)} "
            f"CLVD: {per_str(ratio_clvd)}"
        )

        cdc_moment = best_dc.moment + best_crush.moment
        ratio_cdc_crush = best_crush.moment / cdc_moment
        # In case there is some non-DC component, need to decompose DC
        new_dc = MomentTensor(m=best_dc.standard_decomposition()[1][2])
        ratio_cdc_dc = new_dc.moment / cdc_moment

        cdc_str = f"Crack: {per_str(ratio_cdc_crush)} DC: {per_str(ratio_cdc_dc)}"
        # text = stand_decomp_str + "\n" + cdc_str
        text = cdc_str
        ax.text(position[0], position[1], text, ha="center", va="top", fontsize=10)

    def plot_subplot_labels(self, ax):
        """Add the labels for subplots."""
        ax.text(-1.4, 1.1, "(a)")
        ax.text(-1.3, -1.2, "(b)")
        ax.text(0.36, -1.2, "(c)")

    def __call__(self):
        """Create plots, return figure and axis"""
        fig, hud_ax = self._get_fig_n_axis()
        self._draw_hudson_axes(hud_ax)
        self._draw_cdc_region(hud_ax)

        best_mt = self.best_event.moment_tensor
        best_crush, best_dc = self.mt_decompose(best_mt)

        ensemble_mts, _, ensemble_dcs = self.get_ensemble_mts()
        self.plot_hudson(hud_ax, best_mt, ensemble_mts)

        # Expand y lims to make room for mt.
        left, right = -4.0 / 3.0 - 0.1, 4.0 / 3.0 + 0.1
        hud_ax.set_xlim(left, right)
        hud_ax.set_ylim(-2.4, 1.0)

        self.plot_fuzzy_mt(hud_ax, best_mt, ensemble_mts, position=(left / 1.8, -1.8))
        self.plot_fuzzy_mt(hud_ax, best_dc, ensemble_dcs, position=(right / 1.8, -1.8))

        self.plot_strike_dip_rake_text(hud_ax, best_dc, position=(right / 1.8, -2.4))
        self.plot_decomp_percentages(
            hud_ax, best_mt, best_crush, best_dc, position=(left / 1.8, -2.4)
        )
        self.plot_subplot_labels(hud_ax)

        plt.show()

        return fig
