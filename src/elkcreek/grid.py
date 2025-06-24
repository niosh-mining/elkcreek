from __future__ import annotations

from collections.abc import Sequence
from string import capwords

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d


class Grid:
    """
    A basic class for a 2D grid

    Parameters
    ----------
    label
        An identifier about what's in the grid
    lower_left
        The lower-left corner of the grid
    upper_right
        The upper-right corner of the grid
    data
        Value(s) to assign to the points in the grid. By default, all points will be assigned -99.

    Attributes
    ----------
    header
        Dictionary containing the grid's header information
    grid_points
        Listing of grid coordinates along each dimension
    grid_map
        Meshgrid mapping grid indices to physical space
    data
        Array containing the data values at each grid point
    """

    def __init__(
        self,
        label: str,
        lower_left: Sequence[float],
        upper_right: Sequence[float],
        spacing: float,
        data: NDArray | float = -99,
    ) -> None:
        # Recalculate the upper right based on the provided spacing (to make sure it can be evenly divided)
        lower_left = np.array(lower_left)
        upper_right = np.array(upper_right)
        num_cells = np.array(
            [int(np.ceil((u - l) / spacing)) for u, l in zip(upper_right, lower_left)]
        )
        num_gps = num_cells + 1
        upper_right = lower_left + spacing * num_cells
        self.header = {
            "num_cells": num_cells,
            "num_gps": num_gps,
            "spacing": spacing,
            "lower_left": lower_left,
            "upper_right": upper_right,
            "label": label,
        }

        lims = zip(lower_left, upper_right, num_gps)
        self.grid_points = [np.linspace(x[0], x[1], x[2]) for x in lims]
        self.grid_map = np.meshgrid(*self.grid_points, indexing="ij")
        if not hasattr(data, "__len__"):
            data = data * np.ones(num_gps)
        else:
            data = np.array(data)
        if not (data.shape == num_gps).all():
            raise ValueError(
                f"Data shape must match number of grid points. Expected {num_gps}, got {data.shape}."
            )
        self.data = data

    def plot(self, ax, **kwargs) -> plt.Axes:
        """
        Plot a 2D

        Parameters
        ----------
        ax
            The Axes to plot the grid on. (Would be good to add the ability to create a new figure... it would be relatively trivial)

        Other Parameters
        ----------------
        figsize
            Size of the figure. Default is (9, 9)
        cmap
            Colormap to use for displaying grid values. Default is "rainbow".
        alpha
            Transparency value for the colormap. Default is 0.5.
        legend_label
            Label to display on the colorbar. Default is the grid's gtype.
        shading
            Indicates whether to use shading when rendering. Acceptable values
            are "nearest for a solid color in each grid cell (default) or
            "gouraud" to apply Gouraud shading (see matplotlib docs).
        contour
            Flag to indicate whether to plot contour lines instead of a colormap

        Other parameters accepted by matplotlib.pyplot.pcolormesh

        Returns
        -------
        matplotlib Figure
            Figure containing the resultant plot axes
        matplotlib Axes
            Axes containing the resultant plot
        """
        return plot_grid(self, ax, **kwargs)


def plot_grid(
    grid: Grid,
    ax: plt.Axes,
    contour: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    cbar_ax: plt.Axes | str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a colormap of a grid

    Parameters
    ----------
    grid
        The Grid to plot
    ax
        The axes to put the plot on
    contour
        Flag indicating whether to plot contour lines instead of a colormap
        (default=False).
    flip_x
        If True, flip the limits on the x-axis to go from highest to lowest
    flip_y
        If True, flip the limits on the y-axis to go from highest to lowest
    cbar_ax
        Axes to add a colorbar to. If None, don't use a colorbar. If "same",
        then use the same axes as the gridplot itself. Otherwise, should be
        the actual Axes to add the colorbar to.

    Other Parameters
    ----------------
    cmap
        Colormap to use for displaying grid values. Default is "rainbow".
    alpha
        Transparency value for the colormap. Default is 0.5.
    cbar_params
        Parameters to pass to the colorbar

    If contour=False:
        shading
            Indicates whether to use shading when rendering. Acceptable values
            are "nearest" for a solid color in each grid cell (default) or
            "gouraud" to apply Gouraud shading (see matplotlib docs).
    If contour=True
        fmt
            The string formatting to apply to the contour labels. Default is "%2.2f".
        colors
            Hopefully a list of repeating colors? Note: Cannot specify both this and cmap!
        label_color
            The color to apply to the contour labels. Default is "k" (black).


    Other parameters accepted by matplotlib.pyplot.pcolormesh

    Returns
    -------
    matplotlib Figure
        Figure containing the resultant plot axes
    matplotlib Axes
        Axes containing the resultant plot
    """
    if "colors" in kwargs:
        if not contour:
            raise ValueError("'colors' is not a valid option if contour=False")
        if "cmap" in kwargs:
            raise ValueError("Cannot specify both 'colors' and 'cmap'")
    else:
        cmap = kwargs.pop("cmap", "rainbow")
        # This is getting a little uglier than I would like, but necessary for flexibility
        kwargs["cmap"] = cmap
    alpha = kwargs.pop("alpha", 0.5)
    shading = kwargs.pop("shading", "nearest")
    # Need to do this whether or not a colorbar is used because the kwargs are a little bit messy...
    cbar_params = {
            "orientation": "horizontal",
            "label": grid.header["label"],
        }
    cbar_params.update(kwargs.pop("cbar_params", {}))


    # Plot the grid with the appropriate user args
    if not contour:
        cmesh = ax.pcolormesh(
            grid.grid_map[0],
            grid.grid_map[1],
            grid.data,
            alpha=alpha,
            shading=shading,
            **kwargs,
        )
    else:
        fmt = kwargs.pop("fmt", "%2.2f")
        label_color = kwargs.pop("label_color", "k")
        cmesh = ax.contour(
            grid.grid_map[0], grid.grid_map[1], grid.data, alpha=alpha, **kwargs
        )
        ax.clabel(cmesh, colors=label_color, fmt=fmt)
    # Make the plot look nice and add a legend
    ax.xaxis.tick_top()
    if cbar_ax is not None:
        if cbar_ax == "same":
            plt.colorbar(cmesh, ax=ax, **cbar_params)
        else:
            plt.colorbar(cmesh, cax=cbar_ax, **cbar_params)

    # Make the plot look nice
    ax.set_aspect("equal")
    if flip_x:
        ax.set_xlim(ax.get_xlim()[::-1])
    if flip_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    return ax


def spatially_tabulate_data(
    df: pd.DataFrame,
    param: str,
    lower_left: Sequence[float],
    upper_right: Sequence[float],
    spacing: float,
    statistic: str = "sum",
    log: bool = False
) -> Grid:
    """
    Spatially tabulate data across a grid

    Parameters
    ----------
    df
        Input DataFrame
    param
        Column to tabulate
    lower_left
        Lower-left corner of the output Grid
    upper_right
        Upper-right corner of the output Grid
    spacing
        Spacing for the grid cells
    statistic, optional
        Statistic to use to tabulate the data. See scipy.stats.binned_statistic_2d.
    log
        Whether the output should be converted to log scale

    Returns
    -------
    Grid containing the tabulated information
    """
    grid = Grid(
        capwords(param.replace("_", " ")), lower_left, upper_right, spacing=spacing
    )
    df = df.loc[df[param].notnull()]
    gridded = binned_statistic_2d(
        df["x"],
        df["y"],
        df[param],
        statistic=statistic,
        bins=grid.header["num_gps"],
        range=[
            [lower_left[0] - spacing/2, upper_right[0] + spacing/2],
            [lower_left[1] - spacing/2, upper_right[1] + spacing/2],
        ],  # I'm pretty sure this is right and the type hint is wrong?
    )
    if log:
        grid.data = np.log10(gridded.statistic)  # Same comment, unless they changed something?
    else:
        grid.data = gridded.statistic
    return grid
