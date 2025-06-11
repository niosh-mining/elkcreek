from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from numpy.typing import NDArray


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

    def plot(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a 2D grid

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
        return plot_grid(self, **kwargs)


def plot_grid(
    grid: Grid,
    contour: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a colormap of a grid

    Parameters
    ----------
    grid
        The Grid to plot
    contour
        Flag indicating whether to plot contour lines instead of a colormap
        (default=False).
    flip_x
        If True, flip the limits on the x-axis to go from highest to lowest
    flip_y
        If True, flip the limits on the y-axis to go from highest to lowest

    Other Parameters
    ----------------
    figsize
        Size of the figure. Default is (9, 9)
    cmap
        Colormap to use for displaying grid values. Default is "rainbow".
    alpha
        Transparency value for the colormap. Default is 0.5.
    legend_label
        Label to display on the colorbar. Default is "Value".

    If contour=False:
        shading
            Indicates whether to use shading when rendering. Acceptable values
            are "nearest" for a solid color in each grid cell (default) or
            "gouraud" to apply Gouraud shading (see matplotlib docs).
    If contour=True
        fontsize
            The fontsize to apply to the contour labels. Default is 12.
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
    figsize = kwargs.pop("figsize", (9, 9))
    if "colors" in kwargs:
        if not contour:
            raise ValueError("'colors' is not a valid option if contour=False")
        if "cmap" in kwargs:
            raise ValueError("Cannot specify both 'colors' and 'cmap'")
    else:
        cmap = kwargs.pop("cmap", "rainbow")
        kwargs["cmap"] = (
            cmap  # This is getting a little uglier than I would like, but necessary for flexibility
        )
    alpha = kwargs.pop("alpha", 0.5)
    legend_label = kwargs.pop("legend_label", "Value")
    shading = kwargs.pop("shading", "nearest")
    colorbar = kwargs.pop("colorbar", True)

    fig = plt.figure(figsize=figsize)

    # Define the plot grid and create a new axes

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[10, 0.5],
        height_ratios=[10, 0.5],
        wspace=0.05,
        hspace=0.05,
    )
    ax = fig.add_subplot(gs[0])

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
        fontsize = kwargs.pop("fontsize", 12)
        fmt = kwargs.pop("fmt", "%2.2f")
        label_color = kwargs.pop("label_color", "k")
        cmesh = ax.contour(
            grid.grid_map[0], grid.grid_map[1], grid.data, alpha=alpha, **kwargs
        )
        ax.clabel(cmesh, colors=label_color, fmt=fmt, fontsize=fontsize)
    # Make the plot look nice and add a legend
    ax.xaxis.tick_top()

    if colorbar:
        ax2 = fig.add_subplot(gs[2])
        fig.colorbar(cmesh, cax=ax2, orientation="horizontal", label=legend_label)

    # Make the plot look nice
    ax.set_aspect("equal")
    if flip_x:
        ax.set_xlim(ax.get_xlim()[::-1])
    if flip_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    return fig, ax
