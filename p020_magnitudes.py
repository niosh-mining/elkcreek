import matplotlib.pyplot as plt
import pandas as pd

from elkcreek.plot import LABEL_MAPPING, configure_font_sizes
import local


def magnitude_plot(eves, ax):

    eves = eves.copy()

    c = local.cp_hex[1]

    # Next plot a histogram of local mag
    col = "local_mag"
    eves[col].hist(ax=ax, bins=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], grid=False, edgecolor="w", facecolor=c)
    ax.set_yscale("log")
    ax.set_xlabel(LABEL_MAPPING[col])
    ax.set_ylabel("Number of Events")


def main():
    # Load the data
    unfiltered = pd.read_parquet(local.combined_cat_path)
    df = pd.read_parquet(local.cleaned_cat_path)

    # Set the font sizes
    configure_font_sizes(local.font_sizes)

    fig, (raw_ax, filt_ax) = plt.subplots(2, 1, figsize=(3.5, 5.8))

    magnitude_plot(unfiltered, ax=raw_ax)
    magnitude_plot(df, ax=filt_ax)

    # Add subplot labels...
    label_x, label_y = -0.3, 1.05
    raw_ax.set_title("All Detected Events")
    raw_ax.text(label_x, label_y, "(a)", transform=raw_ax.transAxes)
    filt_ax.set_title("After Quality Filtering")
    filt_ax.text(label_x, label_y, "(b)", transform=filt_ax.transAxes)

    fig.tight_layout()
    fig.savefig(local.mag_hist_path, **local.savefig_params)


if __name__ == "__main__":
    main()
