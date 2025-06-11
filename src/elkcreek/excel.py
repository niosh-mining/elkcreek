"""Module for reading excel files associated with the project."""

import numpy as np
import pandas as pd


def add_local_time_from_serial_date(df):
    """Convert the serial time (number of days since 1900 to datetime)"""
    base = np.datetime64("1899-12-30")  # historical excel bug!
    possible_cols = [x for x in df.columns if x.startswith("Serial")]
    assert len(possible_cols) == 1
    col = possible_cols[0]
    serial_date = df[col].astype(np.float64)
    td = pd.to_timedelta(serial_date.values, unit="D")
    return df.assign(**{col: base + td}).rename(columns={col: "local_time"})


def read_bpc_excel_file(path):
    """Parse out the BPC data."""

    def rename_and_convert_pressures(df, conversion=6894.76):
        """Convert pressures to Pa and rename columns."""
        cols = [x for x in df.columns if x.endswith("cell pressure (psi)")]
        new_name = {x: f"{x.split(' ')[0]}_pa" for x in cols}
        df_metric_pressures = (df[cols] * conversion).rename(columns=new_name)
        return df.drop(columns=cols).join(df_metric_pressures)

    def convert_face_distance(df, conversion=0.3048):
        out = df.assign(face_distance_m=df["face_distance_ft"] * conversion).drop(
            columns=["face_distance_ft"]
        )
        return out

    def remove_string_pods(df):
        """Remove any string pod columns."""
        cols = [x for x in df.columns if "Displ." in x]
        return df.drop(columns=cols)

    def pivot_bpcs(df):
        """Pivot each BPC into its own row."""
        out = []
        bpc_cols = [x for x in df.columns if x.endswith("_pa")]
        cols_no_pcs = sorted(set(df.columns) - set(bpc_cols))
        for col in bpc_cols:
            pressure = df[col]
            sub = df[cols_no_pcs].assign(
                pressure_pa=pressure.values, name=col.split("_")[0]
            )
            out.append(sub.dropna(subset=["pressure_pa"]))
        return pd.concat(out, ignore_index=True, axis=0)

    split_name = path.name.split(" ")
    site = "_".join(split_name[:3]).lower()
    panel = int(split_name[4])

    column_name_map = {
        "Distance from instrument site to face (ft)": "face_distance_ft",
        "Data-logger number": "data_logger",
    }
    drop_cols = ["Date", "Time from zero time (hr)", "Time of day"]

    df = (
        pd.read_excel(path, skiprows=9)
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .pipe(add_local_time_from_serial_date)
        .rename(columns=column_name_map)
        .drop(columns=drop_cols)
        .pipe(rename_and_convert_pressures)
        .pipe(convert_face_distance)
        .pipe(remove_string_pods)
        .assign(site=site, panel=panel)
    )
    out = pivot_bpcs(df)
    return out


def read_support_can_excel_file(path, group=""):
    """Read the data from the support can files."""

    def _split_at_unnamed_column(df):
        """Split the columns at the first un-named column."""
        for num, col in enumerate(df.columns):
            if col.lower().startswith("unnamed"):
                break
        else:
            raise ValueError("Could not find unnamed column")
        cols = df.columns[:num]
        return df[cols]

    def _get_cans_from_columns(df):
        """Get the can names from the columns."""
        ser = pd.Series(df.columns)
        cans = ser.str.extract(r"\bCan (\w)\b").dropna()[0]
        return cans.unique()

    def _extract_can_info(long_df, cans, ft_to_m=0.3048):
        """Extract information from the cans."""
        out = []
        col_set = set(long_df.columns)
        time = long_df["local_time"]
        for can in cans:
            panel_1_dist_col = f"Distance from support Can {can}  to Panel 1 face (ft)"
            panel_2_dist_col = f"Distance from support Can {can}  to Panel 2 face (ft)"
            load_col = f"Load on support Can {can} (ton)"
            assert {panel_2_dist_col, panel_1_dist_col, load_col}.issubset(col_set)
            panel_2_dist = long_df[panel_2_dist_col].values * ft_to_m
            panel_1_dist = long_df[panel_1_dist_col].values * ft_to_m

            has_panel_1_dist = ~pd.isnull(panel_1_dist)
            has_panel_2_dist = ~pd.isnull(panel_2_dist)

            face_dist = np.empty_like(long_df[panel_1_dist_col])
            panel = np.array([""] * len(panel_1_dist))
            face_dist[has_panel_1_dist] = panel_1_dist[has_panel_1_dist]
            face_dist[has_panel_2_dist] = panel_2_dist[has_panel_2_dist]
            panel[has_panel_2_dist] = "2"
            panel[has_panel_1_dist] = "1"

            df_data = dict(
                load_tons=long_df[load_col].values,
                local_time=time.values,
                panel=panel,
                face_distance_m=face_dist,
            )
            sub = pd.DataFrame(df_data).assign(can=can).dropna(subset=["load_tons"])
            out.append(sub)

        out_df = pd.concat(out, axis=0)
        return out_df

    drop_cols = ["Date", "Time since zero time (hr)"]

    # First get df in long form
    long_df = (
        pd.read_excel(path, skiprows=8)
        .pipe(add_local_time_from_serial_date)
        .pipe(_split_at_unnamed_column)
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .drop(columns=drop_cols)
    )

    # Then find cans
    cans = _get_cans_from_columns(long_df)

    df = _extract_can_info(long_df, cans)
    df["group"] = group  # add site
    return df


def read_support_can_displacement(path, group=""):
    """Read the displacement data from a can file."""

    def _get_sites_dict(df):
        """Get {site_name: series}"""
        ser = pd.Series(df.columns)
        sites_raw = ser.str.extract(r"Displacement at (.+?) \((?:in|mm)\)")[0]
        cols = pd.Series(df.columns)[~pd.isnull(sites_raw)]
        sites = sites_raw.dropna().str.replace("support Can ", "")
        out = {x: df[y] for x, y in zip(sites, cols)}
        return out

    def _extract_displacement(long_df, site_dict, inch_to_m=0.0254):
        """Extract information from the cans."""
        out = []
        col_set = set(long_df.columns)
        time = long_df["local_time"]
        for site, ser in site_dict.items():
            ser_m = ser * inch_to_m
            df_data = dict(
                local_time=time.values,
                displacement_m=ser_m.values,
            )
            sub = pd.DataFrame(df_data).assign(site=site)
            out.append(sub)

        out_df = pd.concat(out, axis=0)
        return out_df

    drop_cols = ["Date", "Time since zero time (hr)"]

    # First get df in long form
    long_df = (
        pd.read_excel(path, skiprows=8)
        .pipe(add_local_time_from_serial_date)
        # .pipe(_split_at_unnamed_column)
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .drop(columns=drop_cols)
    )

    # Then find cans
    disp_dic = _get_sites_dict(long_df)
    df = _extract_displacement(long_df, disp_dic)
    df["group"] = group
    return df
