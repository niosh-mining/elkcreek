"""
Extract the loads on the support cans.
"""

import pandas as pd
from elkcreek.excel import read_support_can_displacement, read_support_can_excel_file

import local


def main():
    """Extract the support can load data from excel file."""
    can_load = []
    can_displacement = []
    for path in local.support_can_path.glob("*xlsx"):
        if path.name.startswith("Manual Roof"):
            continue
        df = read_support_can_excel_file(path)
        can_load.append(df)

        df2 = read_support_can_displacement(path)
        can_displacement.append(df2)
    # Save load
    df = pd.concat(can_load, ignore_index=True, axis=0)
    df.to_parquet(local.extracted_support_can_path)
    # Save displacement
    df = pd.concat(can_displacement, ignore_index=True, axis=0)
    df.to_parquet(local.extracted_can_disp_path)


if __name__ == "__main__":
    # Since the spreadsheets have different data, we can just concat
    # to make one spreadsheet with all can data.
    main()
