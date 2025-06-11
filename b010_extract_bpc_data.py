"""
Extract the borehole pressure data.
"""

import pandas as pd
from elkcreek.excel import read_bpc_excel_file

import local


def main():
    """Extract the borehole pressure data from excel files."""
    out = []
    for path in local.bpc_path.glob("*xlsx"):
        out.append(read_bpc_excel_file(path))
    df = pd.concat(out, axis=0, ignore_index=True)
    df.to_parquet(local.extracted_bpc_data_path)


if __name__ == "__main__":
    main()
