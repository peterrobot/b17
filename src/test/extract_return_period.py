from pathlib import Path

import numpy as np
import pandas as pd

from b17.wrap import load_src_df


def extract_return_period_basin(
    src_path: Path,
    sample_path: Path,
    trg_path: Path,
    basin_name: str,
    upper_return_year: float | None = None,
    lower_return_year: float | None = None,
    convert_cfs: bool = True,
):
    src_file = src_path / f"{basin_name}.txt"
    sample_file = sample_path / f"Sample_{basin_name}.csv"
    trg_file = trg_path / f"Selected_{basin_name}.csv"

    src_df = load_src_df(src_file=src_file, convert_cfs=convert_cfs)
    sample_df = pd.read_csv(sample_file, sep=",", header=0, index_col=0)
    if lower_return_year is not None:
        lower = np.float64(
            sample_df[sample_df["Return Period"] == lower_return_year][
                "Discharge"
            ].values[0]
        )
    else:
        lower = -1.0

    if upper_return_year is not None:
        upper = np.float64(
            sample_df[sample_df["Return Period"] == upper_return_year][
                "Discharge"
            ].values[0]
        )
    else:
        upper = 1e10

    sel_src_df = (
        src_df[(src_df["peak_va"] >= lower) & (src_df["peak_va"] <= upper)][
            ["peak_dt", "peak_va"]
        ]
        .rename(mapper={"peak_dt": "time", "peak_va": "discharge"}, axis=1)
        .set_index("time")
    )

    sel_src_df.to_csv(trg_file)


def extract_return_period_basins(
    src_path: Path,
    sample_path: Path,
    trg_path: Path,
    upper_return_year: float,
    lower_return_year: float,
    basin_names: list[str],
    convert_cfs: bool = True,
):

    for basin_name in basin_names:
        try:
            extract_return_period_basin(
                src_path=src_path,
                sample_path=sample_path,
                trg_path=trg_path,
                upper_return_year=upper_return_year,
                lower_return_year=lower_return_year,
                basin_name=basin_name,
                convert_cfs=convert_cfs,
            )
        except FileNotFoundError as e:
            print(f"Skip basin {basin_name}: {e}")


if __name__ == "__main__":

    years = [1.0001, 5, 25, 50, 100, None]

    basin_names = np.loadtxt(
        "/home/peter/Documents/prj/umlf/res/basin_list.txt",
        dtype=str,
        encoding="utf-8",
    ).tolist()

    for i in range(len(years) - 1):
        if years[i + 1] is not None:
            trg_path = Path(
                f"/home/peter/Documents/Output/UMLF/PeakFlow/PeakFlow{years[i]:.0f}to{years[i+1]:.0f}"
            )
        else:
            trg_path = Path(
                f"/home/peter/Documents/Output/UMLF/PeakFlow/PeakFlow{years[i]:.0f}to_inf"
            )
        trg_path.mkdir(parents=True, exist_ok=True)
        extract_return_period_basins(
            src_path=Path("/home/peter/Documents/Data/annual_peak_flow"),
            sample_path=Path("/home/peter/Documents/Output/MCRLSTM/PeakFlow"),
            trg_path=trg_path,
            upper_return_year=years[i + 1],
            lower_return_year=years[i],
            basin_names=basin_names,
            convert_cfs=True,
        )
