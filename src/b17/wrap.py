import functools
import multiprocessing
import traceback
from pathlib import Path

import loguru
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from b17.b17_imp import b17, extract_ret_df


def load_src_df(src_file: Path, convert_cfs: bool = True) -> pd.DataFrame:
    df = pd.read_csv(src_file, header=72, sep="\t").iloc[1:][
        ["agency_cd", "site_no", "peak_dt", "peak_va"]
    ]
    df["peak_dt"] = pd.to_datetime(
        df["peak_dt"], format="%Y-%m-%d", errors="coerce"
    )
    df = df.dropna(subset=["peak_dt"])
    if convert_cfs:
        df["peak_va"] = df["peak_va"].astype(np.float32) * 0.02832
    return df


def single_basin_b17(
    basin_id: str,
    src_path: Path,
    trg_path: Path,
    target_return_year: float,
    convert_cfs: bool = True,
    is_override: bool = False,
) -> tuple[Path, Path]:
    ret_path = trg_path / f"Sample_{basin_id}.csv"
    sel_src_path = trg_path / f"Selected_{basin_id}.csv"

    if not is_override and ret_path.exists() and sel_src_path.exists():
        return ret_path, sel_src_path

    src_df = load_src_df(
        src_file=src_path / f"{basin_id}.txt", convert_cfs=convert_cfs
    )
    ret_df = extract_ret_df(b17(data_in=src_df["peak_va"].to_numpy()))
    flow = np.float64(
        ret_df[ret_df["Return Period"] == target_return_year][
            "Discharge"
        ].values[0]
    )

    sel_src_df = (
        src_df[src_df["peak_va"] > flow][["peak_dt", "peak_va"]]
        .rename(mapper={"peak_dt": "time", "peak_va": "discharge"}, axis=1)
        .set_index("time")
    )

    ret_df.to_csv(ret_path)
    sel_src_df.to_csv(sel_src_path)

    return ret_path, sel_src_path


def wrap_b17(
    basin_id: str,
    src_path: Path,
    trg_path: Path,
    target_return_year: float,
    convert_cfs: bool = True,
    is_override: bool = False,
):
    try:
        return single_basin_b17(
            basin_id=basin_id,
            src_path=src_path,
            trg_path=trg_path,
            target_return_year=target_return_year,
            convert_cfs=convert_cfs,
            is_override=is_override,
        )
    except Exception as e:
        error_message = f"Critical error for basin {basin_id}: {e!s}"
        error_traceback = traceback.format_exc()
        loguru.logger.error(error_message)
        loguru.logger.error(error_traceback)
        return basin_id, error_traceback


def batch_save_df(
    basin_list: list[str],
    src_path: Path,
    trg_path: Path,
    target_return_year: float,
    convert_cfs: bool = True,
    is_override: bool = False,
    n_processes: int | None = None,
):
    assert src_path.exists() and src_path.is_dir()

    trg_path.mkdir(parents=True, exist_ok=True)

    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    func = functools.partial(
        wrap_b17,
        src_path=src_path,
        trg_path=trg_path,
        target_return_year=target_return_year,
        convert_cfs=convert_cfs,
        is_override=is_override,
    )
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=len(basin_list), desc="Sampling basins") as pbar:
            for _ in pool.imap_unordered(
                func=func,
                iterable=basin_list,
                chunksize=1,
            ):
                pbar.update(1)
