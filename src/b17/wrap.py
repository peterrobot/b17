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
    """Load and preprocess source peak flow data from USGS files.

    This function reads USGS peak flow data files, parses relevant columns,
    converts dates, and optionally converts discharge units from CFS to CMS.

    Args:
        src_file: Path to the USGS peak flow data file (.txt format).
        convert_cfs: Whether to convert discharge from cubic feet per second (CFS) to cubic meters per second (CMS). Defaults to True.

    Returns:
        pd.DataFrame: Processed DataFrame containing:
            - agency_cd: Agency code
            - site_no: Site number (basin ID)
            - peak_dt: Peak flow date as datetime
            - peak_va: Peak flow value (in CFS or CMS)

    Raises:
        FileNotFoundError: If src_file does not exist.
        pd.errors.EmptyDataError: If the file is empty or malformed.

    Example:
        >>> df = load_src_df(
        ...     src_file=Path("data/peaks/01013500.txt"),
        ...     convert_cfs=True
        ... )
        >>> print(df.head())
    """
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
    """Calculate B17 flood frequency analysis for a single basin.

    This function performs B17 flood frequency analysis on peak flow data
    for a hydrological basin, extracts discharge values for a target return
    period, and saves the results to CSV files.

    Args:
        basin_id: Unique identifier for the basin to analyze.
        src_path: Path to source directory containing basin data files.
        trg_path: Path to target directory where results will be saved.
        target_return_year: Return period in years for which to extract discharge.
        convert_cfs: Whether to convert discharge from CFS to CMS. Defaults to True.
        is_override: Whether to override existing results. Defaults to False.

    Returns:
        tuple[Path, Path]: A tuple containing paths to:
            - Return period analysis results CSV file
            - Selected peak flow events CSV file

    Raises:
        FileNotFoundError: If source data file for the basin is not found.
        ValueError: If target_return_year is not found in return period analysis.

    Example:
        >>> ret_path, sel_path = single_basin_b17(
        ...     basin_id="01013500",
        ...     src_path=Path("data/peaks"),
        ...     trg_path=Path("results/b17"),
        ...     target_return_year=100.0,
        ...     convert_cfs=True,
        ...     is_override=False
        ... )
    """
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
) -> tuple[Path, Path] | tuple[str, str]:
    """Wrapper function for B17 flood frequency analysis with error handling.

    This function wraps the single_basin_b17 function and provides comprehensive
    error handling to ensure individual basin failures don't affect batch processing.

    Args:
        basin_id: Unique identifier for the basin to analyze.
        src_path: Path to source directory containing basin data files.
        trg_path: Path to target directory where results will be saved.
        target_return_year: Return period in years for which to extract discharge.
        convert_cfs: Whether to convert discharge from CFS to CMS. Defaults to True.
        is_override: Whether to override existing results. Defaults to False.

    Returns:
        tuple[Path, Path] | tuple[str, str]:
            - If successful: tuple of paths to return period results and selected events CSV files
            - If failed: tuple containing basin_id and error traceback string

    Example:
        >>> result = wrap_b17(
        ...     basin_id="01013500",
        ...     src_path=Path("data/peaks"),
        ...     trg_path=Path("results/b17"),
        ...     target_return_year=100.0
        ... )
        >>> if isinstance(result[0], Path):
        ...     print(f"Success: {result[0]}")
        ... else:
        ...     print(f"Failed: {result[0]} - {result[1]}")
    """
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
    """Process multiple basins in parallel using B17 flood frequency analysis.

    This function performs batch processing of multiple basins using multiprocessing
    to calculate B17 flood frequency analysis and save results to CSV files.

    Args:
        basin_list: List of basin IDs to process.
        src_path: Path to source directory containing basin data files.
        trg_path: Path to target directory where results will be saved.
        target_return_year: Return period in years for which to extract discharge.
        convert_cfs: Whether to convert discharge from CFS to CMS. Defaults to True.
        is_override: Whether to override existing results. Defaults to False.
        n_processes: Number of parallel processes to use. If None, uses all available CPUs.

    Raises:
        AssertionError: If src_path does not exist or is not a directory.
        FileNotFoundError: If source data files are not found for any basin.

    Example:
        >>> batch_save_df(
        ...     basin_list=["01013500", "01022500", "01031500"],
        ...     src_path=Path("data/peaks"),
        ...     trg_path=Path("results/b17"),
        ...     target_return_year=100.0,
        ...     n_processes=4
        ... )
    """
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
