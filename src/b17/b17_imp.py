import importlib.resources

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import PchipInterpolator, make_interp_spline


def station_stats(data_in: np.ndarray) -> tuple[float, int, float, float]:
    """
    Calculates Station Statistics: Mean, Standard Deviation, and Skew of log10(data).
    """
    data = data_in
    n = len(data)

    x = np.log10(data)
    x_mean = float(np.mean(x))

    # Handle cases with no variance
    if n < 3:
        return 0.0, n, 0.0, x_mean

    s = np.sqrt((np.sum(x**2) - np.sum(x) ** 2 / n) / (n - 1))

    if s == 0.0 or n < 3:
        return 0.0, n, s, x_mean

    g = (
        n**2 * np.sum(x**3)
        - 3 * n * np.sum(x) * np.sum(x**2)
        - 2 * np.sum(x) ** 3
    ) / ((n * (n - 1) * (n - 2)) * s**3)

    return g, n, s, x_mean


def freq_curve(x_mean, s, g, ktable):
    """
    Computes frequency curve coordinates by interpolating K values.
    """
    p_unique = np.unique(ktable[:, 1])
    k_vals = np.zeros(len(p_unique))

    for i, p_val in enumerate(p_unique):
        mask = ktable[:, 1] == p_val
        g_table = ktable[mask, 0]
        k_table = ktable[mask, 2]

        interp_func = make_interp_spline(g_table, k_table, k=3)
        k_vals[i] = interp_func(g)

    k_with_p = np.vstack((k_vals, p_unique)).T
    flood_freq = np.vstack((p_unique, x_mean + k_vals * s)).T

    g_zero = ktable[ktable[:, 0] == 0.0]
    g_zero = g_zero[g_zero[:, 1].argsort()]

    return k_with_p, flood_freq, g_zero


def plot_pos(data_in, n):
    """
    Calculates Gringorten and Weibull plotting positions.
    """
    dsort = data_in[data_in.argsort()[::-1]]
    r = np.arange(1, len(dsort) + 1)

    gp = (r - 0.44) / (n + 0.12)
    gw = r / (n + 1)

    return np.vstack((gp, gw, dsort[:], dsort[:])).T


def historical_adjust(H, L, Z, X, Xz):
    """
    Adjusts statistics for historical data/outliers per Appendix 6 of B17B.
    """
    N = len(X)
    # Handle division by zero if N + L is zero
    if N + L == 0:
        W = 0
    else:
        W = (H - Z) / (N + L)

    # Handle division by zero
    denominator_m = H - W * L
    if denominator_m == 0:
        Mbar = 0
    else:
        Mbar = (W * np.sum(X) + np.sum(Xz)) / denominator_m

    denominator_s = H - W * L - 1
    if denominator_s <= 0:
        Sbar = 0
    else:
        Sbar = np.sqrt(
            (W * np.sum((X - Mbar) ** 2) + np.sum((Xz - Mbar) ** 2))
            / denominator_s
        )

    denominator_g = (H - W * L - 1) * (H - W * L - 2)
    if denominator_g <= 0 or Sbar == 0:
        Gbar = 0
    else:
        Gbar = ((H - W * L) / denominator_g) * (
            (W * np.sum((X - Mbar) ** 3) + np.sum((Xz - Mbar) ** 3)) / Sbar**3
        )

    Xh = np.sort(np.concatenate([X, Xz]))[::-1]
    E = np.arange(1, Z + N + 1)
    m = E.astype(float)

    mh = W * E - ((W - 1) * (Z + 0.50))
    m[Z:] = mh[Z:]

    wp = m / (H + 1)

    hp = np.vstack((wp, 10**Xh)).T
    return Gbar, Mbar, Sbar, hp


def b17(
    data_in: np.ndarray,
    gg: float | None = None,
) -> tuple[np.ndarray, list[float], np.ndarray, float, float, np.ndarray]:
    """
    Performs a Flood Flow Frequency Analysis based on the USGS Bulletin 17B guidelines.

    This function fits a Log-Pearson Type III (LP3) distribution to a series of
    annual peak streamflow data to estimate flood quantiles at various exceedance
    probabilities (or return periods).

    The core of the analysis is based on the method of moments for the log-transformed
    data and follows these key mathematical and procedural steps:

    1.  **Log Transform**: The analysis is performed on the base-10 logarithms of the
        annual peak flow data `Q`. Let `X = log10(Q)`.

    2.  **Station Statistics**: The initial sample mean (`Xmean`), standard deviation (`S`),
        and skew coefficient (`G`) of the `X` series are calculated.

    3.  **Outlier Detection**: High and low outliers are identified using a 10%
        significance test. The test statistic is `K = (X_outlier - Xmean) / S`, which
        is compared against critical `KN` values from Bulletin 17B, where `N` is the
        record length. If outliers are found, they are removed, and statistics are
        conditionally recomputed.

    4.  **Historical Data Adjustment**: If historical peaks (or high outliers) exist,
        the statistics are adjusted to give more weight to these significant events,
        resulting in a historically-adjusted skew (`G_hist`), mean (`Mbar`), and
        standard deviation (`Sbar`).

    5.  **Conditional Probability**: If low outliers or zero-flow years were removed,
        the sample is considered truncated. A conditional probability adjustment is
        applied, and a synthetic skew (`GS`), mean (`XS`), and standard deviation (`SS`)
        are computed to better represent the distribution.

    6.  **Weighted Skew**: A final, weighted skew (`GD`) is calculated by combining the
        station skew (`GS`) with a user-provided generalized regional skew (`gg`).
        The weighting is based on the respective mean square errors (MSE) of the
        station and regional skews:
        `GD = (MSE_regional * GS + MSE_station * gg) / (MSE_regional + MSE_station)`

    7.  **Frequency Curve**: Flood quantiles in log space are calculated using the
        fundamental LP3 equation: `X_T = XS + K * SS`, where `X_T` is the log of the
        T-year flood, `XS` and `SS` are the final mean and standard deviation, and `K` is
        the LP3 frequency factor. `K` is a function of the weighted skew `GD` and the
        exceedance probability, found by interpolating standard tables. The final
        discharge is `Q_T = 10**X_T`.

    8.  **Confidence Intervals**: 95% confidence limits are computed to quantify the
        uncertainty in the estimated frequency curve.

    Args:
        peak_flows (np.ndarray): A 1-dimensional list, Pandas Series, or NumPy array
                                 containing the annual peak streamflow data.
        gg (float, optional): The generalized regional skew. Defaults to 0.0.

    Returns:
        tuple: A tuple containing the following six elements:

        - **dataout** (np.ndarray): The final computed frequency curve. An Nx6 array with columns:
            1. Return Period (years)
            2. Exceedance Probability (e.g., 0.01 for 100-year)
            3. Final Frequency Flow (discharge)
            4. Upper 95% Confidence Interval Flow
            5. Lower 95% Confidence Interval Flow
            6. Expected Probability Flow (adjusted for sample size)

        - **skews** (list[float]): A log of skews calculated at various stages
          of the analysis, including initial, post-outlier, historically
          adjusted, synthetic (if applicable), and the final weighted skew.

        - **pp** (np.ndarray): Plotting positions for the original non-zero input
          data. An Mx4 array with columns:
            1. Gringorten plotting position (probability)
            2. Weibull plotting position (probability)
            3. Original Peak Flow value (sorted descending)
            4. Dummy Year/Index

        - **XS** (float): The final mean of the log10-transformed flows used to
          compute the frequency curve.

        - **SS** (float): The final standard deviation of the log10-transformed
          flows used to compute the frequency curve.

        - **hp** (np.ndarray): Historically adjusted plotting positions, which
          are calculated if high outliers are detected. A Kx2 array with columns:
            1. Adjusted Weibull plotting position (probability)
            2. Flow value (includes historical/high outliers)
    """
    if gg is None:
        gg = 0.0

    kn_file = importlib.resources.files("b17.data").joinpath("KNtable.nc")
    k_file = importlib.resources.files("b17.data").joinpath("ktable.nc")
    pn_file = importlib.resources.files("b17.data").joinpath("PNtable.nc")
    with importlib.resources.as_file(kn_file) as nc_path:
        with xr.open_dataset(nc_path) as ds:
            kn_table = ds["KNtable"].values

    with importlib.resources.as_file(k_file) as nc_path:
        with xr.open_dataset(nc_path) as ds:
            ktable = ds["ktable"].values

    with importlib.resources.as_file(pn_file) as nc_path:
        with xr.open_dataset(nc_path) as ds:
            pn_table = ds["Pntable"].values

    non_zero_flood = data_in[data_in > 0]
    n_total = len(data_in)
    G, N, S, X_mean = station_stats(non_zero_flood)
    skews = [G]

    # Use original stats for all outlier threshold calculations
    G_orig, N_orig, S_orig, X_mean_orig = G, N, S, X_mean

    kn_row_idx = np.argmin(np.abs(kn_table[:, 0] - N_orig))
    kn_val = kn_table[kn_row_idx, 1]

    # Calculate BOTH thresholds from ORIGINAL stats first.
    qh = 10 ** (X_mean_orig + kn_val * S_orig)
    ql = 10 ** (X_mean_orig - kn_val * S_orig)

    if -0.4 <= G_orig <= 0.4:
        # Filter based on both thresholds
        datafilter = non_zero_flood[
            (non_zero_flood > ql) & (non_zero_flood < qh)
        ]
    elif G_orig > 0.4:
        # Filter for high first, then low, using the original thresholds
        datafilter_high = non_zero_flood[non_zero_flood < qh]
        datafilter = datafilter_high[datafilter_high > ql]
    else:  # G < -0.4
        # Filter for low first, then high, using the original thresholds
        datafilter_low = non_zero_flood[non_zero_flood > ql]
        datafilter = datafilter_low[datafilter_low < qh]

    # Calculate outlier counts using the original thresholds
    QL_cnt = np.sum(data_in <= ql)
    QH_cnt = np.sum(data_in > qh)

    # ### MODIFICATION 1 ###
    # Replicate MATLAB's logic: re-compute stats only if low outliers are detected.
    if QL_cnt > 0:
        G, N, S, X_mean = station_stats(datafilter)

    skews.append(G)
    Xz = np.log10(data_in[data_in > qh])
    X = np.log10(datafilter)

    # Historical adjustment uses the filtered data
    G_hist, Mbar, Sbar, hp = historical_adjust(n_total, QL_cnt, QH_cnt, X, Xz)
    skews.append(G_hist)
    G = G_hist  # Adopt the historically adjusted skew

    if QL_cnt > 0:
        Pest = N / n_total
        if Pest < 0.75:
            raise ValueError(
                "Warning: Too many outliers or zero flow years. Results may be unreliable."
            )

        _, flood_freq_temp, _ = freq_curve(X_mean, S, G, ktable)
        adj_p = flood_freq_temp[:, 0] * Pest
        interp_func = PchipInterpolator(
            adj_p, flood_freq_temp[:, 1], extrapolate=True
        )

        Q01 = 10 ** interp_func(0.01)
        Q10 = 10 ** interp_func(0.10)
        Q50 = 10 ** interp_func(0.50)

        GS = -2.5 + 3.12 * (np.log10(Q01 / Q10) / np.log10(Q10 / Q50))
        if GS < -2.0 or GS > 2.5:
            print(
                "Warning: Synthetic skew exceeds acceptable limits. User should plot data for further evaluation."
            )
        skews.append(float(GS))

        KS_vals, _, _ = freq_curve(0, 1, GS, ktable)
        K01 = KS_vals[np.isclose(KS_vals[:, 1], 0.01), 0][0]
        K50 = KS_vals[np.isclose(KS_vals[:, 1], 0.50), 0][0]

        SS = np.log10(Q01 / Q50) / (K01 - K50)
        XS = np.log10(Q50) - K50 * SS
    else:
        GS = G_hist
        # ### MODIFICATION 2 ###
        # Use station stats (post-outlier removal), not historically adjusted stats, to match MATLAB.
        XS = X_mean
        SS = S

    if abs(G - gg) > 0.5:
        print(
            f"Warning: Large discrepancy (> 0.5) between calculated station skew (G = {G:.3f}) and generalized skew (G = {gg:.3f}). More weight should be given to the Station skew."
        )

    MSEGbar = 0.302
    A = -0.33 + 0.08 * abs(G) if abs(G) <= 0.90 else -0.52 + 0.30 * abs(G)
    B = 0.94 - 0.26 * abs(G) if abs(G) <= 1.50 else 0.55
    MSEG = 10 ** (A - B * np.log10(n_total / 10))

    GD = (MSEGbar * GS + MSEG * gg) / (MSEGbar + MSEG)
    skews.append(GD)

    K_final, finalfreq_log, Gzero = freq_curve(XS, SS, GD, ktable)

    Galpha_k_row = Gzero[np.isclose(Gzero[:, 1], 0.05)]
    Galpha_k = Galpha_k_row[0, 2] if Galpha_k_row.size > 0 else 0

    a = 1 - (Galpha_k**2) / (2 * (n_total - 1))
    b = K_final[:, 0] ** 2 - (Galpha_k**2) / n_total
    sqrt_term = np.maximum(0, K_final[:, 0] ** 2 - a * b)

    Ku = (K_final[:, 0] + np.sqrt(sqrt_term)) / a
    Kl = (K_final[:, 0] - np.sqrt(sqrt_term)) / a

    LQu_log = XS + Ku * SS
    LQl_log = XS + Kl * SS

    p_unique = np.unique(pn_table[:, 1])
    pexp_vals = np.zeros(len(p_unique))

    for i, p_val in enumerate(p_unique):
        mask = pn_table[:, 1] == p_val
        n_vals = pn_table[mask, 0]
        pn_vals = pn_table[mask, 2]
        interp_func = PchipInterpolator(n_vals, pn_vals)
        pexp_vals[i] = interp_func(N - 1)

    pexp_table = np.vstack((p_unique, pexp_vals)).T

    common_probs, ia, ib = np.intersect1d(
        pexp_table[:, 0], finalfreq_log[:, 0], return_indices=True
    )
    if len(common_probs) > 1:
        interp_func = PchipInterpolator(
            pexp_table[ia, 1], finalfreq_log[ib, 1], extrapolate=True
        )
        expectedP_log = interp_func(finalfreq_log[:, 0])
        expectedP_log[finalfreq_log[:, 0] < 0.002] = np.nan
    else:
        expectedP_log = np.full_like(finalfreq_log[:, 1], np.nan)

    data_out = np.vstack(
        [
            1.0 / finalfreq_log[:, 0],
            finalfreq_log[:, 0],
            10 ** finalfreq_log[:, 1],
            10**LQu_log,
            10**LQl_log,
            10**expectedP_log,
        ]
    ).T
    data_out = data_out[data_out[:, 0].argsort()]

    pp = plot_pos(non_zero_flood, n_total)

    return data_out, skews, pp, XS, SS, hp


def extract_ret_df(ret, is_format: bool = True) -> pd.DataFrame:

    p_df = pd.DataFrame(
        data=ret[0],
        columns=[
            "Return Period",
            "Probability",
            "Discharge",
            "Upper 95%",
            "Lower 95%",
            "Expect Probability Flow",
        ],
    )

    data_name = [
        "Discharge",
        "Upper 95%",
        "Lower 95%",
        "Expect Probability Flow",
    ]

    if is_format:
        p_df["Return Period"] = p_df["Return Period"].round(5)
        p_df[data_name] = p_df[data_name].round(2)

    return p_df
