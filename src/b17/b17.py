import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator, make_interp_spline


def station_stats(datain: np.ndarray) -> tuple[float, int, float, float]:
    """
    Calculates Station Statistics: Mean, Standard Deviation, and Skew of log10(data).
    """
    data = datain[:, 1]
    n = len(data)

    x = np.log10(data)
    x_mean = float(np.mean(x))

    s = np.sqrt((np.sum(x**2) - np.sum(x) ** 2 / n) / (n - 1))

    if s == 0.0:
        return 0.0, n, s, x_mean

    g = (
        n**2 * np.sum(x**3)
        - 3 * n * np.sum(x) * np.sum(x**2)
        + 2 * np.sum(x) ** 3
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


def plot_pos(datain, n):
    """
    Calculates Gringorten and Weibull plotting positions.
    """
    dsort = datain[datain[:, 1].argsort()[::-1]]
    r = np.arange(1, len(dsort) + 1)

    gp = (r - 0.44) / (n + 0.12)
    gw = r / (n + 1)

    return np.vstack((gp, gw, dsort[:, 1], dsort[:, 0])).T


def historical_adjust(H, L, Z, X, Xz):
    """
    Adjusts statistics for historical data/outliers per Appendix 6 of B17B.
    """
    N = len(X)
    W = (H - Z) / (N + L)

    Mbar = (W * np.sum(X) + np.sum(Xz)) / (H - W * L)
    Sbar = np.sqrt(
        (W * np.sum((X - Mbar) ** 2) + np.sum((Xz - Mbar) ** 2))
        / (H - W * L - 1)
    )
    Gbar = ((H - W * L) / ((H - W * L - 1) * (H - W * L - 2))) * (
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
    datain,
    gg=None,
    imgfile=None,
):
    """
    Python script for Flood Flow Frequency Analysis using USGS Bulletin 17B,
    modified to use xarray for interacting with NetCDF input files.
    """
    if gg is None:
        gg = 0.0

    # MODIFIED: Load required tables from NetCDF files using xarray
    # Using a with statement ensures the files are properly closed.
    with xr.open_dataset("KNtable.nc") as ds:
        kn_table = ds["KNtable"].values
    with xr.open_dataset("ktable.nc") as ds:
        ktable = ds["ktable"].values
    with xr.open_dataset("PNtable.nc") as ds:
        pn_table = ds["Pntable"].values

    non_zero_flood = datain[datain[:, 1] > 0]
    n_total = len(datain)
    G, N, S, Xmean = station_stats(non_zero_flood)
    skews = [G]

    kn_row_idx = np.argmin(np.abs(kn_table[:, 0] - N))
    kn_val = kn_table[kn_row_idx, 1]

    datafilter = non_zero_flood.copy()

    if -0.4 <= G <= 0.4:

        qh = 10 ** (Xmean + kn_val * S)
        ql = 10 ** (Xmean - kn_val * S)

        datafilter = non_zero_flood[
            (non_zero_flood[:, 1] > ql) & (non_zero_flood[:, 1] < qh)
        ]

        QLcnt = np.sum(datain[:, 1] <= ql)
        QHcnt = np.sum(datain[:, 1] > qh)
        if QLcnt > 0:
            G, N, S, Xmean = station_stats(datafilter)
    elif G > 0.4:
        qh = 10 ** (Xmean + kn_val * S)
        datafilter = non_zero_flood[non_zero_flood[:, 1] < qh]
        QHcnt = np.sum(datain[:, 1] > qh)

        G_temp, N_temp, S_temp, Xmean_temp = station_stats(datafilter)
        kn_row_idx = np.argmin(np.abs(kn_table[:, 0] - N_temp))
        kn_val = kn_table[kn_row_idx, 1]
        ql = 10 ** (Xmean_temp - kn_val * S_temp)

        datafilter = datafilter[datafilter[:, 1] > ql]
        QLcnt = np.sum(datain[:, 1] <= ql)
        if QLcnt > 0:
            G, N, S, Xmean = station_stats(datafilter)
    else:  # G < -0.4
        # Test for low outliers, recompute, then test for high
        ql = 10 ** (Xmean - kn_val * S)
        datafilter = non_zero_flood[non_zero_flood[:, 1] > ql]
        QLcnt = np.sum(datain[:, 1] <= ql)

        G_temp, N_temp, S_temp, Xmean_temp = station_stats(datafilter)
        kn_row_idx = np.argmin(np.abs(kn_table[:, 0] - N_temp))
        kn_val = kn_table[kn_row_idx, 1]
        qh = 10 ** (Xmean_temp + kn_val * S_temp)

        datafilter = datafilter[datafilter[:, 1] < qh]
        QHcnt = np.sum(datain[:, 1] > qh)
        if QLcnt > 0:
            G, N, S, Xmean = station_stats(datafilter)

    skews.append(G)
    Xz = np.log10(datain[datain[:, 1] > qh, 1])
    X = np.log10(datafilter[:, 1])
    skews.append(G)

    G_hist, Mbar, Sbar, hp = historical_adjust(n_total, QLcnt, QHcnt, X, Xz)
    skews.append(G_hist)
    G = G_hist

    if QLcnt > 0:
        Pest = N / n_total
        if Pest < 0.75:
            raise ValueError(
                "Warning: Too many outliers or zero flow years. Results may be unreliable."
            )

        _, flood_freq_temp, _ = freq_curve(Xmean, S, G, ktable)

        # Adjust probabilities
        adj_p = flood_freq_temp[:, 0] * Pest

        # Interpolate to find Q values at standard probabilities
        interp_func = PchipInterpolator(adj_p, flood_freq_temp[:, 1])
        adj_freq_log = interp_func(flood_freq_temp[:, 0])

        Q01 = 10 ** interp_func(0.01)
        Q10 = 10 ** interp_func(0.10)
        Q50 = 10 ** interp_func(0.50)

        # Generate synthetic Log-Pearson statistics
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
        # If no low outliers, use historical stats
        GS = G_hist
        XS = Mbar
        SS = Sbar
    if abs(G - gg) > 0.5:
        print(
            f"Warning: Large discrepancy (> 0.5) between calculated station skew (G = {G:.3f}) and generalized skew (G = {gg:.3f}). More weight should be given to the Station skew."
        )

    MSEGbar = 0.302
    A = -0.33 + 0.08 * abs(G) if abs(G) <= 0.90 else -0.52 + 0.30 * abs(G)
    B = 0.94 - 0.26 * abs(G) if abs(G) <= 1.50 else 0.55
    MSEG = 10 ** (A - B * np.log10(n_total / 10))
    GD = (MSEGbar * G + MSEG * gg) / (MSEGbar + MSEG)
    skews.append(GD)

    K_final, finalfreq_log, Gzero = freq_curve(XS, SS, GD, ktable)

    Galpha_k = Gzero[np.isclose(Gzero[:, 1], 0.05), 2][0]
    a = 1 - (Galpha_k**2) / (2 * (n_total - 1))
    b = K_final[:, 0] ** 2 - (Galpha_k**2) / n_total
    sqrt_term = np.maximum(0, K_final[:, 0] ** 2 - a * b)

    Ku = (K_final[:, 0] + np.sqrt(sqrt_term)) / a
    Kl = (K_final[:, 0] - np.sqrt(sqrt_term)) / a

    LQu_log = XS + Ku * SS
    LQl_log = XS + Kl * SS

    """
    pexp_interp = PchipInterpolator(pn_table[:, 1], pn_table[:, 2])
    expectedP_log = PchipInterpolator(
        pexp_interp(finalfreq_log[:, 0]), finalfreq_log[:, 1]
    )(finalfreq_log[:, 0])
    """

    p_unique = np.unique(pn_table[:, 1])
    pexp_vals = np.zeros(len(p_unique))

    for i, p_val in enumerate(p_unique):
        mask = pn_table[:, 1] == p_val
        n_vals = pn_table[mask, 0]
        pn_vals = pn_table[mask, 2]

        interp_func = PchipInterpolator(n_vals, pn_vals)
        pexp_vals[i] = interp_func(N - 1)

    pexp_table = np.vstack((p_unique, pexp_vals)).T

    # Find intersection and interpolate
    common_probs = np.intersect1d(pexp_table[:, 0], finalfreq_log[:, 0])
    if len(common_probs) > 0:
        mask_pexp = np.isin(pexp_table[:, 0], common_probs)
        mask_freq = np.isin(finalfreq_log[:, 0], common_probs)

        interp_func = PchipInterpolator(
            pexp_table[mask_pexp, 1], finalfreq_log[mask_freq, 1]
        )
        expectedP_log = interp_func(finalfreq_log[:, 0])

        expectedP_log[finalfreq_log[:, 0] < 0.002] = np.nan
    else:
        expectedP_log = np.full_like(finalfreq_log[:, 1], np.nan)

    dataout = np.vstack(
        [
            1.0 / finalfreq_log[:, 0],
            finalfreq_log[:, 0],
            10 ** finalfreq_log[:, 1],
            10**LQu_log,
            10**LQl_log,
            10**expectedP_log,
        ]
    ).T
    dataout = dataout[dataout[:, 0].argsort()]

    pp = plot_pos(non_zero_flood, n_total)

    return dataout, skews, pp, XS, SS, hp


if __name__ == "__main__":
    try:
        # MODIFIED: Load example data from NetCDF using xarray
        with xr.open_dataset("examples.nc") as ds:
            examples = ds["examples"].values

        ex1_mask = ~np.isnan(examples[:, 1])
        ex1_data = examples[ex1_mask][:, [0, 1]]

        b17(
            ex1_data,
            gg=0.590537,
            imgfile="b17_python_xarray_plot.png",
        )

    except FileNotFoundError:
        print(
            "\nCould not find 'examples.nc'. Please run the conversion script first."
        )
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
