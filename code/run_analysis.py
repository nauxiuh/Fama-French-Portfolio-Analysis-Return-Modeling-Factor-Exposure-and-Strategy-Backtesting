"""Run the full empirical analysis and produce figures + tables for the report.

This script consumes Data.csv, performs all econometric work, and writes:
  - /sessions/elegant-sharp-tesla/mnt/outputs/figures/*.pdf  (figures for LaTeX)
  - /sessions/elegant-sharp-tesla/mnt/outputs/tables/*.tex   (tables for LaTeX)
  - /sessions/elegant-sharp-tesla/mnt/outputs/tables/*.json  (raw numbers for the report writer)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_returns, build_factors, build_factors_loo, PORTFOLIO_NAMES
import econ_lib as el

import os
OUT = Path(os.environ.get("FMA4200_OUT", Path(__file__).resolve().parent.parent))
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.0,
})


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_json(name, obj):
    def _enc(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, pd.Series):
            return o.tolist()
        raise TypeError(type(o))
    with open(TAB / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=_enc)


def main():
    print("=== Loading data ===")
    rets = load_returns()
    factors = build_factors(rets)
    n_obs, n_port = rets.shape
    print(f"Loaded {n_obs} monthly observations on {n_port} portfolios.")

    write_json("data_overview", {
        "n_obs": n_obs,
        "start": rets.index.min(),
        "end": rets.index.max(),
        "portfolios": PORTFOLIO_NAMES,
    })

    # ============ Section 1: Descriptive statistics ============
    print("=== Descriptive statistics ===")
    desc_rows = []
    for c in PORTFOLIO_NAMES:
        x = rets[c].values
        jb_stat, jb_p = el.jarque_bera(x)
        lb_stat, lb_p = el.ljung_box(x, lags=12)
        lb2_stat, lb2_p = el.ljung_box(x ** 2, lags=12)
        arch_stat, arch_p = el.arch_lm(x - x.mean(), lags=5)
        adf = el.adf_test(x)
        desc_rows.append({
            "portfolio": c,
            "mean": x.mean(), "std": x.std(ddof=1),
            "min": x.min(), "max": x.max(),
            "skew": el.skewness(x), "ex_kurt": el.excess_kurtosis(x),
            "jb": jb_stat, "jb_p": jb_p,
            "lb12": lb_stat, "lb12_p": lb_p,
            "lb_sq12": lb2_stat, "lb_sq12_p": lb2_p,
            "arch5": arch_stat, "arch5_p": arch_p,
            "adf": adf["adf_stat"], "adf_lag": adf["lag"],
        })
    desc = pd.DataFrame(desc_rows).set_index("portfolio")
    desc.to_csv(TAB / "descriptive.csv")
    print(desc.round(3))
    write_json("descriptive", desc.reset_index().to_dict("records"))

    # ============ Figure: time-series of returns ============
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    for ax, c in zip(axes.ravel(), PORTFOLIO_NAMES):
        ax.plot(rets.index, rets[c], color="steelblue", linewidth=0.4)
        ax.set_title(c)
        ax.axhline(0, color="black", lw=0.4)
    save_fig(fig, "returns_timeseries")

    # ============ Figure: histograms with normal overlay ============
    fig, axes = plt.subplots(3, 2, figsize=(11, 7))
    for ax, c in zip(axes.ravel(), PORTFOLIO_NAMES):
        x = rets[c].values
        ax.hist(x, bins=60, density=True, color="steelblue", alpha=0.6, edgecolor="white")
        xs = np.linspace(x.min(), x.max(), 200)
        mu, sd = x.mean(), x.std()
        pdf = 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        ax.plot(xs, pdf, "r-", lw=1.2, label="Normal pdf")
        ax.set_title(f"{c}  (skew={el.skewness(x):.2f}, ex.kurt={el.excess_kurtosis(x):.2f})")
        ax.legend()
    save_fig(fig, "return_histograms")

    # ============ Figure: ACF / PACF of each portfolio ============
    fig, axes = plt.subplots(6, 2, figsize=(11, 12))
    for i, c in enumerate(PORTFOLIO_NAMES):
        x = rets[c].values
        a = el.acf(x, 24)
        p = el.pacf(x, 24)
        n = len(x)
        ci = 1.96 / np.sqrt(n)
        axes[i, 0].bar(range(len(a)), a, width=0.3, color="steelblue")
        axes[i, 0].axhline(ci, color="red", ls=":")
        axes[i, 0].axhline(-ci, color="red", ls=":")
        axes[i, 0].set_title(f"ACF of {c}")
        axes[i, 1].bar(range(len(p)), p, width=0.3, color="darkorange")
        axes[i, 1].axhline(ci, color="red", ls=":")
        axes[i, 1].axhline(-ci, color="red", ls=":")
        axes[i, 1].set_title(f"PACF of {c}")
    save_fig(fig, "acf_pacf")

    # ============ Section 2: ARMA model selection per portfolio ============
    print("=== ARMA model selection ===")
    arma_results = {}
    for c in PORTFOLIO_NAMES:
        sel = el.arma_select(rets[c].values, max_p=2, max_q=2, criterion="bic")
        best = sel["best"]
        arma_results[c] = {
            "p": int(best["p"]), "q": int(best["q"]),
            "mu": float(best["mu"]),
            "phi": [float(v) for v in best["phi"]],
            "theta": [float(v) for v in best["theta"]],
            "sigma2": float(best["sigma2"]),
            "aic": float(best["aic"]), "bic": float(best["bic"]),
            "log_lik": float(best["log_lik"]),
            "table": [[int(p), int(q), float(a), float(b), float(ll)] for p, q, a, b, ll in sel["table"]],
        }
        # residual diagnostics
        e = best["resid"]
        lb_stat, lb_p = el.ljung_box(e, 12)
        lb2_stat, lb2_p = el.ljung_box(e ** 2, 12)
        arch_stat, arch_p = el.arch_lm(e, lags=5)
        arma_results[c].update({
            "resid_lb": float(lb_stat), "resid_lb_p": float(lb_p),
            "resid_lb_sq": float(lb2_stat), "resid_lb_sq_p": float(lb2_p),
            "resid_arch": float(arch_stat), "resid_arch_p": float(arch_p),
        })
        print(f"  {c}: ARMA({best['p']},{best['q']}) BIC={best['bic']:.1f} ARCH-LM p={arch_p:.3f}")
    write_json("arma_results", arma_results)

    # ============ Figure: ARMA residual diagnostics for representative portfolio ============
    rep = "BIG.LoBM"
    fit = el.fit_arma(rets[rep].values, arma_results[rep]["p"], arma_results[rep]["q"])
    resid = fit["resid"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(resid, color="steelblue", lw=0.5)
    axes[0, 0].set_title(f"ARMA residuals: {rep}")
    axes[0, 0].axhline(0, color="black", lw=0.4)
    # ACF of residuals
    a = el.acf(resid, 24)
    ci = 1.96 / np.sqrt(len(resid))
    axes[0, 1].bar(range(len(a)), a, width=0.3, color="darkorange")
    axes[0, 1].axhline(ci, color="red", ls=":"); axes[0, 1].axhline(-ci, color="red", ls=":")
    axes[0, 1].set_title("ACF of residuals")
    # QQ plot
    sorted_r = np.sort((resid - resid.mean()) / resid.std())
    q_theory = []
    for i in range(len(sorted_r)):
        p = (i + 0.5) / len(sorted_r)
        # Inverse normal via Beasley-Springer
        q_theory.append(_inv_normal(p))
    axes[1, 0].scatter(q_theory, sorted_r, s=4, alpha=0.5)
    lims = [min(q_theory), max(q_theory)]
    axes[1, 0].plot(lims, lims, color="red", lw=1)
    axes[1, 0].set_title("Normal Q-Q of residuals")
    axes[1, 0].set_xlabel("Theoretical quantiles")
    # ACF of squared residuals
    a2 = el.acf(resid ** 2, 24)
    axes[1, 1].bar(range(len(a2)), a2, width=0.3, color="darkorange")
    axes[1, 1].axhline(ci, color="red", ls=":"); axes[1, 1].axhline(-ci, color="red", ls=":")
    axes[1, 1].set_title("ACF of squared residuals")
    save_fig(fig, "arma_diagnostics")

    # ============ Section 3: GARCH(1,1) on ARMA residuals ============
    print("=== GARCH(1,1) ===")
    garch_results = {}
    for c in PORTFOLIO_NAMES:
        # Fit ARMA, then GARCH on residuals
        res = arma_results[c]
        fit_a = el.fit_arma(rets[c].values, res["p"], res["q"])
        e = fit_a["resid"]
        g = el.fit_garch11(e)
        std_e = g["std_resid"]
        lb_std_sq_stat, lb_std_sq_p = el.ljung_box(std_e ** 2, 12)
        arch_std_stat, arch_std_p = el.arch_lm(std_e, 5)
        garch_results[c] = {
            "omega": g["omega"], "alpha": g["alpha"], "beta": g["beta"],
            "persistence": g["alpha"] + g["beta"],
            "uncond_var": g["uncond_var"],
            "log_lik": g["log_lik"], "aic": g["aic"], "bic": g["bic"],
            "std_lb_sq": lb_std_sq_stat, "std_lb_sq_p": lb_std_sq_p,
            "std_arch": arch_std_stat, "std_arch_p": arch_std_p,
        }
        print(f"  {c}: omega={g['omega']:.4f} alpha={g['alpha']:.3f} beta={g['beta']:.3f} (a+b={g['alpha']+g['beta']:.3f})")
    write_json("garch_results", garch_results)

    # ============ Figure: GARCH conditional volatility for representative portfolio ============
    fit_a = el.fit_arma(rets[rep].values, arma_results[rep]["p"], arma_results[rep]["q"])
    g = el.fit_garch11(fit_a["resid"])
    sigma = np.sqrt(g["sigma2"])
    e_idx = rets.index[max(arma_results[rep]["p"], arma_results[rep]["q"]):]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(e_idx, fit_a["resid"], color="steelblue", lw=0.4)
    axes[0].plot(e_idx, 2 * sigma, color="red", lw=0.5, label="±2σ_t (GARCH)")
    axes[0].plot(e_idx, -2 * sigma, color="red", lw=0.5)
    axes[0].set_title(f"ARMA residuals with ±2 GARCH(1,1) conditional SD bands: {rep}")
    axes[0].legend()
    axes[1].plot(e_idx, sigma, color="darkred", lw=0.6)
    axes[1].set_title("Estimated conditional standard deviation σ_t")
    save_fig(fig, "garch_conditional_vol")

    # ============ Section 4: Factor model + predictive regressions ============
    print("=== Factor models (LOO FF-3) and predictive regressions ===")
    # (a) Contemporaneous factor model using leave-one-out factors (legitimate fit, not prediction).
    factor_fit = {}
    for c in PORTFOLIO_NAMES:
        ff = build_factors_loo(rets, c)
        df = rets[[c]].join(ff).dropna()
        y = df[c].values
        X = np.column_stack([np.ones(len(df)), df["Mkt_RF"].values, df["SMB"].values, df["HML"].values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        sigma2_hat = ss_res / (len(y) - len(beta))
        cov = sigma2_hat * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tstat = beta / se
        names = ["alpha", "Mkt_RF", "SMB", "HML"]
        factor_fit[c] = {
            "coefs": dict(zip(names, beta.tolist())),
            "t_stats": dict(zip(names, tstat.tolist())),
            "r2": float(r2),
            "resid_std": float(np.sqrt(sigma2_hat)),
        }
        print(f"  {c}: alpha={beta[0]:+.3f} (t={tstat[0]:+.2f}), beta_M={beta[1]:.2f}, beta_S={beta[2]:+.2f}, beta_H={beta[3]:+.2f}, R²={r2:.3f}")
    write_json("factor_model", factor_fit)

    # (b) Genuine predictive regression with lagged exogenous variables.
    print("=== Predictive (lagged exogenous) regressions ===")
    macro = build_factors(rets).copy()
    rf_const = macro["RF"].iloc[0]
    # build lagged exog: Mkt_lag1, lagged SMB, lagged HML, lagged Term_spread
    exog_results = {}
    for c in PORTFOLIO_NAMES:
        ff = build_factors_loo(rets, c)
        df = pd.DataFrame({
            "y": rets[c],
            "y_lag1": rets[c].shift(1),
            "Mkt_lag1": ff["Mkt_RF"].shift(1),
            "SMB_lag1": ff["SMB"].shift(1),
            "HML_lag1": ff["HML"].shift(1),
            "Term_lag1": macro["Term_spread"].shift(1),
        }).dropna()
        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]
        cols = ["y_lag1", "Mkt_lag1", "SMB_lag1", "HML_lag1", "Term_lag1"]
        X_tr = np.column_stack([np.ones(len(train))] + [train[v].values for v in cols])
        y_tr = train["y"].values
        beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        y_hat = X_tr @ beta
        ss_res = np.sum((y_tr - y_hat) ** 2)
        ss_tot = np.sum((y_tr - y_tr.mean()) ** 2)
        r2_in = 1 - ss_res / ss_tot
        sigma2 = ss_res / (len(y_tr) - len(beta))
        cov = sigma2 * np.linalg.inv(X_tr.T @ X_tr)
        se = np.sqrt(np.diag(cov))
        tstat = beta / se
        X_te = np.column_stack([np.ones(len(test))] + [test[v].values for v in cols])
        y_te_hat = X_te @ beta
        y_te = test["y"].values
        oos_rmse = np.sqrt(np.mean((y_te - y_te_hat) ** 2))
        # AR(1) benchmark
        Xb = np.column_stack([np.ones(len(train)), train["y_lag1"].values])
        bb, *_ = np.linalg.lstsq(Xb, y_tr, rcond=None)
        y_te_pred_ar = bb[0] + bb[1] * test["y_lag1"].values
        oos_rmse_ar = np.sqrt(np.mean((y_te - y_te_pred_ar) ** 2))
        # naive: train mean
        oos_rmse_naive = np.sqrt(np.mean((y_te - y_tr.mean()) ** 2))
        # OOS R^2 vs naive
        oos_r2 = 1 - np.mean((y_te - y_te_hat) ** 2) / np.mean((y_te - y_tr.mean()) ** 2)
        exog_results[c] = {
            "coefs": dict(zip(["const"] + cols, beta.tolist())),
            "t_stats": dict(zip(["const"] + cols, tstat.tolist())),
            "r2_in_sample": float(r2_in),
            "oos_rmse_arx": float(oos_rmse),
            "oos_rmse_ar1": float(oos_rmse_ar),
            "oos_rmse_naive": float(oos_rmse_naive),
            "oos_r2_vs_naive": float(oos_r2),
        }
        print(f"  {c}: R²_in={r2_in:.4f}, RMSE OOS: ARX={oos_rmse:.3f} AR1={oos_rmse_ar:.3f} Naive={oos_rmse_naive:.3f} (OOS R²={oos_r2:+.4f})")
    write_json("exog_results", exog_results)

    # ============ Section 5: VAR + cointegration + statistical arbitrage ============
    print("=== VAR / cointegration ===")
    var2 = el.fit_var(rets.values, p=1)
    Sigma_VAR = var2["Sigma"]
    write_json("var_results", {
        "lag": 1,
        "Sigma": Sigma_VAR.tolist(),
        "beta_shape": list(var2["beta"].shape),
    })

    # Engle-Granger cointegration on cumulative log returns
    log_index = np.cumsum(np.log1p(rets.values / 100.0), axis=0)
    coint_pairs = []
    for i in range(n_port):
        for j in range(i + 1, n_port):
            res = el.engle_granger(log_index[:, i], log_index[:, j])
            crit = el.ADF_CRIT_CONST["5%"]
            coint_pairs.append({
                "y": PORTFOLIO_NAMES[i], "x": PORTFOLIO_NAMES[j],
                "alpha": res["alpha"], "beta": res["beta"],
                "adf_stat": res["adf_stat"], "adf_lag": res["adf_lag"],
                "cointegrated_5pct": bool(res["adf_stat"] < crit),
            })
    coint_df = pd.DataFrame(coint_pairs)
    print(coint_df.round(3))
    write_json("cointegration_pairs", coint_pairs)

    # pick the pair with most negative ADF stat (strongest evidence) for pairs trading
    best_pair = coint_df.iloc[coint_df["adf_stat"].idxmin()]
    print(f"Trading pair: {best_pair['y']} vs {best_pair['x']}  beta={best_pair['beta']:.3f}")

    # pairs trading backtest
    i = PORTFOLIO_NAMES.index(best_pair["y"])
    j = PORTFOLIO_NAMES.index(best_pair["x"])
    spread = log_index[:, i] - best_pair["alpha"] - best_pair["beta"] * log_index[:, j]
    # rolling z-score
    win = 24
    z = np.full(n_obs, np.nan)
    for t in range(win, n_obs):
        m = spread[t - win:t].mean()
        s = spread[t - win:t].std()
        z[t] = (spread[t] - m) / s if s > 0 else 0
    # signal: short spread when z>2, long when z<-2, close when |z|<0.5
    pos = np.zeros(n_obs)
    for t in range(1, n_obs):
        if pos[t - 1] == 0:
            if z[t] > 2:
                pos[t] = -1
            elif z[t] < -2:
                pos[t] = 1
            else:
                pos[t] = 0
        else:
            if abs(z[t]) < 0.5:
                pos[t] = 0
            else:
                pos[t] = pos[t - 1]
    # P&L: position taken at end of t, realised over t+1 spread change
    spread_ret = np.diff(spread, prepend=spread[0])  # ~ log-return of spread
    pnl = pos * spread_ret * 100  # convert to "percent"
    # We treat pnl as if a 1-unit dollar spread; this is a stylised illustration
    pairs_perf = el.performance_summary(pnl[24:])
    write_json("pairs_trading", {
        "pair": [best_pair["y"], best_pair["x"]],
        "beta": float(best_pair["beta"]),
        "performance": pairs_perf,
    })
    # equity curves
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rets.index, spread, label="spread (log)", color="steelblue", lw=0.6)
    ax.set_title(f"Cointegration spread: {best_pair['y']} − {best_pair['beta']:.2f}·{best_pair['x']}")
    ax.axhline(spread.mean(), color="black", lw=0.4)
    save_fig(fig, "pairs_spread")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rets.index, z, color="darkorange", lw=0.5)
    ax.axhline(2, color="red", ls=":"); ax.axhline(-2, color="red", ls=":")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title("24-month rolling z-score of the spread")
    save_fig(fig, "pairs_zscore")

    fig, ax = plt.subplots(figsize=(10, 4))
    cum_pnl = np.cumsum(pnl[24:])
    ax.plot(rets.index[24:], cum_pnl, color="seagreen", lw=0.7)
    ax.set_title("Cumulative P&L of the pairs trading strategy")
    save_fig(fig, "pairs_pnl")

    # ============ Section 6: Mean-variance analysis ============
    print("=== Mean-variance optimisation ===")
    mu = rets.mean().values
    Sigma = rets.cov().values
    ef = el.mv_efficient_frontier(mu, Sigma, n_points=120)
    write_json("frontier", {
        "mu": mu.tolist(), "Sigma": Sigma.tolist(),
        "w_tan": ef["w_tan"].tolist(), "mu_tan": float(ef["mu_tan"]),
        "sigma_tan": float(ef["sigma_tan"]),
        "w_gmv": ef["w_gmv"].tolist(), "mu_gmv": float(ef["mu_gmv"]),
        "sigma_gmv": float(ef["sigma_gmv"]),
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ef["sigmas"], ef["targets"], color="steelblue", label="Efficient frontier")
    ax.scatter(np.sqrt(np.diag(Sigma)), mu, color="black", s=20)
    for i, name in enumerate(PORTFOLIO_NAMES):
        ax.annotate(name, (np.sqrt(Sigma[i, i]), mu[i]), fontsize=7)
    ax.scatter([ef["sigma_tan"]], [ef["mu_tan"]], color="red", marker="*", s=120, label="Tangency")
    ax.scatter([ef["sigma_gmv"]], [ef["mu_gmv"]], color="darkorange", marker="o", s=60, label="GMV")
    ax.set_xlabel("Monthly std. dev. (%)")
    ax.set_ylabel("Monthly expected return (%)")
    ax.set_title("Efficient frontier — full sample")
    ax.legend()
    save_fig(fig, "efficient_frontier")

    # ============ Backtest: walk-forward MV strategies ============
    rets_arr = rets.values
    window = 60
    bk_tan = el.rolling_mv_backtest(rets_arr, window=window, strategy="tangency")
    bk_gmv = el.rolling_mv_backtest(rets_arr, window=window, strategy="gmv")
    bk_eq = el.rolling_mv_backtest(rets_arr, window=window, strategy="equal")
    bk_rp = el.rolling_mv_backtest(rets_arr, window=window, strategy="risk_parity")
    bk_tan_s = el.rolling_mv_backtest(rets_arr, window=window, strategy="tangency", shrink=True)
    bk_lo = el.rolling_mv_backtest(rets_arr, window=window, strategy="long_only")
    bk_lo_s = el.rolling_mv_backtest(rets_arr, window=window, strategy="long_only", shrink=True)

    perf = {
        "Plug-in Tangency": el.performance_summary(bk_tan["returns"]),
        "Plug-in GMV": el.performance_summary(bk_gmv["returns"]),
        "Equally Weighted": el.performance_summary(bk_eq["returns"]),
        "Risk Parity": el.performance_summary(bk_rp["returns"]),
        "Shrinkage Tangency": el.performance_summary(bk_tan_s["returns"]),
        "Long-only Tangency": el.performance_summary(bk_lo["returns"]),
        "Long-only + Shrinkage": el.performance_summary(bk_lo_s["returns"]),
    }
    print(pd.DataFrame(perf).round(4))
    write_json("backtest_performance", perf)

    # cumulative wealth chart
    def to_wealth(r):
        r = np.nan_to_num(r, nan=0.0)
        return np.cumprod(1 + r / 100.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    idx = rets.index
    for name, r in [
        ("Plug-in Tangency", bk_tan["returns"]),
        ("Plug-in GMV", bk_gmv["returns"]),
        ("Equally Weighted", bk_eq["returns"]),
        ("Risk Parity", bk_rp["returns"]),
        ("Shrinkage Tangency", bk_tan_s["returns"]),
        ("Long-only Tangency", bk_lo["returns"]),
        ("Long-only + Shrinkage", bk_lo_s["returns"]),
    ]:
        w = to_wealth(r)
        ax.plot(idx, w, label=name, lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Walk-forward backtest — wealth growth (window={window} months)")
    ax.set_ylabel("Cumulative wealth (log scale, $1 initial)")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "backtest_wealth")

    # Long-only weight stability
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name in enumerate(PORTFOLIO_NAMES):
        ax.plot(idx, bk_lo_s["weights"][:, k], label=name, lw=0.6)
    ax.set_title("Long-only + shrinkage tangency weights through time")
    ax.legend(ncol=3, fontsize=7)
    save_fig(fig, "weights_long_only_shrink")

    # weight stability chart
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name in enumerate(PORTFOLIO_NAMES):
        ax.plot(idx, bk_tan["weights"][:, k], label=name, lw=0.6)
    ax.set_title("Plug-in tangency weights through time")
    ax.legend(ncol=3, fontsize=7)
    save_fig(fig, "weights_tangency")

    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name in enumerate(PORTFOLIO_NAMES):
        ax.plot(idx, bk_tan_s["weights"][:, k], label=name, lw=0.6)
    ax.set_title("Shrinkage tangency weights through time")
    ax.legend(ncol=3, fontsize=7)
    save_fig(fig, "weights_shrink")

    print("Done.")


def _inv_normal(p: float) -> float:
    """Beasley-Springer inverse normal."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p > p_high:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    else:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


if __name__ == "__main__":
    main()
