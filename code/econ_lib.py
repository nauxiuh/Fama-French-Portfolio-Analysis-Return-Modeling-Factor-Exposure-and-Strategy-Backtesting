"""
Pure-numpy/pandas econometrics library for FMA4200 final project.

Implements (without scipy/statsmodels):
- Descriptive statistics: skewness, excess kurtosis, Jarque-Bera test, normal CDF
- ACF, PACF, Ljung-Box test, ARCH-LM test
- ADF test (regression-based, MacKinnon critical values)
- AR(p), MA(q), ARMA(p,q), ARIMA(p,d,q) estimation via conditional sum-of-squares + Nelder-Mead
- GARCH(1,1) via maximum likelihood + Nelder-Mead
- VAR(p) via OLS, Johansen-style pairwise cointegration via Engle-Granger
- Markowitz mean-variance frontier, tangency portfolio
- Out-of-sample backtest helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Tuple, Sequence, Optional, List, Dict

# ---------- Random / utility ----------

def standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Phi(z) via Abramowitz & Stegun approximation (no scipy)."""
    z = np.asarray(z, dtype=float)
    # erf approximation
    sign = np.sign(z)
    x = np.abs(z) / np.sqrt(2.0)
    # constants for erf
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    erf = sign * y
    return 0.5 * (1.0 + erf)

def chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared via series for incomplete gamma (small df)."""
    # Use regularised upper incomplete gamma Q(df/2, x/2) via continued fraction
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    # Use series for P(a,z) when z < a+1 else continued fraction for Q
    if z < a + 1.0:
        # series
        term = 1.0 / a
        total = term
        for n in range(1, 500):
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        # log gamma via Stirling
        log_gamma = _log_gamma(a)
        p = total * np.exp(-z + a * np.log(z) - log_gamma)
        return max(0.0, 1.0 - p)
    else:
        # continued fraction
        b = z + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        log_gamma = _log_gamma(a)
        q = np.exp(-z + a * np.log(z) - log_gamma) * h
        return max(0.0, min(1.0, q))

def _log_gamma(x: float) -> float:
    """Lanczos approximation."""
    g = 7
    p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _log_gamma(1 - x)
    x -= 1
    a = p[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += p[i] / (x + i)
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)

# ---------- Descriptive stats ----------

def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=0)
    return float(np.mean((x - m) ** 3) / (s ** 3))

def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=0)
    return float(np.mean((x - m) ** 4) / (s ** 4) - 3.0)

def jarque_bera(x: np.ndarray) -> Tuple[float, float]:
    n = len(x)
    S = skewness(x)
    K = excess_kurtosis(x)
    stat = n / 6.0 * (S ** 2 + (K ** 2) / 4.0)
    p = chi2_sf(stat, 2)
    return float(stat), float(p)

# ---------- ACF / PACF ----------

def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    c0 = np.dot(x, x) / n
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = np.dot(x[:-k], x[k:]) / n / c0
    return out

def pacf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """PACF via Levinson-Durbin recursion."""
    r = acf(x, nlags)
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    sigma2 = r[0]
    for k in range(1, nlags + 1):
        if k == 1:
            phi[1, 1] = r[1]
        else:
            num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
            denom = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
            phi[k, k] = num / denom
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_vals[k] = phi[k, k]
    return pacf_vals

def ljung_box(x: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    n = len(x)
    r = acf(x, lags)[1:]
    stat = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    p = chi2_sf(stat, lags)
    return float(stat), float(p)

def arch_lm(resid: np.ndarray, lags: int = 5) -> Tuple[float, float]:
    """Engle's ARCH-LM test based on squared residual regression."""
    r2 = resid ** 2
    n = len(r2)
    y = r2[lags:]
    X = np.column_stack([np.ones(n - lags)] + [r2[lags - i - 1 : n - i - 1] for i in range(lags)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot
    stat = (n - lags) * r_sq
    p = chi2_sf(stat, lags)
    return float(stat), float(p)

# ---------- ADF test ----------

# MacKinnon (1996) critical values for the constant-only ADF (5% level ~ -2.86, 1% ~ -3.43, 10% ~ -2.57)
ADF_CRIT_CONST = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

def adf_test(x: np.ndarray, max_lag: int = 12) -> Dict:
    """ADF with constant; lag chosen by AIC."""
    x = np.asarray(x, dtype=float)
    best = None
    for p in range(0, max_lag + 1):
        dx = np.diff(x)
        n = len(dx)
        if n - p - 1 <= 10:
            continue
        # ADF regression: Delta x_t = a + b * x_{t-1} + sum gamma_i Delta x_{t-i}
        y = dx[p:]
        x_lag = x[p:-1]
        X_cols = [np.ones(len(y)), x_lag]
        for i in range(1, p + 1):
            X_cols.append(dx[p - i : len(dx) - i])
        X = np.column_stack(X_cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sse = np.sum(resid ** 2)
        k = X.shape[1]
        nobs = len(y)
        sigma2 = sse / (nobs - k)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se_b = np.sqrt(cov[1, 1])
        t_stat = beta[1] / se_b
        aic = nobs * np.log(sse / nobs) + 2 * k
        if best is None or aic < best["aic"]:
            best = {"t_stat": t_stat, "aic": aic, "lag": p, "nobs": nobs}
    return {
        "adf_stat": best["t_stat"],
        "lag": best["lag"],
        "nobs": best["nobs"],
        "crit": ADF_CRIT_CONST,
    }

# ---------- Nelder-Mead optimiser ----------

def nelder_mead(
    f: Callable,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 2000,
    initial_step: float = 0.1,
) -> Tuple[np.ndarray, float]:
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        x[i] += initial_step if x[i] == 0 else initial_step * abs(x[i])
        simplex.append(x)
    fvals = [f(x) for x in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if np.std(fvals) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        xc = centroid + rho * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        # shrink
        new_simplex = [simplex[0]]
        new_fvals = [fvals[0]]
        for j in range(1, len(simplex)):
            xs = simplex[0] + sigma * (simplex[j] - simplex[0])
            new_simplex.append(xs)
            new_fvals.append(f(xs))
        simplex = new_simplex
        fvals = new_fvals
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

# ---------- ARMA(p,q) via conditional SS ----------

def arma_residuals(params: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
    n = len(y)
    mu = params[0]
    phi = params[1:1 + p] if p > 0 else np.array([])
    theta = params[1 + p:1 + p + q] if q > 0 else np.array([])
    e = np.zeros(n)
    start = max(p, q)
    for t in range(start, n):
        y_pred = mu
        for i in range(p):
            y_pred += phi[i] * (y[t - i - 1] - mu)
        for j in range(q):
            y_pred += theta[j] * e[t - j - 1]
        e[t] = y[t] - y_pred
    return e[start:]

def fit_arma(y: np.ndarray, p: int, q: int) -> Dict:
    y = np.asarray(y, dtype=float)
    n_params = 1 + p + q
    x0 = np.zeros(n_params)
    x0[0] = y.mean()
    if p > 0:
        x0[1] = 0.1
    if q > 0:
        x0[1 + p] = 0.1

    def nll(params):
        if p > 0 and np.any(np.abs(params[1:1+p]) >= 0.99):
            return 1e10
        if q > 0 and np.any(np.abs(params[1+p:]) >= 0.99):
            return 1e10
        e = arma_residuals(params, y, p, q)
        sigma2 = np.mean(e ** 2)
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return 1e10
        n = len(e)
        return 0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

    best, fmin = nelder_mead(nll, x0, tol=1e-7, max_iter=4000, initial_step=0.05)
    e = arma_residuals(best, y, p, q)
    sigma2 = np.mean(e ** 2)
    n_eff = len(e)
    log_lik = -fmin
    k = n_params + 1  # plus sigma^2
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n_eff) - 2 * log_lik
    return {
        "params": best,
        "p": p, "q": q,
        "mu": best[0],
        "phi": best[1:1+p] if p > 0 else np.array([]),
        "theta": best[1+p:1+p+q] if q > 0 else np.array([]),
        "sigma2": sigma2,
        "resid": e,
        "log_lik": log_lik,
        "aic": aic,
        "bic": bic,
        "n_eff": n_eff,
    }

def arma_select(y: np.ndarray, max_p: int = 3, max_q: int = 3, criterion: str = "aic") -> Dict:
    best = None
    table = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fit = fit_arma(y, p, q)
                table.append((p, q, fit["aic"], fit["bic"], fit["log_lik"]))
                if best is None or fit[criterion] < best[criterion]:
                    best = fit
            except Exception:
                continue
    return {"best": best, "table": table}

def arima_fit(y: np.ndarray, p: int, d: int, q: int) -> Dict:
    y = np.asarray(y, dtype=float)
    diffed = y.copy()
    for _ in range(d):
        diffed = np.diff(diffed)
    fit = fit_arma(diffed, p, q)
    fit["d"] = d
    fit["original_y"] = y
    return fit

# ---------- GARCH(1,1) ----------

def fit_garch11(resid: np.ndarray) -> Dict:
    r = np.asarray(resid, dtype=float)
    n = len(r)
    r2 = r ** 2

    def nll(params):
        omega, alpha, beta = params
        if omega <= 1e-10 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        sigma2 = np.empty(n)
        sigma2[0] = r2.mean()
        for t in range(1, n):
            sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
        if np.any(sigma2 <= 0):
            return 1e10
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r2 / sigma2)
        return -ll

    x0 = np.array([0.05 * r2.mean(), 0.1, 0.85])
    best, fmin = nelder_mead(nll, x0, tol=1e-7, max_iter=4000, initial_step=0.05)
    omega, alpha, beta = best
    sigma2 = np.empty(n)
    sigma2[0] = r2.mean()
    for t in range(1, n):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
    log_lik = -fmin
    k = 3
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "sigma2": sigma2,
        "std_resid": r / np.sqrt(sigma2),
        "log_lik": log_lik,
        "aic": aic, "bic": bic,
        "uncond_var": omega / max(1e-10, 1 - alpha - beta),
    }

# ---------- VAR(p) via OLS ----------

def fit_var(Y: np.ndarray, p: int) -> Dict:
    """Y: (T,k) returns. Returns coefficients and residuals."""
    Y = np.asarray(Y, dtype=float)
    T, k = Y.shape
    n = T - p
    X = np.ones((n, 1 + k * p))
    for i in range(p):
        X[:, 1 + i * k:1 + (i + 1) * k] = Y[p - i - 1:T - i - 1]
    Y_dep = Y[p:]
    beta, *_ = np.linalg.lstsq(X, Y_dep, rcond=None)
    resid = Y_dep - X @ beta
    Sigma = (resid.T @ resid) / (n - 1 - k * p)
    return {"beta": beta, "resid": resid, "Sigma": Sigma, "p": p, "n": n}

# ---------- Engle-Granger cointegration ----------

def engle_granger(y: np.ndarray, x: np.ndarray) -> Dict:
    """Test cointegration: regress y = a + b*x + e, ADF on e."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    adf = adf_test(e)
    return {"alpha": beta[0], "beta": beta[1], "resid": e, "adf_stat": adf["adf_stat"], "adf_lag": adf["lag"]}

# ---------- Mean-variance optimisation ----------

def mv_efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, n_points: int = 50) -> Dict:
    inv = np.linalg.inv(Sigma)
    ones = np.ones_like(mu)
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    Det = A * C - B ** 2
    mu_min, mu_max = mu.min() * 1.2, mu.max() * 1.2
    targets = np.linspace(mu_min, mu_max, n_points)
    sigmas, weights = [], []
    for t in targets:
        var_t = (A * t ** 2 - 2 * B * t + C) / Det
        sigmas.append(np.sqrt(max(var_t, 0)))
        lam = (C - B * t) / Det
        nu = (A * t - B) / Det
        w = inv @ (lam * ones + nu * mu)
        weights.append(w)
    # tangency portfolio (risk free = 0 for now)
    w_tan = inv @ mu / (ones @ inv @ mu)
    mu_tan = mu @ w_tan
    var_tan = w_tan @ Sigma @ w_tan
    # GMV
    w_gmv = inv @ ones / A
    mu_gmv = mu @ w_gmv
    var_gmv = w_gmv @ Sigma @ w_gmv
    return {
        "targets": targets,
        "sigmas": np.array(sigmas),
        "weights": np.array(weights),
        "w_tan": w_tan, "mu_tan": mu_tan, "sigma_tan": np.sqrt(var_tan),
        "w_gmv": w_gmv, "mu_gmv": mu_gmv, "sigma_gmv": np.sqrt(var_gmv),
        "A": A, "B": B, "C": C,
    }

def mv_tangency(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    inv = np.linalg.inv(Sigma)
    excess = mu - rf
    w = inv @ excess
    return w / np.sum(w)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto the probability simplex {w >= 0, sum(w)=1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u - cssv / (np.arange(1, n + 1)) > 0)[0][-1]
    lam = cssv[rho] / (rho + 1)
    return np.maximum(v - lam, 0)


def mv_long_only(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0,
                 max_iter: int = 300, lr: float = 0.1) -> np.ndarray:
    """Long-only tangency (max Sharpe) via projected gradient ascent on Sharpe ratio.

    Solves max_w (mu-rf)' w / sqrt(w' Sigma w)  s.t. w >= 0, 1' w = 1.
    """
    N = len(mu)
    w = np.ones(N) / N
    excess = mu - rf
    for _ in range(max_iter):
        denom = np.sqrt(max(w @ Sigma @ w, 1e-12))
        num = excess @ w
        # gradient of Sharpe wrt w
        grad = excess / denom - num * (Sigma @ w) / (denom ** 3)
        w_new = _project_simplex(w + lr * grad)
        if np.linalg.norm(w_new - w) < 1e-10:
            break
        w = w_new
    return w


def mv_long_only_gmv(Sigma: np.ndarray, max_iter: int = 1000, lr: float = 0.05) -> np.ndarray:
    """Long-only global minimum variance portfolio."""
    N = Sigma.shape[0]
    w = np.ones(N) / N
    for _ in range(max_iter):
        grad = 2 * Sigma @ w  # gradient of w' Sigma w
        w_new = _project_simplex(w - lr * grad)
        if np.linalg.norm(w_new - w) < 1e-10:
            break
        w = w_new
    return w

# ---------- Ledoit-Wolf shrinkage (simple constant correlation target) ----------

def ledoit_wolf_shrink(returns: np.ndarray) -> np.ndarray:
    """Returns shrunk covariance towards constant-correlation target."""
    X = returns - returns.mean(axis=0)
    T, N = X.shape
    sample = (X.T @ X) / T
    var = np.diag(sample)
    std = np.sqrt(var)
    corr = sample / np.outer(std, std)
    mask = ~np.eye(N, dtype=bool)
    r_bar = corr[mask].mean()
    target = r_bar * np.outer(std, std)
    np.fill_diagonal(target, var)
    # asymptotic shrinkage intensity
    pi_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            pi_mat[i, j] = np.mean((X[:, i] * X[:, j] - sample[i, j]) ** 2)
    pi_hat = pi_mat.sum()
    gamma = np.sum((sample - target) ** 2)
    # rho is small term approx 0 for simplicity (ignore)
    kappa = pi_hat / max(gamma, 1e-12)
    delta = max(0.0, min(1.0, kappa / T))
    return delta * target + (1 - delta) * sample, delta

# ---------- Backtest helpers ----------

def rolling_mv_backtest(
    returns: np.ndarray,
    window: int = 60,
    strategy: str = "tangency",
    rf: float = 0.0,
    shrink: bool = False,
) -> Dict:
    """Walk-forward: each month estimate mu, Sigma from past `window` months, hold 1 month."""
    T, N = returns.shape
    portfolio_ret = np.full(T, np.nan)
    weights_hist = np.full((T, N), np.nan)
    for t in range(window, T):
        sample = returns[t - window:t]
        mu = sample.mean(axis=0)
        if shrink:
            Sigma, _ = ledoit_wolf_shrink(sample)
        else:
            Sigma = np.cov(sample.T, ddof=1)
        if strategy == "tangency":
            try:
                w = mv_tangency(mu, Sigma, rf)
            except np.linalg.LinAlgError:
                w = np.ones(N) / N
        elif strategy == "gmv":
            try:
                inv = np.linalg.inv(Sigma)
                w = inv @ np.ones(N) / (np.ones(N) @ inv @ np.ones(N))
            except np.linalg.LinAlgError:
                w = np.ones(N) / N
        elif strategy == "equal":
            w = np.ones(N) / N
        elif strategy == "risk_parity":
            d = np.diag(Sigma)
            inv_sd = 1.0 / np.sqrt(np.maximum(d, 1e-12))
            w = inv_sd / inv_sd.sum()
        elif strategy == "long_only":
            w = mv_long_only(mu, Sigma, rf)
        elif strategy == "long_only_gmv":
            w = mv_long_only_gmv(Sigma)
        else:
            raise ValueError(strategy)
        portfolio_ret[t] = w @ returns[t]
        weights_hist[t] = w
    return {"returns": portfolio_ret, "weights": weights_hist}

def performance_summary(r: np.ndarray, freq: int = 12, rf: float = 0.0) -> Dict:
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {}
    mu = r.mean()
    sd = r.std(ddof=1)
    ann_ret = mu * freq
    ann_vol = sd * np.sqrt(freq)
    sharpe = (mu - rf) / sd * np.sqrt(freq) if sd > 0 else np.nan
    cum = np.cumprod(1 + r / 100.0)
    max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
    return {
        "mean_monthly": mu, "sd_monthly": sd,
        "ann_return": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "cum_return": cum[-1] - 1, "n": len(r),
    }
