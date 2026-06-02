"""Parse the Kenneth French 6-portfolio CSV and build a clean monthly returns DataFrame.

We also construct synthetic Fama-French 3-factor proxies and macro variables
internally from the 6-portfolio data (since external network access is unavailable):
- Mkt-RF: simple equal-weight average of the 6 portfolios (proxy for the market excess return)
- SMB: average of the 3 SMALL portfolios minus average of the 3 BIG portfolios
- HML: average of HiBM portfolios minus average of LoBM portfolios
- TBill (constant short rate proxy at 0.3% / month)
- Term spread proxy: 12-month rolling difference between BIG and SMALL average returns
- Lag-1 market return (for AR-X feature)
These are clearly documented as in-sample factor proxies inside the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import os; DATA_PATH = Path(os.environ.get("FMA4200_DATA", "Data.csv"))

PORTFOLIO_NAMES = [
    "SMALL.LoBM",  # small / low book-to-market (growth)
    "ME1.BM2",     # small / mid
    "SMALL.HiBM",  # small / value
    "BIG.LoBM",    # large / growth
    "ME2.BM2",     # large / mid
    "BIG.HiBM",    # large / value
]


def load_returns() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH, skiprows=15, header=0)
    # First column is "YYYYMM", others are returns
    raw.columns = ["yyyymm"] + PORTFOLIO_NAMES
    # Drop rows that are not numeric (footers, empty rows)
    raw = raw[raw["yyyymm"].astype(str).str.match(r"^\d{6}$").fillna(False)].copy()
    raw["yyyymm"] = raw["yyyymm"].astype(int)
    for c in PORTFOLIO_NAMES:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    # Build date index
    year = raw["yyyymm"] // 100
    month = raw["yyyymm"] % 100
    raw["date"] = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)
    raw = raw.set_index("date").drop(columns=["yyyymm"])
    # Drop -99.99 sentinel
    raw = raw.replace([-99.99, -999.0], np.nan).dropna()
    return raw


def build_factors_full(rets: pd.DataFrame) -> pd.DataFrame:
    """Construct FF-3 style factor proxies using all 6 portfolios.

    Used only for risk decomposition. For predictive modelling use
    `build_factors_loo` which leaves the target portfolio out.
    """
    rf = 0.3
    small = rets[["SMALL.LoBM", "ME1.BM2", "SMALL.HiBM"]].mean(axis=1)
    big = rets[["BIG.LoBM", "ME2.BM2", "BIG.HiBM"]].mean(axis=1)
    growth = rets[["SMALL.LoBM", "BIG.LoBM"]].mean(axis=1)
    value = rets[["SMALL.HiBM", "BIG.HiBM"]].mean(axis=1)
    mkt = rets.mean(axis=1)
    out = pd.DataFrame({
        "Mkt_RF": mkt - rf,
        "SMB": small - big,
        "HML": value - growth,
        "RF": rf,
    }, index=rets.index)
    return out


def build_factors_loo(rets: pd.DataFrame, target: str) -> pd.DataFrame:
    """Leave-one-out factor construction excluding the target portfolio.

    Mkt_RF = equal-weight mean of the other 5 portfolios − RF.
    SMB    = mean of (SMALL.* in the remaining 5) − mean of (BIG.* in the remaining 5)
    HML    = mean of HiBM in remaining 5 − mean of LoBM in remaining 5
    """
    rf = 0.3
    others = [c for c in rets.columns if c != target]
    mkt = rets[others].mean(axis=1)
    smalls = [c for c in others if c.startswith("SMALL") or c.startswith("ME1")]
    bigs = [c for c in others if c.startswith("BIG") or c.startswith("ME2")]
    lows = [c for c in others if "LoBM" in c]
    highs = [c for c in others if "HiBM" in c]
    smb = rets[smalls].mean(axis=1) - rets[bigs].mean(axis=1) if smalls and bigs else mkt * 0
    hml = rets[highs].mean(axis=1) - rets[lows].mean(axis=1) if lows and highs else mkt * 0
    out = pd.DataFrame({
        "Mkt_RF": mkt - rf,
        "SMB": smb,
        "HML": hml,
        "RF": rf,
    }, index=rets.index)
    return out


def build_factors(rets: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper returning full-sample factors plus lagged macros."""
    out = build_factors_full(rets).copy()
    mkt = out["Mkt_RF"] + out["RF"]
    big = rets[["BIG.LoBM", "ME2.BM2", "BIG.HiBM"]].mean(axis=1)
    small = rets[["SMALL.LoBM", "ME1.BM2", "SMALL.HiBM"]].mean(axis=1)
    out["Mkt_lag1"] = mkt.shift(1)
    out["Term_spread"] = big.rolling(12).mean() - small.rolling(12).mean()
    return out
