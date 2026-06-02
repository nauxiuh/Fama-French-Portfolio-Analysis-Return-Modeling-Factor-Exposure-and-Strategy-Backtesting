# Modelling the Six Size–Value Portfolios and Designing Trading Strategies on the Kenneth French Data Library (Jul. 1926 – Jan. 2026)

## Abstract

We study the monthly value-weighted returns of the six size–book-to-market portfolios from Kenneth French's Data Library over a 99.6-year sample. We characterise the distributional properties, fit a sequence of linear and conditional-heteroskedasticity time-series models, and use Fama–French style factor proxies as exogenous predictors. We then turn to portfolio construction: we examine multivariate dependence and pairwise cointegration, estimate the unconditional mean–variance frontier, and benchmark a battery of allocation rules out-of-sample. The plug-in mean–variance strategy collapses under estimation error (annualised Sharpe of 0.02 with extreme leverage), while a long-only reformulation combined with Ledoit–Wolf covariance shrinkage achieves a Sharpe of 0.69 and dominates the equally-weighted benchmark on a risk-adjusted basis. The global minimum-variance portfolio is the single best-performing rule (Sharpe = 0.76), confirming the well-known empirical superiority of strategies that rely on covariance information only.

---

## Repository Structure

```
├── code/
│   ├── run_analysis.py    # Driver: runs the full pipeline, writes all figures and tables
│   ├── econ_lib.py        # Econometrics library (ARMA, GARCH, VAR, cointegration, MV optimisation)
│   ├── load_data.py       # Data loading and Fama-French factor proxy construction
│   └── Data.csv           # Kenneth French 6-portfolio monthly returns (July 1926 to January 2026)
├── figures/               # Output: all plots as PDF and PNG
├── tables/                # Output: all numeric results as JSON and CSV
├── report_source/
│   ├── report.tex         # LaTeX source for the written report
│   └── refs.bib           # Bibliography
└── report.pdf             # Compiled report
```

---

## Data

The dataset is Kenneth French's 6 Portfolios Formed on Size and Book-to-Market (2×3), covering T = 1,195 months from July 1926 through January 2026. Portfolios are formed at the end of each June by sorting NYSE/AMEX/Nasdaq common stocks into two market-cap groups (SMALL, BIG) and three book-to-market groups:

```
SMALL.LoBM   small-cap growth
ME1.BM2      small-cap blend
SMALL.HiBM   small-cap value
BIG.LoBM     large-cap growth
ME2.BM2      large-cap blend
BIG.HiBM     large-cap value
```

Because the official Fama-French factor file is not fetched from the network, factor proxies are constructed via a **leave-one-out (LOO)** scheme: for each target portfolio *i*, the market, SMB, and HML proxies are built from the remaining five portfolios only. This prevents mechanical fit when the target appears on both sides of a regression.

---

## 1. Descriptive Statistics

For each portfolio: mean, standard deviation, skewness, excess kurtosis, Jarque-Bera normality test, Ljung-Box test on returns and squared returns (Q(12) and Q²(12)), Engle's ARCH-LM test at five lags, and the ADF stationarity test.

**Table 1: Descriptive statistics (monthly returns, %). p-values in parentheses.**

| Portfolio | Mean | SD | Skew | Ex. Kurt | JB | Q(12) | Q²(12) | ARCH(5) | ADF |
|---|---|---|---|---|---|---|---|---|---|
| SMALL.LoBM | 0.97 | 7.43 | 0.57 | 6.92 | 2451 (0.000) | 44.0 (0.000) | 435 (0.000) | 162 (0.000) | −9.61 |
| ME1.BM2 | 1.24 | 6.94 | 1.10 | 13.41 | 9192 (0.000) | 75.7 (0.000) | 670 (0.000) | 195 (0.000) | −9.72 |
| SMALL.HiBM | 1.42 | 8.08 | 1.96 | 20.85 | 22420 (0.000) | 87.9 (0.000) | 650 (0.000) | 194 (0.000) | −9.37 |
| BIG.LoBM | 0.96 | 5.27 | −0.13 | 5.20 | 1351 (0.000) | 24.1 (0.020) | 477 (0.000) | 120 (0.000) | −9.17 |
| ME2.BM2 | 0.96 | 5.60 | 1.16 | 17.20 | 15000 (0.000) | 66.3 (0.000) | 845 (0.000) | 196 (0.000) | −9.22 |
| BIG.HiBM | 1.21 | 7.08 | 1.44 | 17.45 | 15574 (0.000) | 54.9 (0.000) | 799 (0.000) | 245 (0.000) | −9.33 |

*ADF critical values: −3.43 (1%), −2.86 (5%), −2.57 (10%).*

Three patterns dominate. First, the historical size and value premia are clearly visible: SMALL.HiBM earns 1.42%/month while BIG.LoBM earns only 0.96%, a 46 bp/month spread (~5.6%/year). Second, distributions are far from normal: JB rejects at any conventional level for all six portfolios. Third, every series exhibits strong autocorrelation in squared returns and rejects the no-ARCH null at p < 10⁻²², motivating GARCH modelling.

![Return histograms](figures/return_histograms.png)

![Returns time series](figures/returns_timeseries.png)

*Periods of clustered volatility (1929–32, 1937, 1973–74, 1987, 2000–02, 2008–09, 2020) are visible in all panels.*

---

## 2. ARMA Model Selection

ARMA(p,q) models with p, q ∈ {0, 1, 2} are estimated by minimising the Bayesian Information Criterion. A custom Nelder-Mead optimiser drives the conditional sum-of-squares likelihood.

**Table 2: Best ARMA model by BIC, with residual diagnostics.**

| Portfolio | (p, q) | Log-Lik | AIC | BIC | LB(12) | p | ARCH(5) | p |
|---|---|---|---|---|---|---|---|---|
| SMALL.LoBM | (0,2) | −4072.6 | 8153 | 8174 | 13.8 | 0.32 | 146 | 0.000 |
| ME1.BM2 | (1,0) | −3992.1 | 7990 | 8005 | 36.9 | 0.000 | 186 | 0.000 |
| SMALL.HiBM | (2,1) | −4162.2 | 8334 | 8360 | 34.4 | 0.001 | 224 | 0.000 |
| BIG.LoBM | (0,1) | −3674.0 | 7354 | 7369 | 17.3 | 0.140 | 111 | 0.000 |
| ME2.BM2 | (2,2) | −3724.8 | 7462 | 7492 | 20.6 | 0.056 | 170 | 0.000 |
| BIG.HiBM | (2,2) | −4009.8 | 8032 | 8062 | 20.1 | 0.065 | 191 | 0.000 |

Selected orders are low, consistent with limited linear predictability of monthly returns. For four of the six portfolios the Ljung-Box test on residuals fails to reject white noise at 5%. Crucially, the ARCH-LM test rejects the no-ARCH null at any level for every portfolio's residuals — the ARMA filter removes linear dependence but leaves volatility clustering fully intact.

![ARMA diagnostics](figures/arma_diagnostics.png)

![ACF / PACF](figures/acf_pacf.png)

---

## 3. GARCH(1,1) on ARMA Residuals

A GARCH(1,1) is layered onto each portfolio's ARMA residuals:

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

The conditional Gaussian log-likelihood is maximised by Nelder-Mead.

**Table 3: GARCH(1,1) parameter estimates.**

| Portfolio | ω | α | β | α+β | Var∞ | Log-Lik | LB²(12) p | ARCH(5) p |
|---|---|---|---|---|---|---|---|---|
| SMALL.LoBM | 1.45 | 0.136 | 0.843 | 0.980 | 71.0 | −3926 | 0.96 | 0.83 |
| ME1.BM2 | 1.22 | 0.133 | 0.843 | 0.976 | 50.5 | −3777 | 0.92 | 0.80 |
| SMALL.HiBM | 1.29 | 0.131 | 0.849 | 0.980 | 63.6 | −3884 | 0.96 | 0.75 |
| BIG.LoBM | 0.69 | 0.131 | 0.849 | 0.980 | 34.7 | −3537 | 0.65 | 0.80 |
| ME2.BM2 | 0.67 | 0.131 | 0.848 | 0.979 | 31.4 | −3490 | 0.50 | 0.52 |
| BIG.HiBM | 1.26 | 0.136 | 0.833 | 0.969 | 41.3 | −3732 | 0.71 | 0.68 |

Persistence (α + β) is uniformly close to but below unity (range 0.969–0.980), indicating very long-lived volatility with mean-reverting unconditional variance. After GARCH filtering, Ljung-Box and ARCH-LM p-values on squared standardised residuals are all ≥ 0.50 for every portfolio — the GARCH(1,1) fully absorbs the conditional heteroskedasticity.

![GARCH conditional volatility](figures/garch_conditional_vol.png)

*The conditional volatility series visibly tracks major stress episodes: 1929–32, 1973–74, 1987, 2000–02, 2008–09, and 2020.*

---

## 4. Factor Models and Predictive Regressions

### 4.1 Contemporaneous Factor Model

Each portfolio is regressed on the LOO market (Mkt-RF), SMB, and HML proxies simultaneously — a risk decomposition, not a prediction exercise.

**Table 4: Contemporaneous LOO 3-factor regression. t-statistics in parentheses.**

| Portfolio | α | β_Mkt | β_SMB | β_HML | R² |
|---|---|---|---|---|---|
| SMALL.LoBM | −0.003 (−0.05) | 1.089 (88.5) | +0.955 (34.8) | −0.636 (−29.9) | 0.918 |
| ME1.BM2 | +0.401 (11.2) | 0.953 (135.2) | +0.407 (33.4) | +0.015 (1.4) | 0.969 |
| SMALL.HiBM | +0.444 (8.0) | 1.019 (92.4) | +0.730 (31.4) | +0.602 (32.5) | 0.945 |
| BIG.LoBM | +0.521 (9.0) | 0.842 (82.7) | −0.723 (−26.9) | −0.582 (−28.2) | 0.861 |
| ME2.BM2 | +0.235 (5.0) | 0.889 (101.6) | −0.509 (−29.1) | +0.075 (5.3) | 0.918 |
| BIG.HiBM | +0.240 (4.0) | 1.076 (90.6) | −0.558 (−24.4) | +0.594 (29.0) | 0.917 |

Factor loadings align precisely with economic intuition. SMB loadings are positive for the three SMALL portfolios and negative for the three BIG portfolios; HML loadings are positive for HiBM and negative for LoBM portfolios. R² is high throughout (0.86–0.97). SMALL.LoBM is the only portfolio without a statistically significant alpha.

### 4.2 Predictive Regression with Lagged Variables

The model uses only lagged inputs: lag-1 own return, lagged market, lagged SMB, lagged HML, and a lagged term-spread proxy. Estimated on the first 70% of the sample, evaluated on the last 30% (Apr. 1996 – Jan. 2026).

**Table 5: Predictive regression — out-of-sample RMSE and R².**

| Portfolio | In-sample R² | RMSE-ARX | RMSE-AR(1) | RMSE-Naive | OOS R² vs Naive |
|---|---|---|---|---|---|
| SMALL.LoBM | 0.050 | 7.155 | 7.031 | 6.989 | −0.048 |
| ME1.BM2 | 0.056 | 5.843 | 5.745 | 5.651 | −0.069 |
| SMALL.HiBM | 0.062 | 6.281 | 6.131 | 6.105 | −0.058 |
| BIG.LoBM | 0.015 | 4.632 | 4.569 | 4.555 | −0.034 |
| ME2.BM2 | 0.053 | 4.755 | 4.574 | 4.535 | −0.100 |
| BIG.HiBM | 0.041 | 5.839 | 5.683 | 5.661 | −0.064 |

OOS R² values are negative for all portfolios — the naive historical mean beats every forecasting model out-of-sample. In-sample R² values of 1.5–6.2% reflect noise fitting that does not generalise. This is consistent with Welch and Goyal (2008): the in-sample evidence for return predictability rarely survives out-of-sample testing at monthly frequency.

---

## 5. Cointegration and Pairs Trading

Cumulative log-return indices P_{i,t} = Σ log(1 + R_{i,s}/100) are tested pairwise using the Engle-Granger two-step procedure across all 15 portfolio pairs.

**Table 6: Engle-Granger cointegration test — all 15 pairs.**

*ADF critical values: −3.43 (1%), −2.86 (5%), −2.57 (10%).*

| y | x | β̂ | ADF t | Cointegrated (5%) |
|---|---|---|---|---|
| SMALL.LoBM | ME1.BM2 | 0.685 | −2.09 | No |
| SMALL.LoBM | SMALL.HiBM | 0.603 | −2.57 | No |
| SMALL.LoBM | BIG.LoBM | 0.924 | −1.52 | No |
| SMALL.LoBM | ME2.BM2 | 0.839 | −2.67 | No (10%) |
| **SMALL.LoBM** | **BIG.HiBM** | **0.705** | **−2.98** | **Yes** |
| ME1.BM2 | SMALL.HiBM | 0.879 | −1.46 | No |
| ME1.BM2 | BIG.LoBM | 1.349 | −0.63 | No |
| ME1.BM2 | ME2.BM2 | 1.224 | −2.56 | No |
| ME1.BM2 | BIG.HiBM | 1.026 | −2.02 | No |
| SMALL.HiBM | BIG.LoBM | 1.529 | −0.40 | No |
| SMALL.HiBM | ME2.BM2 | 1.390 | −2.15 | No |
| SMALL.HiBM | BIG.HiBM | 1.168 | −2.34 | No |
| BIG.LoBM | ME2.BM2 | 0.902 | −1.14 | No |
| BIG.LoBM | BIG.HiBM | 0.754 | −1.22 | No |
| ME2.BM2 | BIG.HiBM | 0.838 | −2.38 | No |

Only one of fifteen pairs — SMALL.LoBM vs BIG.HiBM (small-cap growth vs large-cap value) — passes the 5% Engle-Granger test. Statistical evidence for cointegration across the six portfolios is weak overall. The economic interpretation is that the six portfolios share a dominant common factor (the market) but differ in their long-run growth rates, so cumulative-return indices are not co-trended.

### Pairs Trading Strategy

On the strongest pair, the spread s_t = P₁,t − α̂ − β̂·P₆,t is standardised by a rolling 24-month mean and standard deviation. The z-score rule: enter short-spread when z > 2, long-spread when z < −2, close when |z| < 0.5.

| Metric | Value |
|---|---|
| Annualised return | −6.34% |
| Annualised volatility | 11.98% |
| Sharpe ratio | −0.53 |
| Max drawdown | −99.9% |

The strategy fails because the spread drifts: SMALL.LoBM and BIG.HiBM have substantial differences in their average cumulative growth, and the residual is not zero-mean stationary in any economically robust sense. **Marginal statistical evidence of cointegration is not sufficient for profitable pairs trading at the monthly frequency.**

![Pairs spread](figures/pairs_spread.png)

![Pairs z-score](figures/pairs_zscore.png)

![Pairs P&L](figures/pairs_pnl.png)

---

## 6. Mean-Variance Analysis and Backtesting

The unconditional efficient frontier is computed analytically from full-sample µ and Σ, and plotted alongside the six asset points, the global minimum-variance (GMV) portfolio, and the (unconstrained) tangency portfolio.

![Efficient frontier](figures/efficient_frontier.png)

*The tangency portfolio (red star) involves substantial short positions and high volatility; the GMV (orange dot) is interior to the asset cloud.*

### Walk-forward Backtest

Seven allocation rules are backtested using a 60-month rolling estimation window with monthly rebalancing over 1,135 months.

**Table 7: Walk-forward backtest performance. Annualised quantities use √12 scaling.**

| Strategy | Ann. Return (%) | Ann. Vol (%) | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Plug-in Tangency | 10.2 | 437.9 | 0.02 | −1.93 |
| Plug-in GMV | **12.0** | **15.7** | **0.76** | −0.77 |
| Equally Weighted | 14.1 | 22.0 | 0.64 | −0.68 |
| Risk Parity | 13.9 | 21.2 | 0.66 | −0.68 |
| Shrinkage Tangency | −49.0 | 664.1 | −0.07 | −133.2 |
| **Long-only Tangency** | **15.3** | **21.9** | **0.70** | **−0.63** |
| **Long-only + Shrinkage** | **15.2** | **21.9** | **0.69** | **−0.63** |

**Strategy ranking on Sharpe ratio:**
> GMV (0.76) > Long-only Tangency (0.70) ≈ Long-only + Shrinkage (0.69) > Risk Parity (0.66) > Equally Weighted (0.64) ≫ Plug-in Tangency (0.02) > Shrinkage Tangency (−0.07)

### Why Plug-in Tangency Fails

The standard error of a monthly mean over a 60-month window is approximately σ_i / √60 — around 0.7–1.1 percentage points for these portfolios, of the same order as the means themselves. The tangency portfolio is therefore dominated by the noisiest sample mean estimates, takes extreme leveraged positions (individual weights routinely exceed ±5), and is whipsawed by estimation noise. The unconstrained shrinkage tangency suffers the same problem: shrinkage stabilises Σ̂ but does not address noise in µ̂, so without a weight bound the strategy still leverages volatile mean estimates.

![Backtest wealth](figures/backtest_wealth.png)

![Tangency weights through time](figures/weights_tangency.png)

### Three Improvements

**(i) Long-only constraint.** Imposing w ≥ 0, 1ᵀw = 1 dramatically tames the tangency portfolio. Jagannathan and Ma (2003) show this is theoretically equivalent to shrinking extreme sample covariances toward zero. Empirically, realised annualised volatility drops from 438% to 22% and Sharpe improves from 0.02 to 0.70.

**(ii) Ledoit-Wolf covariance shrinkage.** The shrinkage estimator combines the sample covariance with a constant-correlation target:

```
Σ̂_LW = δ̂·Σ_target + (1 − δ̂)·Σ̂_sample
```

where optimal δ̂ ∈ [0,1] minimises the expected Frobenius distance to the true Σ. Applied together with the long-only constraint, it achieves a Sharpe of 0.69 with max drawdown of −63%.

**(iii) Global minimum-variance (GMV).** The GMV portfolio uses Σ̂ but ignores µ̂ altogether. It is the highest-Sharpe rule in the backtest (0.76), with the lowest realised volatility (15.7%) and shallowest drawdown (−77%), outperforming the equally-weighted benchmark on every risk-adjusted metric.

![Long-only + shrinkage weights](figures/weights_long_only_shrink.png)

*Long-only + shrinkage weights stay bounded in [0,1] and gradually rotate between size and value tilts, in sharp contrast to the unbounded plug-in tangency.*

---

## Conclusions

Three findings stand out.

**First, the conditional-variance story dominates the conditional-mean story.** Every portfolio exhibits strongly persistent volatility (α + β ≈ 0.98) captured well by GARCH(1,1) on top of a low-order ARMA mean. By contrast, lagged factor regressions and AR(1) models all fail to beat a naive mean forecast out-of-sample. Investors interested in tactical timing will struggle; those interested in risk forecasting will find GARCH(1,1) entirely adequate.

**Second, statistical-arbitrage opportunities across the six portfolios are limited.** Only one of fifteen pairs is cointegrated at 5%, and even that pair produces a losing z-score trade. The six portfolios share a dominant common factor but differ in long-run growth rates, so spreads are non-stationary or only marginally so.

**Third, plug-in mean–variance is dominated by virtually every alternative.** A 60-month rolling tangency portfolio achieves Sharpe of 0.02 with extreme leverage. Three different improvements — no-short-sale constraints, Ledoit-Wolf shrinkage, and ignoring the mean estimate altogether (GMV and risk parity) — all produce Sharpes between 0.66 and 0.76, comfortably above the 1/N benchmark.

The unifying message is that **estimation error dominates portfolio choice** with six risky assets and monthly data. Rules that downweight or eliminate the noisiest inputs (µ̂, leverage, large individual weights) earn a robustness premium that is large relative to any predictive signal extractable from lagged returns.

---

## Econometrics Library (econ_lib.py)

All routines are implemented from scratch in NumPy / Pandas / Matplotlib — no scipy or statsmodels:

- Standard normal CDF and chi-squared survival function (Abramowitz and Stegun approximations)
- Skewness, excess kurtosis, Jarque-Bera test
- ACF, PACF via Levinson-Durbin recursion
- Ljung-Box test, Engle ARCH-LM test
- ADF test with AIC-based lag selection and MacKinnon critical values
- Nelder-Mead simplex optimiser
- ARMA(p,q) via conditional sum-of-squares, ARIMA(p,d,q)
- GARCH(1,1) via maximum likelihood
- VAR(p) via OLS
- Engle-Granger cointegration test
- Markowitz efficient frontier, tangency and GMV portfolios
- Ledoit-Wolf shrinkage towards constant-correlation target
- Long-only tangency via projected gradient ascent on Sharpe ratio
- Walk-forward backtesting engine
- Performance summary (annualised return, volatility, Sharpe, max drawdown)

---

## Dependencies

```bash
pip install numpy pandas matplotlib
```

No other packages are required.

## Usage

```bash
python code/run_analysis.py
```

Results are written to `figures/` (PDF and PNG) and `tables/` (JSON and CSV). Runtime is approximately 40 seconds on a standard laptop. The output directory can be overridden with the `FMA4200_OUT` environment variable; the data path with `FMA4200_DATA`.

---

## References

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.

Campbell, J. Y., & Shiller, R. J. (1988). The dividend-price ratio and expectations of future dividends and discount factors. *Review of Financial Studies*, 1(3), 195–228.

Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample. *Review of Financial Studies*, 21(4), 1509–1531.

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies*, 22(5), 1915–1953.

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of U.K. inflation. *Econometrica*, 50(4), 987–1007.

Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.

Jagannathan, R., & Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *Journal of Finance*, 58(4), 1651–1683.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.

Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60–70.

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.
