# Multiscale Inefficiency Index

Quantifying market inefficiency by combining **true multifractality** (spectrum width from MF-DFA surrogates) with **rolling Hurst deviations** (Lo’s modified R/S). Includes reproducible code to compute the index, plot figures, and backtest a simple long/short filter strategy.

---

## Overview

This repo implements the pipeline from the paper *“Multiscale Inefficiency Index”*:

- Estimate the Hurst exponent using **R/S** and **Lo’s modified R/S**.  
- Run **MF-DFA** to get \(h(q)\), Hölder exponents \(\alpha(q)\), and the multifractal spectrum \(f(\alpha)\).  
- Build a **surrogate** series (phase randomization) to isolate **true** multifractality from distributional effects.  
- Define the **Inefficiency Index**  
  I = Delta Alpha surrogate (width of Multifractal spectrum surrogate transform) * |Rolling Hurst - 0.5|.
- Use I to **filter Hurst signals** in a long/short strategy and compare to a Hurst-only baseline.


---

## Data

- **Indices:** S&P 500, Russell 2000, FTSE 100, Nikkei 225, DAX, SSEC.  
- **Sampling:** monthly (for the broad analysis) and daily for selected MF-DFA deep-dives and backtest.  
- **Span:** September 10, 1987 → February 28, 2025 (exact span depends on index & frequency).  
- **Pre-processing:** log prices → ADF test → first-difference (returns) for stationarity.

---

## Methods 

- **R/S & Modified R/S (Lo, 1991)**  
  - R/S gives a scaling estimate. 
  - **Modified R/S** accounts for short-term autocorrelation (Newey–West style denominator) and has known limits for testing long memory via statistic \(V\). Critical values (one-tailed): 10% = 1.620, 5% = 1.747, 0.5% = 2.098.

- **MF-DFA (Kantelhardt et al., 2002)**  
  - Compute \(F_q(s)\) across scales; slopes on log-log give \(h(q)\).  
  - Legendre transform → \(\alpha(q)\), \(f(\alpha)\).  
  - **Surrogate & shuffled** series to separate correlation-driven multifractality from heavy-tails / finite-sample artifacts; \(\Delta\alpha_{\text{surrogate}}\) quantifies **true** multifractality. 

- **Inefficiency Index**  
  - I = Delta Alpha surrogate (width of Multifractal spectrum surrogate transform) * |Rolling Hurst - 0.5|.

---

## Results

The strategy uses the **Inefficiency Index** I as a filter on Hurst-based long/short signals.

- **Signal rule**:  
  - If I is above the rolling 6-month **1.5σ** threshold **and** \(H < 0.5\) → **short**  
  - Else → **long**
- Benchmarks:  
  - **Long-Only**: always long  
  - **Long/Short without inefficiency**: Hurst-only signal (no I filter). Long H > 0.5, short H < 0.5.

**Backtest parameters**:
- Asset: SSEC (Shanghai Composite)
- Frequency: daily
- Horizon: same rolling window for H and I
- Transactions Costs: zero

**Results**:

| Strategy                           | Ann. Return | Ann. Vol | Sharpe | MaxDD    |
|------------------------------------|------------:|---------:|-------:|---------:|
| Long/Short **with** inefficiency   | 9.521       | 23.307   | 0.409  | -56.474  |
| Long-Only                          | 3.326       | 23.316   | 0.143  | -71.985  |
| Long/Short **without** inefficiency| 5.075       | 23.314   | 0.218  | -62.687  |

---

**Conclusion:**  
The Inefficiency Index acts as a **signal quality filter**, avoiding trades during periods where Hurst signals are less reliable. It improves both raw and risk-adjusted returns, while also reducing worst-case losses. The improvement is non-trivial, especially for volatile indices like SSEC