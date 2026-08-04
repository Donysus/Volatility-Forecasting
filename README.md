# Multimodal Volatility Forecasting

Replication code and data for the working paper **"How Much Does Each Modality Matter? An Empirical Study of Market, Macro-Financial, and Text Signals in Short-Horizon Equity Volatility Forecasting"**

**Paper:** [SSRN 7110278](https://ssrn.com/abstract=7110278) · DOI: 10.2139&#8203;.ssrn.7110278
**Author:** Raghav Jha, Delhi Technological University

---

## What this study asks

Multimodal machine learning is widely reported to improve financial forecasting, but the incremental value of each individual data source is rarely measured under a common out-of-sample protocol. This project quantifies how much forecast accuracy each additional modality actually buys when predicting short-horizon S&P 500 volatility.

Three nested modality sets are compared:

| Set | Contents |
|---|---|
| **A** | Market only: HAR terms built from Garman-Klass realized volatility |
| **B** | A plus macro-financial: VIX, ten-year Treasury yield, broad dollar index |
| **C** | B plus text: daily Economic Policy Uncertainty index, FOMC statement tone |

Five model classes are estimated on each set: OLS in HAR form, ridge regression, random forest, gradient boosting, and a multilayer perceptron.

## Main findings

- **Macro-financial data helps, and mostly through one variable.** Moving ridge regression from Set A to Set B raises out-of-sample R-squared from 0.300 to 0.369 and cuts QLIKE loss by 12.4 percent. Permutation importance attributes most of this to the VIX alone. The gains are concentrated in the 2025 stress episode rather than spread evenly.
- **Text adds nothing measurable once implied volatility is present.** The Set B to Set C increment is negative for the linear models, and all six text features rank near zero in importance. Option prices appear to subsume policy-related text information by the daily close.
- **Complexity does not pay at this sample size.** Tree ensembles and the neural network trail the regularized linear models on every pooled criterion. The widest specification is significantly worse than the HAR baseline.
- **The positive result is not statistically significant.** The headline improvement carries a Diebold-Mariano p-value of 0.40 under squared-error loss and 0.15 under QLIKE. The negative results are decisive; the positive one is suggestive. This is reported as-is rather than overstated.

## Data

All inputs are public. No licensed or proprietary data is used.

| Modality | Series | Source |
|---|---|---|
| Market | S&P 500 daily open, high, low, close, volume | Yahoo Finance |
| Macro-financial | VIX, ten-year Treasury yield, broad dollar index | FRED |
| Text | Daily Economic Policy Uncertainty index | Baker, Bloom, and Davis public distribution |
| Text | 98 FOMC statements, January 2015 to June 2026 | Federal Reserve press-release archive |

Sample period: January 28, 2015 to July 10, 2026 (2,879 trading days). Test period: January 2023 to July 2026 (882 days).

## Method

The forecast target is average realized volatility over the next five trading sessions, measured with the Garman-Klass range estimator and annualized in percent.

Evaluation uses an **expanding window with annual refits**. For each test year, models train only on observations dated before January 1 of that year. Scalers are fit on training data only and every rolling feature uses trailing windows, so no information dated on or after the forecast origin enters any transformation.

Accuracy is reported as RMSE, MAE, out-of-sample R-squared, and QLIKE loss, which is robust to noise in the volatility proxy. Equal predictive accuracy against the HAR baseline is tested with Diebold-Mariano statistics using Newey-West long-run variance with four lags.

## Repository contents

| File | Purpose |
|---|---|
| `analysis.py` | Builds the multimodal dataset, runs all fifteen model-by-modality combinations, computes metrics and DM tests |
| `addendum.py` | Additional DM tests and per-year robustness breakdown |
| `make_figs.py` | Generates the four publication figures |
| `dataset_multimodal.csv` | The assembled feature dataset |
| `fomc_sentiment.csv` | Loughran-McDonald scores for all 98 FOMC statements |
| `test_predictions.csv` | Out-of-sample forecasts from the baseline and best multimodal model |
| `results.json` | All metrics, test statistics, and importance scores reported in the paper |
| `paper.tex` | LaTeX source of the manuscript |

## Reproducing the results

```bash
pip install -r requirements.txt
python analysis.py
python addendum.py
python make_figs.py
```

`analysis.py` reads the raw source files and writes `results.json`, which contains every number reported in the paper. Note that the raw FOMC statement pages and price data are fetched from their original sources; see the data table above.

## Limitations

Volatility is measured from daily ranges rather than intraday returns, which adds proxy noise. The text representation is deliberately simple, so the null result applies to lexicon-scored daily text and not to text in general; transformer embeddings could plausibly recover signal that this design misses. Hyperparameters are held fixed across modality sets rather than tuned per window, which is symmetric across comparisons but may understate the ceiling of the flexible models. The study covers one index in one market, and no trading strategy or transaction-cost analysis is attempted.

## Citation

```
Jha, Raghav, "How Much Does Each Modality Matter? An Empirical Study of Market,
Macro-Financial, and Text Signals in Short-Horizon Equity Volatility Forecasting"
(July 13, 2026). Available at SSRN: https://ssrn.com/abstract=7110278
```

## License

Code released under the MIT License. The manuscript is subject to the SSRN posting license.
