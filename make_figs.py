import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.6, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
C1, C2, C3, C4 = "#1f6f8b", "#c05640", "#5b8c5a", "#7a6c9e"

df = pd.read_csv("dataset_multimodal.csv", index_col=0, parse_dates=True)
fomc = pd.read_csv("fomc_sentiment.csv", index_col=0, parse_dates=True)
tp = pd.read_csv("test_predictions.csv", index_col=0, parse_dates=True)
res = json.load(open("results.json"))

# ---------- Figure 1: the three modalities ----------
fig, ax = plt.subplots(3, 1, figsize=(7.0, 4.6), sharex=True,
                       gridspec_kw={"hspace": 0.42})
ax[0].plot(df.index, df["c"], color=C1, lw=0.8)
ax[0].set_ylabel("S&P 500 index")
ax[0].set_title("Market modality: S&P 500 level and Garman-Klass volatility", loc="left")
ax0b = ax[0].twinx()
ax0b.plot(df.index, df["gk_vol"], color=C2, lw=0.4, alpha=0.6)
ax0b.set_ylabel("GK volatility, ann. pct", color=C2)
ax0b.tick_params(axis="y", colors=C2)

ax[1].plot(df.index, df["vix"], color=C3, lw=0.7, label="VIX")
ax[1].plot(df.index, df["y10"] * 5, color=C4, lw=0.7, label="10-year yield (x5)")
ax[1].set_ylabel("Level")
ax[1].legend(loc="upper left", frameon=False, ncol=2)
ax[1].set_title("Macro-financial modality: implied volatility and rates", loc="left")

ax[2].plot(df.index, df["log_epu"], color="#888888", lw=0.4, alpha=0.8, label="log daily EPU")
ax2b = ax[2].twinx()
ax2b.scatter(fomc.index, fomc["fomc_pol"], s=8, color=C2, zorder=3, label="FOMC LM polarity")
ax2b.axhline(0, color=C2, lw=0.4, ls=":", alpha=0.5)
ax2b.set_ylabel("FOMC polarity", color=C2)
ax2b.tick_params(axis="y", colors=C2)
ax[2].set_ylabel("log EPU")
ax[2].set_title("Text modality: newspaper policy uncertainty and FOMC statement tone", loc="left")
ax[2].xaxis.set_major_locator(mdates.YearLocator(2))
for a in ax: a.margins(x=0.01)
fig.align_ylabels(ax)
fig.savefig("figs/fig1_modalities.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: pipeline diagram ----------
fig, ax = plt.subplots(figsize=(7.0, 2.7))
ax.axis("off")
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                 fc=fc, ec="#333333", lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=9, lw=0.8, color="#333333"))
box(0.01, 0.70, 0.24, 0.24, "Market modality\nS&P 500 OHLCV\nGK volatility, HAR terms,\nreturns, momentum", "#dce9ef")
box(0.01, 0.39, 0.24, 0.24, "Macro-financial modality\nVIX, 10-year Treasury,\nbroad dollar index\n(FRED)", "#e2efe1")
box(0.01, 0.08, 0.24, 0.24, "Text modality\nDaily EPU index,\nFOMC statements with\nLoughran-McDonald tone", "#f0e4df")
box(0.33, 0.37, 0.20, 0.30, "Feature fusion\n19 aligned daily\nfeatures, strict\nrelease-time alignment", "#eeeeee")
box(0.60, 0.55, 0.17, 0.30, "Models\nHAR-OLS, Ridge,\nRF, GBR, MLP", "#eeeeee")
box(0.60, 0.12, 0.17, 0.30, "Protocol\nExpanding window,\nannual refits\n2023 to 2026", "#eeeeee")
box(0.83, 0.34, 0.16, 0.34, "Evaluation\nRMSE, MAE, R2,\nQLIKE, Diebold-\nMariano tests", "#dce9ef")
for yy in (0.82, 0.51, 0.20): arrow(0.25, yy, 0.33, 0.52)
arrow(0.53, 0.55, 0.60, 0.66)
arrow(0.53, 0.47, 0.60, 0.30)
arrow(0.77, 0.68, 0.83, 0.56)
arrow(0.77, 0.26, 0.83, 0.44)
ax.text(0.43, 0.76, "Target: average realized (GK)\nvolatility over next 5 sessions",
        fontsize=7.2, ha="center", style="italic")
fig.savefig("figs/fig2_pipeline.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 3: test-period forecasts ----------
fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.plot(tp.index, tp["actual"], color="#444444", lw=0.9, label="Realized (target)")
ax.plot(tp.index, tp["har"], color=C1, lw=0.8, ls="--", label="HAR baseline")
ax.plot(tp.index, tp["ridge_b"], color=C2, lw=0.8, label="Ridge, market + macro")
ax.set_ylabel("5-day ahead volatility, ann. pct")
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.margins(x=0.01)
ax.set_title("Out-of-sample forecasts, January 2023 to July 2026", loc="left")
fig.savefig("figs/fig3_forecasts.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 4: ablation and importance ----------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
models = ["OLS", "Ridge", "RF", "GBR", "MLP"]
sets = ["A", "B", "C"]
labels = {"A": "A: market only", "B": "B: + macro", "C": "C: + text"}
x = np.arange(len(models)); w = 0.26
cols = {"A": C1, "B": C3, "C": C2}
for i, s in enumerate(sets):
    vals = [res["results"][f"{m}_{s}"]["R2"] for m in models]
    ax[0].bar(x + (i - 1) * w, vals, w, color=cols[s], label=labels[s])
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xticks(x); ax[0].set_xticklabels(models)
ax[0].set_ylabel("Out-of-sample R2")
ax[0].legend(frameon=False, fontsize=6.5)
ax[0].set_title("(a) Modality ablation by model", loc="left")

imp = pd.Series(res["importance"]).sort_values(ascending=True).tail(10)
pretty = {"vix": "VIX", "mom22": "22-day momentum", "dvix5": "5-day VIX change",
          "mom5": "5-day momentum", "absret": "absolute return", "rv_w": "weekly GK vol",
          "rv_d": "daily GK vol", "ret": "return", "days_since_fomc": "days since FOMC",
          "rv_m": "monthly GK vol", "fomc_subj": "FOMC subjectivity", "fomc_pol": "FOMC polarity"}
ax[1].barh([pretty.get(i, i) for i in imp.index], imp.values, color=C4)
ax[1].set_xlabel("Permutation importance (R2 drop)")
ax[1].set_title("(b) Top predictors, GBR full model", loc="left")
fig.tight_layout()
fig.savefig("figs/fig4_ablation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("figures saved:", __import__("os").listdir("figs"))
