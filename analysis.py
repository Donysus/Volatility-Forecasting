import json, re, html, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ---------- 1. PRICE MODALITY ----------
d = json.load(open("data/spx.json"))
r = d["chart"]["result"][0]
q = r["indicators"]["quote"][0]
px = pd.DataFrame({"ts": r["timestamp"], "o": q["open"], "h": q["high"],
                   "l": q["low"], "c": q["close"], "v": q["volume"]})
px["date"] = pd.to_datetime(px["ts"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
px = px.dropna().drop_duplicates("date").set_index("date").sort_index()

# Garman-Klass daily variance, annualized vol in percent
lnhl = np.log(px["h"] / px["l"])
lnco = np.log(px["c"] / px["o"])
gk_var = 0.5 * lnhl**2 - (2 * np.log(2) - 1) * lnco**2
px["gk_vol"] = np.sqrt(252 * gk_var.clip(lower=0)) * 100
px["ret"] = np.log(px["c"]).diff() * 100
px["absret"] = px["ret"].abs()
px["mom5"] = px["c"].pct_change(5) * 100
px["mom22"] = px["c"].pct_change(22) * 100
px["vol_z"] = (px["v"] - px["v"].rolling(60).mean()) / px["v"].rolling(60).std()

# HAR components on GK vol
px["rv_d"] = px["gk_vol"]
px["rv_w"] = px["gk_vol"].rolling(5).mean()
px["rv_m"] = px["gk_vol"].rolling(22).mean()

# Target: average GK vol over next 5 trading days (annualized pct)
fwd_var = (gk_var.clip(lower=0)).shift(-1).rolling(5).mean().shift(-4)
px["target"] = np.sqrt(252 * fwd_var) * 100

# ---------- 2. MACRO MODALITY ----------
def fred(f, name):
    s = pd.read_csv(f, na_values=".")
    s.columns = ["date", name]
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")[name]

vix = fred("data/vix.csv", "vix")
y10 = fred("data/dgs10.csv", "y10")
dxy = fred("data/dxy.csv", "dxy")
macro = pd.concat([vix, y10, dxy], axis=1).ffill()
macro["dvix5"] = macro["vix"].diff(5)
macro["dy10_5"] = macro["y10"].diff(5)
macro["ddxy5"] = macro["dxy"].pct_change(5) * 100

# ---------- 3. TEXT MODALITY ----------
# 3a. Daily EPU (newspaper-based)
epu = pd.read_csv("data/epu.csv")
epu["date"] = pd.to_datetime(epu[["year", "month", "day"]])
epu = epu.set_index("date")["daily_policy_index"].sort_index()
epu_df = pd.DataFrame({"log_epu": np.log(epu)})
epu_df["epu5"] = epu_df["log_epu"].rolling(5).mean()

# 3b. FOMC statements, Loughran-McDonald sentiment
import pysentiment2 as ps
lm = ps.LM()
rows = []
import glob, os
for f in sorted(glob.glob("data/fomc/monetary*a.htm")):
    dt = pd.to_datetime(re.search(r"(\d{8})", f).group(1))
    raw = open(f, errors="ignore").read()
    m = re.search(r'<div class="col-xs-12 col-sm-8 col-md-8">(.*?)</div>', raw, re.S)
    body = m.group(1) if m else raw
    txt = re.sub(r"<[^>]+>", " ", body)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    toks = lm.tokenize(txt)
    sc = lm.get_score(toks)
    rows.append({"date": dt, "fomc_pol": sc["Polarity"], "fomc_subj": sc["Subjectivity"],
                 "fomc_wc": np.log(max(len(txt.split()), 1)), "wc_raw": len(txt.split())})
fomc = pd.DataFrame(rows).set_index("date").sort_index()
fomc = fomc[~fomc.index.duplicated()]

# ---------- 4. MERGE ----------
df = px.copy()
df = df.join(macro[["vix", "dvix5", "y10", "dy10_5", "ddxy5"]], how="left").ffill()
df = df.join(epu_df, how="left").ffill()
# FOMC features effective from release day onward (release 2pm ET, target starts t+1)
fdaily = fomc[["fomc_pol", "fomc_subj", "fomc_wc"]].reindex(df.index, method="ffill")
last_stmt = pd.Series(fomc.index, index=fomc.index).reindex(df.index, method="ffill")
fdaily["days_since_fomc"] = (df.index - last_stmt).dt.days
df = df.join(fdaily)
df = df.loc["2015-01-01":].dropna()
print("Dataset:", df.index.min().date(), "to", df.index.max().date(), "n =", len(df))

SET_A = ["rv_d", "rv_w", "rv_m"]
SET_AP = SET_A + ["ret", "absret", "mom5", "mom22", "vol_z"]
SET_B = SET_AP + ["vix", "dvix5", "y10", "dy10_5", "ddxy5"]
SET_C = SET_B + ["log_epu", "epu5", "fomc_pol", "fomc_subj", "fomc_wc", "days_since_fomc"]
FSETS = {"A": SET_A, "B": SET_B, "C": SET_C}

# ---------- 5. MODELS, EXPANDING-WINDOW ANNUAL REFIT ----------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

def make_model(name):
    if name == "OLS":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "RF":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=7, n_jobs=-1)
    if name == "GBR":
        return GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=7)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                              max_iter=2000, random_state=7, early_stopping=True))
    raise ValueError

TEST_YEARS = [2023, 2024, 2025, 2026]
y = df["target"]

def run(model_name, cols):
    preds = []
    for yr in TEST_YEARS:
        tr = df.index.year < yr
        te = df.index.year == yr
        if te.sum() == 0:
            continue
        m = make_model(model_name)
        m.fit(df.loc[tr, cols], y[tr])
        p = pd.Series(m.predict(df.loc[te, cols]), index=df.index[te])
        preds.append(p)
    return pd.concat(preds)

def qlike(actual, pred):
    a2 = actual**2; p2 = np.maximum(pred, 1e-4) ** 2
    return float(np.mean(a2 / p2 - np.log(a2 / p2) - 1))

def metrics(actual, pred):
    e = actual - pred
    ss = float(np.sum(e**2)); sst = float(np.sum((actual - actual.mean())**2))
    return {"RMSE": float(np.sqrt(np.mean(e**2))), "MAE": float(np.mean(np.abs(e))),
            "R2": 1 - ss / sst, "QLIKE": qlike(actual, pred)}

results, all_preds = {}, {}
for mn in ["OLS", "Ridge", "RF", "GBR", "MLP"]:
    for fs, cols in FSETS.items():
        key = f"{mn}_{fs}"
        p = run(mn, cols)
        act = y.loc[p.index].dropna()
        p = p.loc[act.index]
        results[key] = metrics(act, p)
        all_preds[key] = p
        print(key, {k: round(v, 4) for k, v in results[key].items()})

actual = y.loc[all_preds["OLS_A"].index].dropna()

# Diebold-Mariano with HAC variance (h=5 forecast horizon)
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm
def dm_test(a, p1, p2, h=5):
    d = (a - p1)**2 - (a - p2)**2
    d = d.dropna().values
    T = len(d)
    dbar = d.mean()
    # Newey-West long-run variance, lag h-1
    L = h - 1
    g0 = np.var(d, ddof=0)
    s = g0
    for l in range(1, L + 1):
        gl = np.cov(d[l:], d[:-l], ddof=0)[0, 1]
        s += 2 * (1 - l / (L + 1)) * gl
    dm = dbar / np.sqrt(s / T)
    from scipy import stats
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(pval)

dm_out = {}
har = all_preds["OLS_A"]
for key in ["GBR_C", "RF_C", "Ridge_C", "MLP_C", "GBR_B", "GBR_A"]:
    dm_out[f"HAR_vs_{key}"] = dm_test(actual, har.loc[actual.index], all_preds[key].loc[actual.index])
best_b, best_c = all_preds["GBR_B"], all_preds["GBR_C"]
dm_out["GBRB_vs_GBRC"] = dm_test(actual, best_b.loc[actual.index], best_c.loc[actual.index])
for k, v in dm_out.items():
    print(k, "DM =", round(v[0], 3), "p =", round(v[1], 4))

# Permutation importance for best full model, final refit on pre-2026, scored on 2023 to 2025
best_name = max(results, key=lambda k: results[k]["R2"])
print("BEST:", best_name, results[best_name])
mn, fs = best_name.split("_")
cols = FSETS["C"]
tr = df.index.year < 2023; te = (df.index.year >= 2023) & (df.index.year <= 2026)
mfin = make_model("GBR"); mfin.fit(df.loc[tr, cols], y[tr])
pi = permutation_importance(mfin, df.loc[te, cols], y[te], n_repeats=20, random_state=7)
imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
print(imp.round(4))

# Summary statistics
summ = df[["ret", "gk_vol", "target", "vix", "y10", "log_epu", "fomc_pol"]].describe().T[["mean", "std", "min", "max"]]
print(summ.round(3))

out = {"results": results,
       "dm": {k: {"stat": v[0], "p": v[1]} for k, v in dm_out.items()},
       "importance": imp.to_dict(),
       "n_total": int(len(df)), "n_test": int(len(actual)),
       "date_min": str(df.index.min().date()), "date_max": str(df.index.max().date()),
       "n_fomc": int(len(fomc)), "fomc_wc_mean": float(fomc["wc_raw"].mean()),
       "summary": {i: {c: float(summ.loc[i, c]) for c in summ.columns} for i in summ.index},
       "best": best_name}
json.dump(out, open("results.json", "w"), indent=1)
df.to_csv("dataset_multimodal.csv")
fomc.to_csv("fomc_sentiment.csv")
print("saved results.json")
