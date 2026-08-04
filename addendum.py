import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats

df = pd.read_csv("dataset_multimodal.csv", index_col=0, parse_dates=True)
y = df["target"]
SET_A = ["rv_d", "rv_w", "rv_m"]
SET_AP = SET_A + ["ret", "absret", "mom5", "mom22", "vol_z"]
SET_B = SET_AP + ["vix", "dvix5", "y10", "dy10_5", "ddxy5"]
SET_C = SET_B + ["log_epu", "epu5", "fomc_pol", "fomc_subj", "fomc_wc", "days_since_fomc"]

def run(model, cols):
    preds = []
    for yr in [2023, 2024, 2025, 2026]:
        tr = df.index.year < yr; te = df.index.year == yr
        m = make_pipeline(StandardScaler(), model())
        m.fit(df.loc[tr, cols], y[tr])
        preds.append(pd.Series(m.predict(df.loc[te, cols]), index=df.index[te]))
    return pd.concat(preds)

har = run(LinearRegression, SET_A)
ridge_b = run(lambda: Ridge(alpha=10.0), SET_B)
ridge_c = run(lambda: Ridge(alpha=10.0), SET_C)
act = y.loc[har.index].dropna()
har, ridge_b, ridge_c = har.loc[act.index], ridge_b.loc[act.index], ridge_c.loc[act.index]

def dm(a, p1, p2, h=5):
    d = ((a - p1)**2 - (a - p2)**2).values
    T, L = len(d), h - 1
    s = np.var(d, ddof=0)
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=0)[0, 1]
    st = d.mean() / np.sqrt(s / T)
    return float(st), float(2 * (1 - stats.norm.cdf(abs(st))))

def mets(a, p):
    e = a - p
    return {"RMSE": float(np.sqrt((e**2).mean())), "MAE": float(e.abs().mean()),
            "R2": float(1 - (e**2).sum() / ((a - a.mean())**2).sum())}

extra = {"DM_HAR_vs_RidgeB": dm(act, har, ridge_b),
         "DM_HAR_vs_RidgeC": dm(act, har, ridge_c),
         "DM_RidgeB_vs_RidgeC": dm(act, ridge_b, ridge_c),
         "n_test": int(len(act))}
peryear = {}
for yr in [2023, 2024, 2025, 2026]:
    ix = act.index.year == yr
    peryear[yr] = {"n": int(ix.sum()),
                   "HAR": mets(act[ix], har[ix]),
                   "RidgeB": mets(act[ix], ridge_b[ix])}
for k, v in extra.items(): print(k, v)
for yr, v in peryear.items():
    print(yr, "n", v["n"], "HAR R2", round(v["HAR"]["R2"], 3), "RidgeB R2", round(v["RidgeB"]["R2"], 3))

res = json.load(open("results.json"))
res["extra_dm"] = {k: v for k, v in extra.items()}
res["per_year"] = {str(k): v for k, v in peryear.items()}
json.dump(res, open("results.json", "w"), indent=1)
har.to_frame("har").join(ridge_b.to_frame("ridge_b")).join(act.to_frame("actual")).to_csv("test_predictions.csv")
print("saved")
