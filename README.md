# === PEAD pipeline (USD, MSCI members, Size×BM peers, exclude-self, light trim, min-peer fallback) ===
# Input: df_daily with at least these columns (rename here if needed)
COL_ID        = "sedolcd"
COL_DATE      = "date"
COL_CCY       = "instrmtccy"
COL_PRICE     = "quoteclose"
COL_MKTCAP    = "marketvalue"
COL_TR        = "totalreturn"          # daily simple return; else fallback to closereturn or pct_change
COL_PR_RET    = "closereturn"
COL_BPS       = "wsf_bps_ltm"          # else "wsf_bps_qtr"
COL_BPS_RPT   = "wsf_bps_rpt_date_qtr" # else fallback to EPS rpt
COL_EPS_RPT   = "wsf_eps_rpt_date_qtr"
COL_EPS_ACT   = "wsf_eps_qtr"
COL_IBES_MEAN = "ibes_eps_mean_qtr"
COL_MSCI      = "MSCI-WRLD"            # point-in-time weight (positive if member)

# --- knobs ---
NBINS     = 4          # 3 or 4 buckets for Size & BM
MIN_CELL  = 20         # min peers; fallback to Size-only if smaller
TRIM_LO   = 0.01       # light trimming
TRIM_HI   = 0.99

import numpy as np
import pandas as pd

# ---------- 0) FILTER & BASICS ----------
df = df_daily.copy()
df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
for c in [COL_BPS_RPT, COL_EPS_RPT]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

df = df[df[COL_CCY] == "USD"].sort_values([COL_ID, COL_DATE]).reset_index(drop=True)

# daily return
if COL_TR in df.columns and df[COL_TR].notna().any():
    df["ret_d"] = df[COL_TR]
elif COL_PR_RET in df.columns and df[COL_PR_RET].notna().any():
    df["ret_d"] = df[COL_PR_RET]
else:
    df["ret_d"] = df.groupby(COL_ID)[COL_PRICE].pct_change()

# trading-day rank per id
df["td_rank"] = df.groupby(COL_ID).cumcount()

# ---------- 1) EOM PANEL (last trading day in each month) ----------
def last_valid(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

px_eom = (
    df.groupby([COL_ID, df[COL_DATE].dt.to_period("M")], as_index=False)
      .agg(month_end=(COL_DATE, "max"),
           quoteclose=(COL_PRICE, last_valid),
           marketvalue=(COL_MKTCAP, last_valid),
           msci_w=(COL_MSCI, last_valid))
      .assign(is_member=lambda d: d["msci_w"].fillna(0) > 0)
      .sort_values([COL_ID, "month_end"])
      .reset_index(drop=True)
)

# map month period -> month_end trading date for each id (for daily prev_eom mapping)
eom_map = px_eom.assign(month_period=px_eom["month_end"].dt.to_period("M"))[[COL_ID, "month_period", "month_end"]]

# ---------- 2) FUNDAS GATING → available_from for BPS ----------
# pick columns (fallbacks)
if COL_BPS not in df.columns:
    COL_BPS = "wsf_bps_qtr"
if COL_BPS_RPT not in df.columns:
    COL_BPS_RPT = COL_EPS_RPT

fundas_bps = (df[[COL_ID, COL_BPS, COL_BPS_RPT]]
              .dropna(subset=[COL_BPS, COL_BPS_RPT])
              .drop_duplicates(subset=[COL_ID, COL_BPS_RPT])
              .rename(columns={COL_BPS: "bps_val", COL_BPS_RPT: "rpt_date"})
              .reset_index(drop=True))

# map rpt_date -> next trading day (strictly after) per id
def map_next_trading_day(events_df, strict_after=True):
    side = "right" if strict_after else "left"
    out = []
    for sid, ev in events_df.groupby(COL_ID, sort=False):
        g  = ev.sort_values("rpt_date").copy()
        cal= df.loc[df[COL_ID]==sid, [COL_DATE]].drop_duplicates().sort_values(COL_DATE)
        if cal.empty: 
            continue
        dates = cal[COL_DATE].values
        idx   = np.searchsorted(dates, g["rpt_date"].values, side=side)
        ok    = idx < len(dates)
        gg    = g.loc[ok].copy()
        gg["available_from"] = pd.to_datetime(dates[idx[ok]])
        out.append(gg[[COL_ID, "rpt_date", "bps_val", "available_from"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[COL_ID,"rpt_date","bps_val","available_from"])

available = map_next_trading_day(fundas_bps.rename(columns={"rpt_date":"rpt_date"}), strict_after=True)

# BPS as-of month_end (last available_from ≤ month_end), per id (robust, no asof errors)
def asof_right_by_id(left_df, right_df, left_on, right_on, by, value_cols):
    out = []
    for sid, L in left_df.groupby(by, sort=False):
        R = right_df[right_df[by]==sid].sort_values(right_on)
        if R.empty: 
            continue
        L = L.sort_values(left_on)
        rvals = R[right_on].values
        idx = np.searchsorted(rvals, L[left_on].values, side="right") - 1
        m = idx >= 0
        tmp = L.loc[m, [by, left_on]].copy()
        for c in value_cols:
            tmp[c] = R.iloc[idx[m]][c].values
        # keep right_on value too (for audit)
        tmp[f"{right_on}_asof"] = pd.to_datetime(rvals[idx[m]])
        out.append(tmp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[by,left_on]+value_cols+[f"{right_on}_asof"])

bps_asof = asof_right_by_id(
    left_df = px_eom[[COL_ID, "month_end"]],
    right_df= available[[COL_ID, "available_from", "bps_val"]],
    left_on = "month_end",
    right_on= "available_from",
    by      = COL_ID,
    value_cols=["bps_val"]
).rename(columns={"bps_val":"bps_val_asof", "available_from_asof":"available_from_asof"})

# ---------- 3) BM at EOM + membership-only buckets (NBINS=3/4) ----------
bm_eom = (px_eom.merge(bps_asof, on=[COL_ID, "month_end"], how="left")
                .assign(BM=lambda d: d["bps_val_asof"] / d["quoteclose"]))

def safe_ntiles(x, n):
    x = x.replace([np.inf, -np.inf], np.nan)
    uniq = x.dropna().nunique()
    if uniq < 2: return pd.Series([np.nan]*len(x), index=x.index)
    bins = min(n, uniq)
    return pd.qcut(x, bins, labels=False, duplicates="drop")

mask_mem = bm_eom["is_member"] == True

bm_eom["size_q"] = np.nan
bm_eom.loc[mask_mem, "size_q"] = (
    bm_eom.loc[mask_mem]
          .groupby("month_end")[COL_MKTCAP]
          .transform(lambda s: safe_ntiles(np.log(s.replace(0, np.nan)), NBINS))
)

bm_eom["bm_q"] = np.nan
bm_eom.loc[mask_mem, "bm_q"] = (
    bm_eom.loc[mask_mem]
          .groupby("month_end")["BM"]
          .transform(lambda s: safe_ntiles(s, NBINS))
)

bm_eom["cell"] = bm_eom["size_q"].astype("Int64").astype(str) + "_" + bm_eom["bm_q"].astype("Int64").astype(str)

# ---------- 4) Map prev EOM → daily, carry cell & membership ----------
df["month_period"] = df[COL_DATE].dt.to_period("M")
df["prev_period"]  = df["month_period"] - 1
df = df.merge(eom_map.rename(columns={"month_period":"prev_period", "month_end":"prev_eom"}),
              on=[COL_ID, "prev_period"], how="left", validate="m:1")
cell_map = bm_eom[[COL_ID, "month_end", "cell", "size_q", "is_member"]].rename(
    columns={"month_end":"prev_eom", "is_member":"is_member_prev"}
)
df = df.merge(cell_map, on=[COL_ID, "prev_eom"], how="left", validate="m:1")

# ---------- 5) MEMBER-ONLY peers (Size×BM), exclude self, LIGHT TRIM, size-only fallback ----------
# size-only buckets already in bm_eom; map size_q like cell_map (done)
# Cell peers (members-only)
mask_cell = df["is_member_prev"].fillna(False) & df["cell"].notna()

# ---------- 1) CELL peers (members only), trimmed, exclude self ----------
mask_cell = df_daily["is_member_prev"].fillna(False) & df_daily["cell"].notna()

tmp = df_daily.loc[mask_cell, [COL_DATE, "cell", "ret_d"]].copy()
g   = tmp.groupby([COL_DATE, "cell"], observed=True, sort=False)

# bandes de trim par (date, cell)
q_lo = g["ret_d"].transform(lambda s: s.quantile(TRIM_LO))
q_hi = g["ret_d"].transform(lambda s: s.quantile(TRIM_HI))
in_band = tmp["ret_d"].between(q_lo, q_hi, inclusive="both")

# moyenne tronquée & effectif tronqué
tmp["ret_trim"] = tmp["ret_d"].where(in_band)
mu  = g["ret_trim"].transform("mean")
n   = g["ret_trim"].transform("count")

# exclude self (jackknife) seulement si le titre est dans l'échantillon tronqué
peer_cell_trim_ex = np.where(
    in_band & (n > 1),
    (n * mu - tmp["ret_d"]) / (n - 1),
    mu
)

# write-back sans merge
df_daily.loc[mask_cell, "ret_peer_cell_trim_ex"] = peer_cell_trim_ex
df_daily.loc[mask_cell, "n_cell_trim"]           = n.values

# ---------- 2) SIZE-only fallback (members only), trimmed, exclude self ----------
mask_size = df_daily["is_member_prev"].fillna(False) & df_daily["size_q"].notna()

tmp2 = df_daily.loc[mask_size, [COL_DATE, "size_q", "ret_d"]].copy()
g2   = tmp2.groupby([COL_DATE, "size_q"], observed=True, sort=False)

q_lo2 = g2["ret_d"].transform(lambda s: s.quantile(TRIM_LO))
q_hi2 = g2["ret_d"].transform(lambda s: s.quantile(TRIM_HI))
in_band2 = tmp2["ret_d"].between(q_lo2, q_hi2, inclusive="both")

tmp2["ret_trim"] = tmp2["ret_d"].where(in_band2)
mu2 = g2["ret_trim"].transform("mean")
n2  = g2["ret_trim"].transform("count")

peer_size_trim_ex = np.where(
    in_band2 & (n2 > 1),
    (n2 * mu2 - tmp2["ret_d"]) / (n2 - 1),
    mu2
)

df_daily.loc[mask_size, "ret_peer_size_trim_ex"] = peer_size_trim_ex
df_daily.loc[mask_size, "n_size_trim"]           = n2.values

# ---------- 3) Benchmark final ----------
df_daily["ret_peer_bench"] = np.where(
    df_daily["n_cell_trim"].fillna(0) >= MIN_CELL,
    df_daily["ret_peer_cell_trim_ex"],
    df_daily["ret_peer_size_trim_ex"]
)

# ---------- 6) EVENTS → EAR & CFE (tradable J+2) ----------
import numpy as np
import pandas as pd

# ==== 1) Préparations (tri, ranks, IBES as-of) ====
df_daily = df_daily.sort_values([COL_ID, COL_DATE]).copy()

# rang de jour de bourse par titre (servira pour J+2)
if "td_rank" not in df_daily.columns:
    df_daily["td_rank"] = df_daily.groupby(COL_ID).cumcount()

# consensus IBES as-of : dernière valeur connue à chaque date
df_daily["ibes_ffill"] = (
    df_daily.groupby(COL_ID)[COL_IBES_MEAN]
            .ffill()      # dernière valeur disponible antérieure (≤ t)
)

# ==== 2) Fonction: mappe rpt_date -> J (1er jour de bourse strictement après) ====
def map_next_trading_day(events_df, strict_after=True):
    side = "right" if strict_after else "left"
    out = []
    for sid, ev in events_df.groupby(COL_ID, sort=False):
        g   = ev.sort_values("rpt_date").copy()
        cal = df_daily.loc[df_daily[COL_ID]==sid, [COL_DATE, "td_rank"]]
        if cal.empty:
            continue
        dates = cal[COL_DATE].values
        ranks = cal["td_rank"].values
        idx   = np.searchsorted(dates, g["rpt_date"].values, side=side)
        ok    = idx < len(dates)
        gg    = g.loc[ok].copy()
        gg["J"]       = pd.to_datetime(dates[idx[ok]])
        gg["J_rank"]  = ranks[idx[ok]]
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[COL_ID,"rpt_date","J","J_rank"])

# évènements résultats (une ligne par (id, rpt_date))
events_raw = (
    df_daily[[COL_ID, COL_EPS_RPT, COL_EPS_ACT]]
    .dropna(subset=[COL_EPS_RPT])
    .drop_duplicates(subset=[COL_ID, COL_EPS_RPT])
    .rename(columns={COL_EPS_RPT: "rpt_date"})
)
events_raw["rpt_date"] = pd.to_datetime(events_raw["rpt_date"])

# J = prochain jour de bourse STRICTEMENT après rpt_date (post-close conservateur)
events = map_next_trading_day(events_raw, strict_after=True)

# rattache l’EPS annoncé observé à J (si besoin)
events = events.merge(
    df_daily[[COL_ID, COL_DATE, COL_EPS_ACT]].rename(columns={COL_DATE:"J"}),
    on=[COL_ID, "J"], how="left"
)

# ==== 3) Récupère retours du titre et des pairs à J-1, J, J+1 ====
# on prépare les lags/leads une seule fois
for c in ["ret_d", "ret_peer_bench", COL_PRICE, "ibes_ffill"]:
    df_daily[f"{c}_lag1"]  = df_daily.groupby(COL_ID)[c].shift(1)
    df_daily[f"{c}_lead1"] = df_daily.groupby(COL_ID)[c].shift(-1)

evt_join = (
    df_daily.rename(columns={COL_DATE:"J"})[[COL_ID,"J",
        "ret_d_lag1","ret_d","ret_d_lead1",
        "ret_peer_bench_lag1","ret_peer_bench","ret_peer_bench_lead1",
        f"{COL_PRICE}_lag1", "ibes_ffill_lag1",
        "td_rank"  # rang du jour J (utile pour ready_day)
    ]]
    .rename(columns={"td_rank":"J_rank_chk"})
)

events = events.merge(evt_join, on=[COL_ID,"J"], how="left")

# ==== 4) EAR (produit arithmétique) ====
# garde uniquement les annonces où on a bien les 3 jours pour titre ET peers
mask_ok = (
    events[["ret_d_lag1","ret_d","ret_d_lead1",
            "ret_peer_bench_lag1","ret_peer_bench","ret_peer_bench_lead1"]]
    .notna().all(axis=1)
)
ev_ok = events.loc[mask_ok].copy()

R3 = (1 + ev_ok["ret_d_lag1"]) * (1 + ev_ok["ret_d"]) * (1 + ev_ok["ret_d_lead1"])
B3 = (1 + ev_ok["ret_peer_bench_lag1"]) * (1 + ev_ok["ret_peer_bench"]) * (1 + ev_ok["ret_peer_bench_lead1"])
ev_ok["EAR"] = R3 - B3

# ==== 5) CFE (consensus error, as-of J-1, scalarisé par le prix J-1) ====
ev_ok["price_Jm1"]     = ev_ok[f"{COL_PRICE}_lag1"]
ev_ok["ibes_mean_Jm1"] = ev_ok["ibes_ffill_lag1"]      # dernier consensus ≤ J-1
ev_ok["CFE"] = (ev_ok[COL_EPS_ACT] - ev_ok["ibes_mean_Jm1"]) / ev_ok["price_Jm1"]

# ==== 6) ready_day = J + 2 jours de bourse (tradable date) ====
ev_ok["ready_rank"] = ev_ok["J_rank"] + 2   # J_rank vient de map_next_trading_day

# map rank -> date via df_daily
rank_map = df_daily[[COL_ID, COL_DATE, "td_rank"]].rename(columns={COL_DATE:"ready_day"})
ev_ok = ev_ok.merge(rank_map, left_on=[COL_ID,"ready_rank"], right_on=[COL_ID,"td_rank"], how="left") \
             .drop(columns=["td_rank"])

# ==== 7) Résultat final des évènements ====
events_final = ev_ok[[COL_ID, "rpt_date", "J", "ready_day", "EAR", "CFE"]].sort_values(["J", COL_ID]).reset_index(drop=True)

# ---------- 7) MONTHLY SIGNALS (carry H=30/60/90, no decay) ----------
# trading-day rank at month_end
rank_V = df[[COL_ID, COL_DATE, "td_rank"]].merge(
    px_eom.rename(columns={"month_end":COL_DATE})[[COL_ID, COL_DATE]], on=[COL_ID, COL_DATE], how="right"
).rename(columns={COL_DATE:"month_end", "td_rank":"rank_V"}).sort_values([COL_ID,"month_end"])

# per id arrays for ready_day rank & EAR/CFE
evt_ready = events.dropna(subset=["ready_day"])[[COL_ID,"ready_day","EAR","CFE"]].copy()
evt_ready = evt_ready.merge(df[[COL_ID, COL_DATE, "td_rank"]].rename(columns={COL_DATE:"ready_day"}),
                            on=[COL_ID,"ready_day"], how="left").rename(columns={"td_rank":"ready_rank"})
evt_ready = evt_ready.sort_values([COL_ID,"ready_rank"])

# function: as-of event per month_end with age filter
def build_signal(rankV_df, evt_df, val_col, H):
    out = []
    for sid, gv in rankV_df.groupby(COL_ID, sort=False):
        rr = evt_df.loc[evt_df[COL_ID]==sid, "ready_rank"].values
        vv = evt_df.loc[evt_df[COL_ID]==sid, val_col].values
        if len(rr)==0:
            tmp = gv[[COL_ID,"month_end"]].copy()
            tmp[f"sig_{val_col}_{H}"] = np.nan
            out.append(tmp); continue
        rv = gv["rank_V"].values
        idx = np.searchsorted(rr, rv, side="right") - 1
        m = idx >= 0
        sig = np.full(len(rv), np.nan, dtype=float)
        age = np.where(m, rv - rr[idx], np.inf)
        sig[m & (age < H)] = vv[idx[m]]
        tmp = gv[[COL_ID,"month_end"]].copy()
        tmp[f"sig_{val_col}_{H}"] = sig
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

signals = rank_V.copy()
for H in (30,60,90):
    signals = signals.merge(build_signal(rank_V, evt_ready, "EAR", H), on=[COL_ID,"month_end"], how="left")
    signals = signals.merge(build_signal(rank_V, evt_ready, "CFE", H), on=[COL_ID,"month_end"], how="left")

signals_monthly = (signals[[COL_ID,"month_end",
                            "sig_EAR_30","sig_EAR_60","sig_EAR_90",
                            "sig_CFE_30","sig_CFE_60","sig_CFE_90"]]
                   .sort_values(["month_end", COL_ID])
                   .reset_index(drop=True))

# ---- OUTPUT ----
# signals_monthly: one row per (sedolcd, month_end) with 6 signal columns (NaN if no active event)

 










 
 
# ========= 7) SIGNAUX MENSUELS (H = 30/60/90) =========
# rank au month_end
me_rank = df_daily[[COL_ID, COL_DATE, "td_rank"]].merge(
    eom.rename(columns={"month_end": COL_DATE}), on=[COL_ID, COL_DATE], how="right"
).rename(columns={COL_DATE: "month_end", "td_rank": "rank_V"})

# dernier évènement avec ready_day <= V (asof)
evt_ready = events.dropna(subset=["ready_day"])[[COL_ID, "ready_day", "EAR", "CFE"]].copy()
sig = pd.merge_asof(
    me_rank.sort_values([COL_ID, "month_end"]),
    evt_ready.sort_values([COL_ID, "ready_day"]),
    left_on="month_end",
    right_on="ready_day",
    by=COL_ID,
    direction="backward",
)

# âge en jours de bourse = rank_V - rank(ready_day)
rdy_rank = rank_map.rename(columns={"d": "ready_day", "td_rank": "ready_rank"})
sig = sig.merge(rdy_rank, on=[COL_ID, "ready_day"], how="left")
sig["age_bdays"] = sig["rank_V"] - sig["ready_rank"]

def carry(x, H):
    return np.where((x["age_bdays"] >= 0) & (x["age_bdays"] < H), x, np.nan)

tmp = sig.copy()
ear30 = carry(tmp[["EAR","age_bdays"]].copy(), 30)["EAR"]
ear60 = carry(tmp[["EAR","age_bdays"]].copy(), 60)["EAR"]
ear90 = carry(tmp[["EAR","age_bdays"]].copy(), 90)["EAR"]
cfe30 = carry(tmp[["CFE","age_bdays"]].copy(), 30)["CFE"]
cfe60 = carry(tmp[["CFE","age_bdays"]].copy(), 60)["CFE"]
cfe90 = carry(tmp[["CFE","age_bdays"]].copy(), 90)["CFE"]

signals_monthly = (
    sig[[COL_ID, "month_end"]]
    .assign(
        sig_EAR_30 = ear30,
        sig_EAR_60 = ear60,
        sig_EAR_90 = ear90,
        sig_CFE_30 = cfe30,
        sig_CFE_60 = cfe60,
        sig_CFE_90 = cfe90,
    )
    .sort_values([ "month_end", COL_ID ])
    .reset_index(drop=True)
)

# ========= SORTIE =========
# signals_monthly: (month_end, sedolcd) + 6 colonnes de signaux
# Tu peux merger sur ton panel mensuel existant via (sedolcd, month_end).


# S&P 500 Forecasting Using Macro-Financial Variables

## Overview
This project leverages **machine learning** techniques to forecast the **weekly log returns** of the **S&P 500** using a dataset of macro-financial variables spanning over 20 years as part of **AI for Alpha**'s data challenge. The project was implemented using an **Object-Oriented Programming (OOP)** approach for better modularity and maintainability.

## Features
- **Exploratory Data Analysis:** ..
- **Data Preprocessing:** Handles missing values, outliers, and feature scaling.
- **Feature Selection:** Uses PCA, correlation analysis with bonferonni correction and permutation importance.
- **Modeling:** Implements a forecaster including time series split, ....
- **Hyperparameter Tuning:** Uses randomized search for optimal model performance.
- **Evaluation Metrics:** Custom scoring function combining RMSE and Directional Accuracy (DA).

## Installation
To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure
```plaintext
📂 Macro-SP500-Forecast
│── 📂 data                    # Raw datasets (S&P Daily Close Price and Macro-Financial Features)
│── 📂 src                     # Source code
│   │── processor.py           # Preprocessing module
│   │── EDA.ipynb              # Exploratory Data Analysis
│   │── skew_transformer.py    # Skew Transforer module
│   │── forecaster.py          # Machine learning models
│   │── arimaforecaster.py     # Autoregressive models
│   │── main.ipynb             # Main script to run the forecasting
│── prpoject_report            # Written Project Report
│── README.md                  # Documentation
```

## License
MIT License.

---
### Author
Solal Danan
