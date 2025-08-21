import pandas as pd
import numpy as np

# Assure les dtypes
df_daily[COL_DATE] = pd.to_datetime(df_daily[COL_DATE])
df_daily[COL_BPS_RPT] = pd.to_datetime(df_daily[COL_BPS_RPT])

# --- 1) calendrier de bourse par titre ---
cal = (
    df_daily[[COL_ID, COL_DATE]]
    .drop_duplicates()
    .sort_values([COL_ID, COL_DATE])
)

# --- 2) observations BPS par (titre, rpt_date) ---
fundas_bps = (
    df_daily[[COL_ID, COL_BPS, COL_BPS_RPT]]
    .dropna(subset=[COL_BPS, COL_BPS_RPT])
    .drop_duplicates(subset=[COL_ID, COL_BPS_RPT])
    .rename(columns={COL_BPS: "bps_val", COL_BPS_RPT: "rpt_date_bps"})
)

# --- 3) available_from = 1er jour de bourse STRICTEMENT après rpt_date ---
parts = []
for sid, g in fundas_bps.groupby(COL_ID, sort=False):
    gg = g.sort_values("rpt_date_bps")
    cg = cal.loc[cal[COL_ID]==sid, [COL_DATE]].sort_values(COL_DATE)
    out = pd.merge_asof(
        gg, cg,
        left_on="rpt_date_bps",
        right_on=COL_DATE,
        direction="forward",
        allow_exact_matches=False   # <-- STRICTEMENT après (post-close)
    )
    out = out.rename(columns={COL_DATE: "available_from"})
    parts.append(out[[COL_ID, "rpt_date_bps", "bps_val", "available_from"]])
available = pd.concat(parts, ignore_index=True).dropna(subset=["available_from"])

# --- 4) EOM par titre ---
eom = (
    df_daily
    .groupby([COL_ID, pd.Grouper(key=COL_DATE, freq="ME")], as_index=False)
    .agg(month_end=(COL_DATE, "max"))
    .sort_values([COL_ID, "month_end"])
)

# --- 5) BPS as-of EOM (dernière valeur avec available_from ≤ month_end) ---
bps_asof = pd.merge_asof(
    eom, available.sort_values([COL_ID, "available_from"]),
    left_on="month_end", right_on="available_from",
    by=COL_ID, direction="backward"
)[[COL_ID, "month_end", "bps_val"]]

import pandas as pd
import numpy as np

# ========= CONFIG COLS (modifie ici si tes noms diffèrent) =========
COL_DATE      = "date"
COL_ID        = "sedolcd"
COL_CCY       = "instrmtccy"
COL_PRICE     = "quoteclose"
COL_MKTCAP    = "marketvalue"
COL_TR        = "totalreturn"      # daily TR (simple return, ex: 0.004). Fallbacks en dessous.
COL_PR_RET    = "closereturn"      # fallback si totalreturn indisponible (simple return)
COL_BPS       = "wsf_bps_ltm"      # sinon wsf_bps_qtr
COL_BPS_RPT   = "wsf_bps_rpt_date_qtr"  # si absent, fallback -> wsf_eps_rpt_date_qtr
COL_EPS_RPT   = "wsf_eps_rpt_date_qtr"
COL_EPS_ACT   = "wsf_eps_qtr"      # EPS annoncé à la rpt date
COL_IBES_MEAN = "ibes_eps_mean_qtr"

# ========= INPUT =========
# df_daily: DataFrame quotidien avec au moins les colonnes ci-dessus
# (Si ton DF est déjà chargé sous un autre nom, renomme pour suivre)
df_daily = df_daily.copy()

# ========= 0) FILTER USD =========
df_daily[COL_DATE] = pd.to_datetime(df_daily[COL_DATE])
df_daily = df_daily[df_daily[COL_CCY] == "USD"].copy()
df_daily.sort_values([COL_ID, COL_DATE], inplace=True)

# ========= 1) RETOURS QUOTIDIENS =========
# On crée ret_d: priorité à 'totalreturn', sinon 'closereturn', sinon %chg du prix.
if COL_TR in df_daily.columns and df_daily[COL_TR].notna().any():
    df_daily["ret_d"] = df_daily[COL_TR]
elif COL_PR_RET in df_daily.columns and df_daily[COL_PR_RET].notna().any():
    df_daily["ret_d"] = df_daily[COL_PR_RET]
else:
    df_daily["ret_d"] = df_daily.groupby(COL_ID)[COL_PRICE].pct_change()

# ========= 2) CALENDRIER MENSUEL (EOM par titre) =========
eom = (
    df_daily[[COL_ID, COL_DATE]]
    .groupby([COL_ID, pd.Grouper(key=COL_DATE, freq="M")])  # M = month-end calendar
    .max()
    .rename(columns={COL_DATE: "month_end"})
    .reset_index()
)
# Price & mktcap EOM
px_eom = df_daily[[COL_ID, COL_DATE, COL_PRICE, COL_MKTCAP]].merge(
    eom, left_on=[COL_ID, COL_DATE], right_on=[COL_ID, "month_end"], how="right"
)
px_eom = px_eom[[COL_ID, "month_end", COL_PRICE, COL_MKTCAP]]

# ========= 3) "available_from" pour BPS (as-of merge) =========
# Fallbacks colonnes BPS / RPT si manquantes
if COL_BPS not in df_daily.columns:
    COL_BPS = "wsf_bps_qtr"
if COL_BPS_RPT not in df_daily.columns or COL_BPS_RPT not in df_daily.columns:
    if COL_EPS_RPT in df_daily.columns:
        COL_BPS_RPT = COL_EPS_RPT
    else:
        raise ValueError("Aucune colonne de rpt_date pour BPS trouvée.")

fundas_bps = (
    df_daily[[COL_ID, COL_DATE, COL_BPS, COL_BPS_RPT]]
    .dropna(subset=[COL_BPS, COL_BPS_RPT])
    .rename(columns={COL_BPS_RPT: "rpt_date_bps", COL_BPS: "bps_val"})
    .copy()
)
fundas_bps["rpt_date_bps"] = pd.to_datetime(fundas_bps["rpt_date_bps"])
# available_from = prochain jour de bourse >= rpt_date_bps (asof forward sur calendrier du titre)
# On fait un merge_asof forward avec le calendrier des dates négociées par titre
cal = df_daily[[COL_ID, COL_DATE]].drop_duplicates().copy()
fundas_bps = fundas_bps.merge(cal, on=COL_ID, how="left")
fundas_bps = (
    pd.merge_asof(
        fundas_bps.sort_values([COL_ID, "rpt_date_bps"]),
        cal.sort_values([COL_ID, COL_DATE]),
        left_on="rpt_date_bps",
        right_on=COL_DATE,
        by=COL_ID,
        direction="forward",
        allow_exact_matches=True,
    )
    .rename(columns={COL_DATE: "available_from"})
    [[COL_ID, "available_from", "bps_val"]]
    .dropna(subset=["available_from"])
)
# BPS as-of à EOM
bps_asof = (
    pd.merge_asof(
        eom.sort_values([COL_ID, "month_end"]),
        fundas_bps.sort_values([COL_ID, "available_from"]),
        left_on="month_end",
        right_on="available_from",
        by=COL_ID,
        direction="backward",
    )
    [[COL_ID, "month_end", "bps_val"]]
)
# BM EOM
bm_eom = bps_asof.merge(px_eom, on=[COL_ID, "month_end"], how="left")
bm_eom["BM"] = bm_eom["bps_val"] / bm_eom[COL_PRICE]

# ========= 4) Buckets size×BM (5×5) au mois M, cellule utilisée le mois suivant =========
bm_eom["size_q"] = (
    bm_eom.groupby("month_end")[COL_MKTCAP]
    .transform(lambda s: pd.qcut(np.log(s.replace(0, np.nan)), 5, labels=False, duplicates="drop"))
)
bm_eom["bm_q"] = (
    bm_eom.groupby("month_end")["BM"]
    .transform(lambda s: pd.qcut(s.replace([np.inf, -np.inf], np.nan), 5, labels=False, duplicates="drop"))
)
bm_eom["cell"] = bm_eom["size_q"].astype("Int64").astype(str) + "_" + bm_eom["bm_q"].astype("Int64").astype(str)

# Map cellule du mois M -> pour les jours du mois (M+1)
# Pour chaque jour, on veut la cellule du EOM précédent
df_daily["prev_eom"] = (df_daily[COL_DATE].dt.to_period("M") - 1).dt.to_timestamp("M")
cell_map = bm_eom[[COL_ID, "month_end", "cell"]].rename(columns={"month_end": "prev_eom"})
df_daily = df_daily.merge(cell_map, on=[COL_ID, "prev_eom"], how="left")

# ========= 5) ret_peer_d (EW par cellule, incluant le titre) =========
peer = (
    df_daily[[COL_DATE, "cell", "ret_d"]]
    .dropna(subset=["cell"])
    .groupby([COL_DATE, "cell"], observed=True)["ret_d"]
    .mean()
    .rename("ret_peer_d")
    .reset_index()
)
df_daily = df_daily.merge(peer, on=[COL_DATE, "cell"], how="left")

# ========= 6) ÉVÉNEMENTS: J (rpt) -> EAR et CFE =========
events = (
    df_daily[[COL_ID, COL_EPS_RPT, COL_EPS_ACT]]
    .dropna(subset=[COL_EPS_RPT])
    .drop_duplicates(subset=[COL_ID, COL_EPS_RPT])
    .rename(columns={COL_EPS_RPT: "rpt_date"})
)
events["rpt_date"] = pd.to_datetime(events["rpt_date"])

# J = prochain jour de bourse >= rpt_date (par titre)
events = events.merge(cal, on=COL_ID, how="left")
events = pd.merge_asof(
    events.sort_values([COL_ID, "rpt_date"]),
    cal.sort_values([COL_ID, COL_DATE]),
    left_on="rpt_date",
    right_on=COL_DATE,
    by=COL_ID,
    direction="forward",
    allow_exact_matches=True,
).rename(columns={COL_DATE: "J"})[[COL_ID, "J"]].dropna()

# Ajoute EPS annoncé à J
eps_at_J = df_daily[[COL_ID, COL_DATE, COL_EPS_ACT]].rename(columns={COL_DATE: "J"})
events = events.merge(eps_at_J, on=[COL_ID, "J"], how="left")

# Retours titre & pairs aux jours J-1,J,J+1
for c in ["ret_d", "ret_peer_d", COL_PRICE, COL_IBES_MEAN]:
    df_daily[f"{c}_lead1"] = df_daily.groupby(COL_ID)[c].shift(-1)
    df_daily[f"{c}_lag1"]  = df_daily.groupby(COL_ID)[c].shift(1)

evt_join = df_daily[[COL_ID, COL_DATE,
                     "ret_d_lag1","ret_d","ret_d_lead1",
                     "ret_peer_d_lag1","ret_peer_d","ret_peer_d_lead1",
                     f"{COL_PRICE}_lag1", f"{COL_IBES_MEAN}_lag1"
                    ]].rename(columns={COL_DATE: "J"})

events = events.merge(evt_join, on=[COL_ID, "J"], how="left")

# EAR (produit arithmétique vs pairs)
R3  = (1+events["ret_d_lag1"]) * (1+events["ret_d"]) * (1+events["ret_d_lead1"])
B3  = (1+events["ret_peer_d_lag1"]) * (1+events["ret_peer_d"]) * (1+events["ret_peer_d_lead1"])
events["EAR"] = R3 - B3

# CFE as-of J-1 (price-scaled)
events["price_Jm1"] = events[f"{COL_PRICE}_lag1"]
events["ibes_mean_Jm1"] = events[f"{COL_IBES_MEAN}_lag1"]
events["CFE"] = (events[COL_EPS_ACT] - events["ibes_mean_Jm1"]) / events["price_Jm1"]

# ready_day = J+2 jours de bourse
# On numérote les jours par titre pour calculer "J+2"
df_daily["td_rank"] = df_daily.groupby(COL_ID).cumcount()
rank_map = df_daily[[COL_ID, COL_DATE, "td_rank"]].rename(columns={COL_DATE: "d"})
events = events.merge(rank_map.rename(columns={"d": "J"}), on=[COL_ID, "J"], how="left")
events["ready_rank"] = events["td_rank"] + 2
ready_map = rank_map.rename(columns={"d": "ready_day"})
events = events.merge(ready_map, left_on=[COL_ID, "ready_rank"], right_on=[COL_ID, "td_rank"], how="left")
events = events.rename(columns={"ready_day": "ready_day"}).drop(columns=["td_rank_x","td_rank_y"])

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
