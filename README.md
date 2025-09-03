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

# --- PREP ---
df_daily = df_daily.sort_values([COL_ID, COL_DATE]).copy()
df_daily[COL_EPS_RPT] = pd.to_datetime(df_daily[COL_EPS_RPT], errors="coerce")

# rang (servira pour J+2)
if "td_rank" not in df_daily.columns:
    df_daily["td_rank"] = df_daily.groupby(COL_ID).cumcount()

# calendrier par titre (unique dates + rang)
cal = (df_daily[[COL_ID, COL_DATE, "td_rank"]]
       .drop_duplicates()
       .sort_values([COL_ID, COL_DATE])
       .reset_index(drop=True))

# consensus as-of: dernière valeur ≤ t puis on prendra lag1 pour J-1
df_daily["ibes_ffill"] = df_daily.groupby(COL_ID)[COL_IBES_MEAN].ffill()

# Lags/leads utiles autour de J
for c in ["ret_d", "ret_peer_bench", COL_PRICE, "ibes_ffill"]:
    df_daily[f"{c}_lag1"]  = df_daily.groupby(COL_ID)[c].shift(1)
    df_daily[f"{c}_lead1"] = df_daily.groupby(COL_ID)[c].shift(-1)

# --- 1) EVENTS RAW (une ligne par (id, rpt_date)) ---
events_raw = (df_daily[[COL_ID, COL_EPS_RPT, COL_EPS_ACT]]
              .dropna(subset=[COL_EPS_RPT])
              .drop_duplicates(subset=[COL_ID, COL_EPS_RPT])
              .rename(columns={COL_EPS_RPT: "rpt_date"})
              .sort_values([COL_ID, "rpt_date"])
              .reset_index(drop=True))

# --- 2) map rpt_date -> J (prochain jour de bourse STRICTEMENT après), EN BOUCLE PAR TITRE ---
parts = []
for sid, g in events_raw.groupby(COL_ID, sort=False):
    g  = g.sort_values("rpt_date")
    cg = cal.loc[cal[COL_ID] == sid, [COL_DATE, "td_rank"]].sort_values(COL_DATE)
    if cg.empty:
        continue
    out = pd.merge_asof(
        left=g,
        right=cg,
        left_on="rpt_date",
        right_on=COL_DATE,
        direction="forward",
        allow_exact_matches=False  # STRICTEMENT après (post-close)
    )
    out = out.rename(columns={COL_DATE: "J", "td_rank": "J_rank"})
    parts.append(out[[COL_ID, "rpt_date", "J", "J_rank", COL_EPS_ACT]])

events = (pd.concat(parts, ignore_index=True)
          .dropna(subset=["J"])
          .reset_index(drop=True))

# --- 3) Joindre les retours titre/peer et les inputs CFE à J-1, J, J+1 ---
join_cols = [COL_ID, COL_DATE,
             "ret_d_lag1","ret_d","ret_d_lead1",
             "ret_peer_bench_lag1","ret_peer_bench","ret_peer_bench_lead1",
             f"{COL_PRICE}_lag1","ibes_ffill_lag1"]
evt_join = (df_daily.rename(columns={COL_DATE:"J"})[[c if c==COL_ID else c.replace(COL_DATE,"J") for c in [COL_ID, COL_DATE]] + 
            [c for c in join_cols if c not in [COL_ID, COL_DATE]]])

events = events.merge(evt_join, on=[COL_ID, "J"], how="left")

# --- 4) EAR sur [J-1, J, J+1] ---
mask = events[[
    "ret_d_lag1","ret_d","ret_d_lead1",
    "ret_peer_bench_lag1","ret_peer_bench","ret_peer_bench_lead1"
]].notna().all(axis=1)

R3 = (1 + events.loc[mask, "ret_d_lag1"]) * (1 + events.loc[mask, "ret_d"]) * (1 + events.loc[mask, "ret_d_lead1"])
B3 = (1 + events.loc[mask, "ret_peer_bench_lag1"]) * (1 + events.loc[mask, "ret_peer_bench"]) * (1 + events.loc[mask, "ret_peer_bench_lead1"])
events.loc[mask, "EAR"] = R3 - B3

# --- 5) CFE (consensus error) avec consensus ≤ J-1, scalé par prix J-1 ---
events.loc[mask, "CFE"] = (
    (events.loc[mask, COL_EPS_ACT] - events.loc[mask, "ibes_ffill_lag1"]) /
    events.loc[mask, f"{COL_PRICE}_lag1"]
)

# --- 6) ready_day = J + 2 jours de bourse (via les rangs) ---
events["ready_rank"] = events["J_rank"] + 2
ready_map = cal.rename(columns={COL_DATE: "ready_day"})[[COL_ID, "td_rank", "ready_day"]]
events = (events.merge(ready_map, left_on=[COL_ID, "ready_rank"], right_on=[COL_ID, "td_rank"], how="left")
                .drop(columns=["td_rank"]))

# --- 7) Résultat propre ---
events_final = (events[[COL_ID, "rpt_date", "J", "ready_day", "EAR", "CFE"]]
                .sort_values(["J", COL_ID])
                .reset_index(drop=True))

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
import numpy as np
import pandas as pd

# --- 1) month-end trading-day rank per stock (rank_V) ---
rank_V = (
    px_eom[[COL_ID, "month_end"]]
    .merge(
        df_daily[[COL_ID, COL_DATE, "td_rank"]]
            .rename(columns={COL_DATE: "month_end"}),
        on=[COL_ID, "month_end"],
        how="left",
        validate="m:1",
    )
    .rename(columns={"td_rank": "rank_V"})
    .sort_values([COL_ID, "month_end"])
    .reset_index(drop=True)
)

# --- 2) ensure events have ready_rank (map date -> rank) ---
evt_ready = (
    events_final.dropna(subset=["ready_day"])[[COL_ID, "ready_day", "EAR", "CFE"]]
    .merge(
        df_daily[[COL_ID, COL_DATE, "td_rank"]].rename(columns={COL_DATE: "ready_day"}),
        on=[COL_ID, "ready_day"],
        how="left",
        validate="m:1",
    )
    .rename(columns={"td_rank": "ready_rank"})
    .sort_values([COL_ID, "ready_rank"])
    .reset_index(drop=True)
)

# --- 3) function: last event active at V with age < H (trading days) ---
def carry_signal(rankV_df, evt_df, val_col, H):
    out = []
    for sid, gv in rankV_df.groupby(COL_ID, sort=False):
        rv = gv["rank_V"].to_numpy()
        e  = evt_df[evt_df[COL_ID] == sid]
        rr = e["ready_rank"].to_numpy()
        vv = e[val_col].to_numpy()
        if rr.size == 0:
            tmp = gv[[COL_ID, "month_end"]].copy()
            tmp[f"sig_{val_col}_{H}"] = np.nan
            out.append(tmp); continue
        # index of most-recent ready_rank <= rank_V
        idx = np.searchsorted(rr, rv, side="right") - 1
        m = idx >= 0
        age = np.full(rv.shape, np.inf)
        age[m] = rv[m] - rr[idx[m]]
        sig = np.full(rv.shape, np.nan, dtype=float)
        sel = m & (age < H)
        sig[sel] = vv[idx[sel]]
        tmp = gv[[COL_ID, "month_end"]].copy()
        tmp[f"sig_{val_col}_{H}"] = sig
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

# --- 4) build all signals (H = 30, 60, 90) ---
signals = rank_V[[COL_ID, "month_end"]].copy()
for H in (30, 60, 90):
    signals = signals.merge(carry_signal(rank_V, evt_ready, "EAR", H),
                            on=[COL_ID, "month_end"], how="left")
    signals = signals.merge(carry_signal(rank_V, evt_ready, "CFE", H),
                            on=[COL_ID, "month_end"], how="left")

# --- 5) final monthly panel (ready to merge with your monthly dataset) ---
signals_monthly = (signals
                   .sort_values(["month_end", COL_ID])
                   .reset_index(drop=True))
# columns: [sedolcd, month_end, sig_EAR_30, sig_EAR_60, sig_EAR_90, sig_CFE_30, sig_CFE_60, sig_CFE_90]



















import numpy as np
import pandas as pd

# --- Alias colonnes ---
ID, DATE = COL_ID, COL_DATE
PRICE    = COL_PRICE            # ex: "quoteclose"
EPS_Q    = "wsf_eps_qtr"        # EPS trimestriel ACTUEL (annoncé au jour J)
H_LIST   = (30, 60, 90)

# ---------- 0) Préparations ----------
df = df_daily.sort_values([ID, DATE]).copy()
if "td_rank" not in df.columns:
    df["td_rank"] = df.groupby(ID).cumcount()

# calendrier par titre
cal = df[[ID, DATE, "td_rank"]].drop_duplicates().sort_values([ID, DATE])

# EPS réel @J (si pas déjà dans events_final)
eps_at_J = df[[ID, DATE, EPS_Q]].rename(columns={DATE:"J"}).drop_duplicates([ID, "J"])

# Base évènements: J & ready_day (on n'altère pas EAR)
events = events_final[[ID, "J", "ready_day"]].copy()
if EPS_Q not in events.columns:
    events = events.merge(eps_at_J, on=[ID, "J"], how="left", validate="m:1")

# ---------- 1) Prix J-1 pour le scaling ----------
# map rang(J) -> rang(J)-1 -> date -> prix
events = (events
          .merge(cal.rename(columns={DATE:"J"}), on=[ID, "J"], how="left", validate="m:1")
          .assign(Jm1_rank=lambda d: d["td_rank"] - 1)
          .merge(cal.rename(columns={DATE:"Jm1"})[[ID, "td_rank", "Jm1"]],
                 left_on=[ID, "Jm1_rank"], right_on=[ID, "td_rank"], how="left")
          .merge(df.rename(columns={DATE:"Jm1"})[[ID, "Jm1", PRICE]],
                 on=[ID, "Jm1"], how="left")
          .rename(columns={PRICE: "price_Jm1"})
          .drop(columns=["td_rank_x","td_rank_y"], errors="ignore"))

# ---------- 2) prev_eom = dernier EOM STRICTEMENT avant J ----------
def map_prev_eom_for_events_asof(ev_df, eom_tab, id_col=COL_ID):
    out = []
    for sid, g in ev_df.groupby(id_col, sort=False):
        left  = g.sort_values("J")
        right = (eom_tab[eom_tab[id_col]==sid][[id_col, "month_end"]]
                 .sort_values("month_end")
                 .rename(columns={"month_end":"prev_eom"}))
        if right.empty:
            continue
        tmp = pd.merge_asof(
            left, right,
            left_on="J", right_on="prev_eom",
            direction="backward",
            allow_exact_matches=False
        )
        out.append(tmp)
    return pd.concat(out, ignore_index=True) if out else ev_df.assign(prev_eom=np.nan)

events = map_prev_eom_for_events_asof(events, px_eom[[COL_ID, "month_end"]])

 
# ---------- 3) Joindre ta prévision ML as-of prev_eom & calculer CFE_ML_Q ----------
# eps_ml_qtr: [ID, "month_end", "eps_ml_qtr"]  (month_end = dernier jour de bourse)
events = events.merge(
    eps_ml_qtr.rename(columns={"month_end":"prev_eom"}),
    on=[ID, "prev_eom"], how="left", validate="m:1"
)

# CFE_ML_Q = (EPS_Q @J - EPS_ML(as-of prev_eom)) / Price_{J-1}
events["CFE_ML_Q"] = (events[EPS_Q] - events["eps_ml_qtr"]) / events["price_Jm1"]

# (option) winsor pour stabilité
lo, hi = events["CFE_ML_Q"].quantile([0.01, 0.99])
events["CFE_ML_Q"] = events["CFE_ML_Q"].clip(lo, hi)

# ---------- 4) Porter en mensuel (H=30/60/90) ----------
# ready_rank pour chaque évènement
events = (events
          .merge(cal.rename(columns={DATE:"ready_day"})[[ID, "ready_day", "td_rank"]],
                 on=[ID, "ready_day"], how="left", validate="m:1")
          .rename(columns={"td_rank":"ready_rank"}))

# rang du month_end par titre
rank_V = (px_eom[[ID, "month_end"]]
          .merge(cal.rename(columns={DATE:"month_end"}), on=[ID, "month_end"], how="left", validate="m:1")
          .rename(columns={"td_rank":"rank_V"})
          .sort_values([ID, "month_end"])
          .reset_index(drop=True))

def carry_signal(rankV_df, evt_df, val_col, H):
    out=[]
    for sid, gv in rankV_df.groupby(ID, sort=False):
        e = evt_df.loc[evt_df[ID]==sid, ["ready_rank", val_col]].dropna(subset=["ready_rank"])
        if e.empty:
            tmp = gv[[ID,"month_end"]].copy(); tmp[f"sig_{val_col}_{H}"] = np.nan; out.append(tmp); continue
        rr = e["ready_rank"].to_numpy()
        vv = e[val_col].to_numpy()
        rv = gv["rank_V"].to_numpy()
        idx = np.searchsorted(rr, rv, side="right") - 1  # dernier event actif avant V
        m = idx >= 0
        age = np.where(m, rv - rr[idx], np.inf)
        sig = np.full(rv.shape, np.nan)
        sig[m & (age < H)] = vv[idx[m]]
        tmp = gv[[ID,"month_end"]].copy(); tmp[f"sig_{val_col}_{H}"] = sig
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

# Signaux CFE_ML_Q
evt_cfe = events.dropna(subset=["CFE_ML_Q", "ready_rank"])[[ID, "ready_rank", "CFE_ML_Q"]]
signals_cfe_ml = rank_V[[ID, "month_end"]].copy()
for H in H_LIST:
    signals_cfe_ml = signals_cfe_ml.merge(
        carry_signal(rank_V, evt_cfe, "CFE_ML_Q", H),
        on=[ID, "month_end"], how="left"
    )

# ---------- 5) Merge avec tes signaux EAR mensuels existants ----------
# Suppose que tu as déjà un DF 'signals_ear_monthly' avec [ID, "month_end", sig_EAR_30/60/90]
# Sinon, garde juste signals_cfe_ml.
signals_monthly = signals_cfe_ml.copy()
if "signals_ear_monthly" in globals():
    signals_monthly = signals_ear_monthly.merge(
        signals_cfe_ml, on=[ID, "month_end"], how="left", validate="1:1"
    )

# -> 'signals_monthly' = grille mensuelle (id, month_end) avec:
#    - sig_CFE_ML_Q_30 / _60 / _90
#    - (+ tes sig_EAR_30 / _60 / _90 si fournis)









import numpy as np
import pandas as pd

# ========= PARAMS =========
ID, DATE = COL_ID, COL_DATE              # tes constantes
D_MAX     = 90                           # horizon max pour la courbe de drift (jours de bourse)
TRAIN_END = pd.Timestamp("2018-12-31")   # fin de fenêtre d'apprentissage (à adapter)
CALIB_METHOD = "reg"                     # "reg" ou "quintile"
NBINS_EAR    = 5                         # nb de bins si method = "quintile"
IMPOSE_MONOTONE = True                   # cummax sur beta(d) (optionnel, méthode "reg")

# ========= 0) Abnormal daily =========
df = (df_daily
      .sort_values([ID, DATE])
      .copy())
df["abn_d"] = df["ret_d"] - df["ret_peer_bench"]   # r - benchmark peers

# calendrier / cumsum rapide par titre
cal = (df[[ID, DATE, "td_rank", "abn_d"]]
       .drop_duplicates()
       .sort_values([ID, DATE])
       .reset_index(drop=True))

csum_dict = {}
for sid, g in cal.groupby(ID, sort=False):
    ranks = g["td_rank"].to_numpy()
    abn   = g["abn_d"].to_numpy()
    csum  = np.concatenate(([0.0], np.cumsum(abn)))  # csum[k] = sum abn[0..k-1]
    csum_dict[sid] = (ranks, csum)

# ========= 1) Evénements d'apprentissage =========
train_evt = (events_final
             .dropna(subset=["ready_day","EAR"])
             .loc[events_final["ready_day"] <= TRAIN_END, [ID, "ready_day", "EAR"]]
             .merge(cal.rename(columns={DATE:"ready_day"}), on=[ID, "ready_day"], how="left", validate="m:1")
             .rename(columns={"td_rank":"ready_rank"})
             .dropna(subset=["ready_rank"])
             .sort_values([ID, "ready_rank"])
             .reset_index(drop=True))

# ========= 2) Calibration =========
def estimate_beta_reg(train_df):
    """beta(d) via régression cross-sectionnelle sans intercept, d=1..D_MAX."""
    num = np.zeros(D_MAX, dtype=float)
    den = 0.0
    for sid, g in train_df.groupby(ID, sort=False):
        rr = g["ready_rank"].to_numpy()
        ss = g["EAR"].to_numpy()
        ranks, csum = csum_dict.get(sid, (None, None))
        if ranks is None: 
            continue
        pos = np.searchsorted(ranks, rr, side="left")
        for j in range(len(rr)):
            p0 = pos[j]
            max_d = min(D_MAX, len(ranks) - p0 - 1)
            if max_d <= 0: 
                continue
            car = csum[p0+1:p0+max_d+1] - csum[p0]   # CAR(d) = somme des abn
            num[:max_d] += ss[j] * car
            den         += ss[j] * ss[j]
    beta = np.divide(num, den, out=np.zeros_like(num), where=den>0)  # beta[d-1] pour d=1..D_MAX
    if IMPOSE_MONOTONE:
        beta = np.maximum.accumulate(beta)  # rend beta(d) non décroissant
    return pd.Series(beta, index=np.arange(1, D_MAX+1), name="beta_d")

def estimate_curves_quintile(train_df, nbins=5):
    """
    Méthode 'quintile' :
      - coupe EAR en nbins (global, sur la période TRAIN)
      - pour chaque bin et d, calcule mean CAR(d)
    Renvoie:
      - car_mean: DataFrame index=d (1..D_MAX), colonnes=bin labels (0..nbins-1)
      - bin_edges: Series des bords (longueur nbins+1)
    """
    # binnings globaux sur EAR (winsor possible ici si besoin)
    ear = train_df["EAR"].replace([np.inf, -np.inf], np.nan).dropna()
    q = np.unique(np.quantile(ear, np.linspace(0, 1, nbins+1)))
    # sécurité si trop peu d'uniques
    if len(q) <= 2:
        q = np.array([ear.min(), ear.median(), ear.max()])
    # affecte chaque event à un bin (labels 0..B-1)
    def bin_id(x):
        # np.digitize -> bins à droite fermés
        return np.clip(np.digitize(x, q[1:-1], right=True), 0, len(q)-2)
    train_df = train_df.copy()
    train_df["bin"] = bin_id(train_df["EAR"].values).astype(int)

    # accumulateurs
    sums = {b: np.zeros(D_MAX, dtype=float) for b in range(len(q)-1)}
    cnts = {b: np.zeros(D_MAX, dtype=float) for b in range(len(q)-1)}

    for sid, g in train_df.groupby(ID, sort=False):
        rr  = g["ready_rank"].to_numpy()
        bb  = g["bin"].to_numpy()
        ranks, csum = csum_dict.get(sid, (None, None))
        if ranks is None: 
            continue
        pos = np.searchsorted(ranks, rr, side="left")
        for j in range(len(rr)):
            p0 = pos[j]
            max_d = min(D_MAX, len(ranks)-p0-1)
            if max_d <= 0:
                continue
            car = csum[p0+1:p0+max_d+1] - csum[p0]
            b = int(bb[j])
            sums[b][:max_d] += car
            cnts[b][:max_d] += 1

    # moyennes par bin & d
    car_mean = pd.DataFrame(
        {b: np.divide(sums[b], cnts[b], out=np.zeros_like(sums[b]), where=cnts[b]>0) 
         for b in sums},
        index=np.arange(1, D_MAX+1)
    )
    car_mean.index.name = "d"
    bin_edges = pd.Series(q, name="edges")
    return car_mean, bin_edges

# lance calibration
if CALIB_METHOD == "reg":
    beta_curve = estimate_beta_reg(train_evt)   # Series index d
    car_q = None; bin_edges = None
else:
    car_q, bin_edges = estimate_curves_quintile(train_evt, nbins=NBINS_EAR)
    beta_curve = None

# ========= 3) Ranks EOM V et prochain EOM W =========
px_eom = px_eom.sort_values([ID, "month_end"]).copy()
px_eom["next_month_end"] = px_eom.groupby(ID)["month_end"].shift(-1)

rankV = (px_eom
         .merge(cal.rename(columns={DATE:"month_end"}), on=[ID, "month_end"], how="left", validate="m:1")
         .rename(columns={"td_rank":"rank_V"}))
rankV = (rankV
         .merge(cal.rename(columns={DATE:"next_month_end", "td_rank":"rank_W"})[[ID, "next_month_end", "rank_W"]],
                on=[ID, "next_month_end"], how="left"))

# événements avec ready_rank (pour toute la période)
evt_all = (events_final
           .dropna(subset=["ready_day","EAR"])
           .merge(cal.rename(columns={DATE:"ready_day"}), on=[ID,"ready_day"], how="left", validate="m:1")
           .rename(columns={"td_rank":"ready_rank"})
           .sort_values([ID,"ready_rank"])
           .reset_index(drop=True))

# ========= 4) Construction du signal decay-adjusted à EOM =========
def signal_decay_for_id(gv, ev):
    """
    gv: sous-DF rankV pour un id (contient month_end, rank_V, rank_W)
    ev: événements de cet id (contient ready_rank, EAR)
    Utilise:
      - beta_curve (méthode "reg")  OU
      - car_q & bin_edges (méthode "quintile")
    """
    sid = gv[ID].iloc[0]
    ranks, csum = csum_dict.get(sid, (None, None))
    out = gv[[ID, "month_end"]].copy()
    sig = np.full(len(gv), np.nan, dtype=float)
    if (ranks is None) or ev.empty:
        out["sig_EAR_decay"] = sig
        return out

    # dernière annonce avant V
    rr  = ev["ready_rank"].to_numpy()
    ear = ev["EAR"].to_numpy()
    pos_t = np.searchsorted(ranks, rr, side="left")  # index t dans csum

    rv = gv["rank_V"].to_numpy()
    rw = gv["rank_W"].to_numpy()
    idx = np.searchsorted(rr, rv, side="right") - 1
    m = idx >= 0
    if not m.any():
        out["sig_EAR_decay"] = sig
        return out

    i   = idx[m]           # indices d'évènement actif
    p0  = pos_t[i]         # index t
    ear_i = ear[i]

    # jours totaux t->W, cap à [1, D_MAX]
    d_tot = (rw[m] - rr[i]).astype(int)
    d_tot = np.clip(d_tot, 1, D_MAX)

    # expected total drift t->W
    if CALIB_METHOD == "reg":
        exp_tot = beta_curve.reindex(d_tot).to_numpy() * ear_i
    else:
        # bin du EAR_i selon edges d'apprentissage
        # np.digitize renvoie 0..B-1
        b = np.clip(np.digitize(ear_i, bin_edges.values[1:-1], right=True), 0, len(bin_edges)-2)
        # valeur moyenne attendue au d_tot de CE bin
        # car_q index d, colonnes bins
        exp_tot = car_q.iloc[d_tot - 1, b].to_numpy()

    # realized drift t->V (somme d'anormaux) : csum[pV] - csum[p0]
    pV = np.searchsorted(ranks, rv[m], side="left")
    realized = csum[pV] - csum[p0]

    sig[m] = exp_tot - realized
    out["sig_EAR_decay"] = sig
    return out

parts = []
for sid, gv in rankV.groupby(ID, sort=False):
    ev = evt_all[evt_all[ID]==sid]
    parts.append(signal_decay_for_id(gv, ev))

sig_decay_monthly = (pd.concat(parts, ignore_index=True)
                     .sort_values(["month_end", ID])
                     .reset_index(drop=True))
# -> colonnes: [ID, month_end, sig_EAR_decay]



























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
