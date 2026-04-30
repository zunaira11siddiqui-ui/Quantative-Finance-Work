Modern Portfolio Theory Calculator
===================================
Diversified universe: Stocks, Bonds, REITs, Indexes, Commodities, Vol Proxy
Uses historically-calibrated synthetic return data (2019-2023 period).
Replace `generate_synthetic_prices()` with yf.download() in a live environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. ASSET UNIVERSE
# ─────────────────────────────────────────────
ASSETS = {
    # Ticker: (asset class, ann_ret, ann_vol)
    # Stocks
    "AAPL":  ("Stock",       0.28,  0.28),
    "MSFT":  ("Stock",       0.30,  0.26),
    "GOOGL": ("Stock",       0.22,  0.27),
    "AMZN":  ("Stock",       0.18,  0.30),
    "NVDA":  ("Stock",       0.55,  0.52),
    "JPM":   ("Stock",       0.14,  0.23),
    "JNJ":   ("Stock",       0.08,  0.14),
    # Bonds
    "TLT":   ("Bond",       -0.05,  0.14),
    "AGG":   ("Bond",        0.01,  0.06),
    "HYG":   ("Bond",        0.05,  0.10),
    "LQD":   ("Bond",        0.02,  0.09),
    # REITs
    "VNQ":   ("REIT",        0.10,  0.20),
    "SPG":   ("REIT",        0.08,  0.30),
    "O":     ("REIT",        0.07,  0.17),
    # Index ETFs
    "SPY":   ("Index",       0.15,  0.18),
    "QQQ":   ("Index",       0.20,  0.22),
    "EFA":   ("Index",       0.06,  0.17),
    "VWO":   ("Index",       0.04,  0.19),
    # Commodities
    "GLD":   ("Commodity",   0.09,  0.13),
    "USO":   ("Commodity",   0.02,  0.35),
    "PDBC":  ("Commodity",   0.06,  0.20),
    # Volatility Proxy
    "VIXY":  ("Vol/Options",-0.40,  0.80),
}

CLASS_COLORS = {
    "Stock":       "#4E9AF1",
    "Bond":        "#F5A623",
    "REIT":        "#7ED321",
    "Index":       "#9B59B6",
    "Commodity":   "#E74C3C",
    "Vol/Options": "#1ABC9C",
}

TICKERS          = list(ASSETS.keys())
RISK_FREE_ANNUAL = 0.04
NUM_PORTFOLIOS   = 60_000
DAYS             = 1259

# ─────────────────────────────────────────────
# 2. GENERATE CORRELATED SYNTHETIC RETURNS
# ─────────────────────────────────────────────
def build_corr_matrix():
    n = len(TICKERS)
    C = np.eye(n)
    idx = {t: i for i, t in enumerate(TICKERS)}

    stocks = ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","JNJ"]
    bonds  = ["TLT","AGG","HYG","LQD"]
    reits  = ["VNQ","SPG","O"]
    indexes= ["SPY","QQQ","EFA","VWO"]

    for g in [stocks, bonds, reits, indexes]:
        for a in g:
            for b in g:
                if a != b:
                    C[idx[a], idx[b]] = 0.65 if g == stocks else 0.55

    for s in stocks:
        for ix in indexes:
            C[idx[s], idx[ix]] = C[idx[ix], idx[s]] = 0.80
    for b in ["TLT","AGG","LQD"]:
        for s in stocks + indexes:
            C[idx[b], idx[s]] = C[idx[s], idx[b]] = -0.20
    for r in reits:
        for s in stocks + indexes:
            C[idx[r], idx[s]] = C[idx[s], idx[r]] = 0.55
    for t in TICKERS:
        if t != "GLD":
            c = -0.05 if ASSETS[t][0] in ("Stock","Index","REIT") else 0.20
            C[idx["GLD"], idx[t]] = C[idx[t], idx["GLD"]] = c
    for s in stocks + indexes:
        C[idx["VIXY"], idx[s]] = C[idx[s], idx["VIXY"]] = -0.70

    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-6, None)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    D = np.sqrt(np.diag(C))
    return C / np.outer(D, D)

C = build_corr_matrix()
daily_mu  = np.array([ASSETS[t][1] / 252 for t in TICKERS])
daily_sig = np.array([ASSETS[t][2] / np.sqrt(252) for t in TICKERS])
L = np.linalg.cholesky(C)
Z = np.random.randn(DAYS, len(TICKERS))
raw_ret = (Z @ L.T) * daily_sig + daily_mu
daily_ret_df = pd.DataFrame(raw_ret, columns=TICKERS)

mean_ret   = daily_ret_df.mean()
cov_matrix = daily_ret_df.cov()
n          = len(TICKERS)
print(f"✅ Asset universe: {n} assets across 6 classes\n")

# ─────────────────────────────────────────────
# 3. PORTFOLIO METRICS
# ─────────────────────────────────────────────
def portfolio_metrics(w):
    ret = np.dot(w, mean_ret) * 252
    vol = np.sqrt(w @ cov_matrix.values @ w) * np.sqrt(252)
    sr  = (ret - RISK_FREE_ANNUAL) / vol if vol > 0 else 0
    return ret, vol, sr

# ─────────────────────────────────────────────
# 4. MONTE CARLO
# ─────────────────────────────────────────────
print(f"🎲 Running {NUM_PORTFOLIOS:,} Monte Carlo simulations …")
mc_ret, mc_vol, mc_sr, mc_w = [], [], [], []
for _ in range(NUM_PORTFOLIOS):
    w = np.random.dirichlet(np.ones(n))
    r, v, s = portfolio_metrics(w)
    mc_ret.append(r); mc_vol.append(v); mc_sr.append(s); mc_w.append(w)
mc_ret = np.array(mc_ret)
mc_vol = np.array(mc_vol)
mc_sr  = np.array(mc_sr)

# ─────────────────────────────────────────────
# 5. SCIPY OPTIMIZATION
# ─────────────────────────────────────────────
print("⚙️  Running SciPy optimizations …")
bounds = tuple((0.0, 0.35) for _ in range(n))
cons   = [{"type":"eq","fun":lambda w: np.sum(w)-1}]
w0     = np.ones(n)/n
opts   = {"maxiter":3000,"ftol":1e-14}

w_ms = minimize(lambda w: -portfolio_metrics(w)[2], w0, method="SLSQP",
                bounds=bounds, constraints=cons, options=opts).x
r_ms, v_ms, s_ms = portfolio_metrics(w_ms)

w_mv = minimize(lambda w: portfolio_metrics(w)[1], w0, method="SLSQP",
                bounds=bounds, constraints=cons, options=opts).x
r_mv, v_mv, s_mv = portfolio_metrics(w_mv)

w_mr = minimize(lambda w: -portfolio_metrics(w)[0], w0, method="SLSQP",
                bounds=bounds, constraints=cons, options=opts).x
r_mr, v_mr, s_mr = portfolio_metrics(w_mr)

# ─────────────────────────────────────────────
# 6. EFFICIENT FRONTIER
# ─────────────────────────────────────────────
targets = np.linspace(r_mv*1.01, r_mr*0.98, 80)
ef_v, ef_r = [], []
for tgt in targets:
    c2 = cons + [{"type":"eq","fun":lambda w, t=tgt: portfolio_metrics(w)[0]-t}]
    res = minimize(lambda w: portfolio_metrics(w)[1], w0, method="SLSQP",
                   bounds=bounds, constraints=c2, options=opts)
    if res.success:
        ef_v.append(res.fun); ef_r.append(tgt)

cml_x = np.linspace(0, v_ms*2.2, 200)
cml_y = RISK_FREE_ANNUAL + s_ms*cml_x

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
print("📊 Generating charts …")
DARK, LIGHT = "#0D1117", "#E6EDF3"
fig = plt.figure(figsize=(22, 15), facecolor=DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
ax_main = fig.add_subplot(gs[0, :2])
ax_pie1 = fig.add_subplot(gs[0, 2])
ax_bar  = fig.add_subplot(gs[1, :2])
ax_pie2 = fig.add_subplot(gs[1, 2])

for ax in [ax_main, ax_pie1, ax_bar, ax_pie2]:
    ax.set_facecolor("#161B22")
    for sp in ax.spines.values():
        sp.set_color("#30363D")
    ax.tick_params(colors=LIGHT, labelsize=8)

sc = ax_main.scatter(mc_vol, mc_ret, c=mc_sr, cmap="plasma", s=1.5, alpha=0.45, zorder=1)
cbar = fig.colorbar(sc, ax=ax_main, pad=0.01)
cbar.set_label("Sharpe Ratio", color=LIGHT, fontsize=9)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LIGHT, fontsize=8)

if ef_v:
    ax_main.plot(ef_v, ef_r, color="#FFD700", lw=3, zorder=4, label="Efficient Frontier")
ax_main.plot(cml_x, cml_y, color="#FFFFFF", lw=1.5, ls="--", alpha=0.7, zorder=3,
             label="Capital Market Line")
ax_main.scatter(v_ms, r_ms, color="#FF4D4D", s=400, marker="*", zorder=6,
                label=f"Max Sharpe  SR={s_ms:.2f}  R={r_ms:.1%}  σ={v_ms:.1%}")
ax_main.scatter(v_mv, r_mv, color="#00FF88", s=400, marker="*", zorder=6,
                label=f"Min Volatility  SR={s_mv:.2f}  R={r_mv:.1%}  σ={v_mv:.1%}")
ax_main.scatter(v_mr, r_mr, color="#FF9900", s=400, marker="*", zorder=6,
                label=f"Max Return  SR={s_mr:.2f}  R={r_mr:.1%}  σ={v_mr:.1%}")
ax_main.scatter(0, RISK_FREE_ANNUAL, color="white", s=80, marker="D", zorder=5,
                label=f"Risk-Free  {RISK_FREE_ANNUAL:.0%}")
for cls, clr in CLASS_COLORS.items():
    ax_main.plot([], [], "o", color=clr, markersize=7, label=cls)

ax_main.set_xlabel("Annual Volatility (Risk)", color=LIGHT, fontsize=10)
ax_main.set_ylabel("Annual Expected Return",  color=LIGHT, fontsize=10)
ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax_main.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor=LIGHT,
               fontsize=7, ncol=2, loc="lower right")
ax_main.set_title("Efficient Frontier — Multi-Asset Universe  (Stocks · Bonds · REITs · Indexes · Commodities · Volatility)",
                  color=LIGHT, fontsize=10, fontweight="bold", pad=10)

def make_pie(ax, weights, title):
    mask = weights > 0.01
    wp   = weights[mask]
    labs = [TICKERS[i] for i in range(n) if mask[i]]
    clrs = [CLASS_COLORS[ASSETS[t][0]] for t in labs]
    _, _, autotexts = ax.pie(wp, labels=labs, colors=clrs, autopct="%1.1f%%",
                             pctdistance=0.78, startangle=90,
                             textprops={"color":LIGHT,"fontsize":7},
                             wedgeprops={"linewidth":0.5,"edgecolor":"#0D1117"})
    for at in autotexts: at.set_fontsize(6)
    ax.set_title(title, color=LIGHT, fontsize=10, fontweight="bold", pad=6)

make_pie(ax_pie1, w_ms, "⭐ Max Sharpe Portfolio")
make_pie(ax_pie2, w_mv, "🛡️  Min Volatility Portfolio")

portfolios = {
    "Max Sharpe":    (r_ms, v_ms, s_ms),
    "Min Volatility":(r_mv, v_mv, s_mv),
    "Max Return":    (r_mr, v_mr, s_mr),
    "Equal Weight":  portfolio_metrics(np.ones(n)/n),
}
x, width = np.arange(4), 0.22
for i,(lbl,(vals_color)) in enumerate([
    ("Return (%)",    ([v[0]*100 for v in portfolios.values()], "#4E9AF1")),
    ("Volatility (%)",([v[1]*100 for v in portfolios.values()], "#F5A623")),
    ("Sharpe Ratio",  ([v[2]     for v in portfolios.values()], "#7ED321")),
]):
    vals, color = vals_color
    bars = ax_bar.bar(x+(i-1)*width, vals, width, label=lbl, color=color,
                      alpha=0.85, edgecolor="#0D1117", linewidth=0.5)
    for bar,val in zip(bars,vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{val:.1f}", ha="center", va="bottom", color=LIGHT, fontsize=7)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(list(portfolios.keys()), color=LIGHT, fontsize=9)
ax_bar.tick_params(colors=LIGHT)
ax_bar.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor=LIGHT, fontsize=8)
ax_bar.set_title("Portfolio Comparison — Return / Volatility / Sharpe",
                 color=LIGHT, fontsize=10, fontweight="bold", pad=6)
ax_bar.set_ylabel("Value", color=LIGHT, fontsize=9)

fig.suptitle("Modern Portfolio Theory  ·  Multi-Asset Optimized Portfolio",
             color=LIGHT, fontsize=16, fontweight="bold", y=0.998)

plt.savefig("/mnt/user-data/outputs/efficient_frontier.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("✅ Chart saved.")

# ─────────────────────────────────────────────
# 8. PRINT & EXPORT
# ─────────────────────────────────────────────
r_eq, v_eq, s_eq = portfolio_metrics(np.ones(n)/n)

for name, w in [("MAX SHARPE", w_ms), ("MIN VOLATILITY", w_mv), ("MAX RETURN", w_mr)]:
    rows = [(TICKERS[i], ASSETS[TICKERS[i]][0], w[i])
            for i in np.argsort(-w) if w[i] > 0.005]
    df = pd.DataFrame(rows, columns=["Ticker","Class","Weight"])
    df["Weight %"] = df["Weight"].map(lambda x: f"{x:.1%}")
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(df[["Ticker","Class","Weight %"]].to_string(index=False))

print(f"""
╔══════════════════════════════════════════════════╗
║             OPTIMAL PORTFOLIO SUMMARY            ║
╠══════════════════════════════════════════════════╣
║  Strategy          Return   Volatility   Sharpe  ║
╠══════════════════════════════════════════════════╣
║  Max Sharpe       {r_ms:>7.2%}    {v_ms:>7.2%}     {s_ms:>5.2f}  ║
║  Min Volatility   {r_mv:>7.2%}    {v_mv:>7.2%}     {s_mv:>5.2f}  ║
║  Max Return       {r_mr:>7.2%}    {v_mr:>7.2%}     {s_mr:>5.2f}  ║
║  Equal Weight     {r_eq:>7.2%}    {v_eq:>7.2%}     {s_eq:>5.2f}  ║
╚══════════════════════════════════════════════════╝
""")

wdf = pd.DataFrame({
    "Ticker": TICKERS,
    "Class":  [ASSETS[t][0] for t in TICKERS],
    "Ann_Return": [ASSETS[t][1] for t in TICKERS],
    "Ann_Vol":    [ASSETS[t][2] for t in TICKERS],
    "W_MaxSharpe":   w_ms,
    "W_MinVol":      w_mv,
    "W_MaxReturn":   w_mr,
    "W_EqualWeight": [1/n]*n,
})
wdf.to_csv("/home/claude/portfolio_weights.csv", index=False)
print("✅ Weights exported.")
