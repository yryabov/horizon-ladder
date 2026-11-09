"""
Figure: the measured density history (companion to the w(z) figure) -- v1
=========================================================================
rho_de(z)/rho_de(0) of the leading Avrami form: Table-3 best fit, posterior
median and 68% band (same 320-sample subset and solver as the w(z) figure),
against the flat LCDM line and the CPL tangent.  Unlike w(z), the density
is defined everywhere -- rho = 0 before birth is a value, not a gap -- so
the band needs no activation window.  The future (z < 0) is shown to the
de Sitter plateau of the attractor.
Inputs: mcmc_avrami_sn4_samples_v1.npy.  Output: fig_rho_history.png + .json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
WR = 2.469e-5 * (1 + 0.2271 * 3.046)
COLS = ["wb", "wc", "lam", "zc", "h", "mnu", "As", "ns", "tau", "Ap"]
S = np.load("/mnt/user-data/outputs/mcmc_avrami_sn4_samples_v1.npy")
c = {k: S[:, i] for i, k in enumerate(COLS)}

def make_grid(xmax=13.0, n=2400):
    xg = np.linspace(np.log(1 / 2500), xmax, n)
    return xg, xg[1] - xg[0], np.exp(xg)

XG, DX, AA = make_grid()

def RE_of(H):
    f = 1 / (AA * H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1] + f[::-1][1:]) / 2 * DX)[::-1], [0]])
    return AA * (cum + 1 / (AA[-1] * H[-1]))

def rho_of(wb, wc, mnu, lam, zc, hT, n_it=170):
    wm = wb + wc + mnu / 93.14
    xc = np.log(1 / (1 + zc))
    al = np.where(XG > xc, 1 - np.exp(-((XG - xc) / lam) ** 2), 0.0)
    lo, hi = 0.02, 2.4
    for _ in range(40):
        mid = (lo + hi) / 2
        r = np.full_like(XG, mid)
        for _ in range(n_it):
            H = np.sqrt(wm * AA ** -3 + WR * AA ** -4 + np.maximum(r, 0))
            r = 0.58 * r + 0.42 * al / RE_of(H) ** 2
        h = float(np.interp(1.0, AA, H))
        if abs(h - hT) < 4e-4:
            break
        lo, hi = (mid, hi) if h < hT else (lo, mid)
    return r  # omega_de(x) on XG, in (100 km/s/Mpc)^2 units

ZR = np.concatenate([np.linspace(-0.995, -0.9, 60), np.linspace(-0.9, 5.75, 640)])
xr = np.log(1 / (1 + ZR))

N_SUB = 320
idx = rng.choice(len(S), N_SUB, replace=False)
band = np.empty((N_SUB, len(ZR)))
for j, i in enumerate(idx):
    r = rho_of(c['wb'][i], c['wc'][i], c['mnu'][i], c['lam'][i], c['zc'][i], c['h'][i])
    band[j] = np.interp(xr, XG, r) / float(r[-1])
q16, q50, q84 = np.percentile(band, [16, 50, 84], axis=0)
hmax = band.max(axis=1)
h16, h84 = np.percentile(hmax, [16, 84])
i0 = int(np.argmin(np.abs(ZR - 0.0)))
t16, t50, t84 = np.percentile(band[:, i0], [16, 50, 84])
print(f"today (posterior): {t50:.4f} +{t84-t50:.4f} -{t50-t16:.4f} rho_dS")

BEST = dict(wb=0.0223960224, wc=0.1200090090, mnu=0.0590788467,
            hT=0.6790651632, lam=0.9026216626, zc=4.1125308865)
rb = rho_of(**BEST)
rbest = np.interp(xr, XG, rb) / float(rb[-1])
today = float(np.interp(0.0, XG, rb)) / float(rb[-1])
print(f"best fit: rho_de(today) = {today:.4f} rho_dS")

# CPL tangent, extrapolated over the full axis including the future:
# w(a) = w0 + wa(1-a) rises linearly in a, crosses -1/3, and the density
# collapses -- drawn to make visible that the tangent has no plateau of its
# own (an extrapolation far outside the tangent's Taylor validity).
W0C, WAC = -0.8725307554, -0.4881683839
ZC_ = ZR
aC = 1 / (1 + ZC_)
with np.errstate(over='ignore', under='ignore'):
    rcpl = today * aC ** (-3 * (1 + W0C + WAC)) * np.exp(-3 * WAC * (1 - aC))

BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 11.5, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=220)

ax.axvspan(-1.0, 0, color="#f0f0f0", zorder=0)
ax.fill_between(ZR, q16, q84, color=BLUE, alpha=0.22, lw=0,
                label="posterior 68% band (Avrami)")
ax.plot(ZR, q50, color=BLUE, lw=1.2, ls=":", label="posterior median")
ax.plot(ZR, rbest, color=INK, lw=2.0, label="Avrami best fit (Table 3)")
ax.plot(ZC_, rcpl, color=VERM, lw=1.6, ls="--",
        label="CPL tangent (Table 3)")
ax.axhline(1.0, color=GREY, lw=1.2, ls="-.",
           label=r"$\Lambda$CDM")
ax.annotate("de Sitter plateau of the attractor", xy=(2.62, 1.0 + 0.018),
            fontsize=10.3, color=INK)
ax.errorbar([0.0], [today], yerr=[[today - t16], [t84 - today]],
            fmt="o", color=INK, ms=5.5, capsize=3.5, lw=1.2, zorder=7)
imax = int(np.argmax(rbest))
rmax = rbest[imax]
print(f"hill height: best {rmax:.4f}; posterior 68% [{h16:.4f}, {h84:.4f}]")
ax.errorbar([ZR[imax]], [rmax], xerr=[[ZR[imax] - 0.37], [0.85 - ZR[imax]]],
            yerr=[[rmax - h16], [h84 - rmax]],
            fmt="o", mfc="white", mec=INK, ecolor=INK, mew=1.1, ms=6.5,
            capsize=3.5, lw=1.2, zorder=7)
ax.annotate(r"$w = -1$ crossing, $z = 0.58^{+0.27}_{-0.21}$,"
            rf"  $\rho_{{\rm de}}(0.58) = {rmax:.2f}^{{+{h84-rmax:.2f}}}_{{-{rmax-h16:.2f}}}\,\rho_{{\rm dS}}$",
            xy=(ZR[imax] + 0.04, rbest[imax] + 0.008), xytext=(0.35, 1.262),
            fontsize=10.3, color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=2))
ax.annotate(rf"$\rho_{{\rm de}}(0) = {today:.2f}^{{+{t84-today:.2f}}}_{{-{today-t16:.2f}}}\,\rho_{{\rm dS}}$",
            xy=(0.0, today), xytext=(0.20, 0.84), fontsize=10.3, color=INK,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=3))
ax.axvline(0, color=INK, lw=0.8, ls=":")
ax.annotate("today", xy=(-0.06, 0.56), rotation=90, fontsize=10.3, color=INK,
            ha="right", va="center")
ax.annotate("future", xy=(-0.5, 0.035), fontsize=10.3, color=INK, ha="center")
ax.annotate("de Sitter\n" r"$z \to -1$" "\n" r"$a \to \infty$",
            xy=(-0.93, 0.68), fontsize=10.3, color=INK, ha="left")
ax.axvline(BEST['zc'], color=INK, lw=0.8, ls=":")
ax.axvspan(3.61, 5.61, color="#f0f0f0", zorder=0)
ax.annotate(r"birth, $z_c = 4.11^{+1.50}_{-0.50}$", xy=(BEST['zc'] - 0.07, 0.30),
            rotation=90, fontsize=10.3, color=INK, ha="right", va="center")

ax.set_xlim(-1.0, 5.75)
ax.set_ylim(0, 1.32)
ax.set_xlabel(r"redshift $z$")
ax.set_ylabel(r"$\rho_{\rm de}(z)\,/\,\rho_{\rm dS}$")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.15, 0.02), fontsize=10.3)
ax.grid(color="#dddddd", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_rho_history.png", bbox_inches="tight")
json.dump(dict(z=list(map(float, ZR)),
               rho_best=list(map(float, rbest)),
               rho_q16=list(map(float, q16)), rho_q50=list(map(float, q50)),
               rho_q84=list(map(float, q84)),
               today_over_rho_dS=today,
               z_cpl=list(map(float, ZC_)), rho_cpl=list(map(float, rcpl))),
          open("/mnt/user-data/outputs/fig_rho_history_data_v1.json", "w"), indent=1)
print("saved fig_rho_history.png + fig_rho_history_data_v1.json")
