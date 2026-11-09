"""
Figure: MCMC posterior of the leading Avrami form (Sec. 4.4) -- v1
==================================================================
Panel (a): the (z_c, lambda) posterior -- the date--rate ridge (sample
correlation +0.95), with 68/95% KDE contours, the posterior median, and the
profile best fit of Table 3.
Panel (b): the ridge projections -- (w_0, Omega_de(2.33)) recomputed with
the self-consistent solver on a posterior subsample (as in the released
MCMC driver), with 68/95% contours and the profile ridge corridors overlaid
(w_0 = -0.951..-0.962; Omega_de(2.33) = 2.6--4.3%).
Inputs: mcmc_avrami_sn4_samples_v1.npy (thinned chain; released).
Output: fig_mcmc_posterior.png + fig_mcmc_posterior_data_v1.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

rng = np.random.default_rng(42)
WR = 2.469e-5 * (1 + 0.2271 * 3.046)
COLS = ["wb", "wc", "lam", "zc", "h", "mnu", "As", "ns", "tau", "Ap"]
S = np.load("/mnt/user-data/outputs/mcmc_avrami_sn4_samples_v1.npy")
c = {k: S[:, i] for i, k in enumerate(COLS)}

# ---- solver (validated blocks) ------------------------------------------
def make_grid(xmax=13.0, n=2400):
    xg = np.linspace(np.log(1 / 2500), xmax, n)
    return xg, xg[1] - xg[0], np.exp(xg)

XG, DX, AA = make_grid()

def RE_of(H):
    f = 1 / (AA * H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1] + f[::-1][1:]) / 2 * DX)[::-1], [0]])
    return AA * (cum + 1 / (AA[-1] * H[-1]))

def derive(wb, wc, mnu, lam, zc, hT, n_it=170):
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
    w = -1 - np.gradient(np.log(np.maximum(r, 1e-30)), XG) / 3
    w0 = float(np.interp(0.0, XG, w))
    xq = np.log(1 / 3.33)
    Ode233 = float(np.interp(xq, XG, r / H ** 2))
    return w0, 100 * Ode233

N_SUB = 320
idx = rng.choice(len(S), N_SUB, replace=False)
w0s, odes = [], []
for i in idx:
    w0, od = derive(c['wb'][i], c['wc'][i], c['mnu'][i],
                    c['lam'][i], c['zc'][i], c['h'][i])
    w0s.append(w0); odes.append(od)
w0s, odes = np.array(w0s), np.array(odes)
q = lambda a: np.percentile(a, [16, 50, 84])
print("w0 posterior:", np.round(q(w0s), 4))
print("Ode233 [%]  :", np.round(q(odes), 2))

# ---- KDE contour helper ---------------------------------------------------
def kde_contours(x, y, gridn=140):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    kde.set_bandwidth(kde.factor * 1.45)   # smooth the thinned-chain tail
    dens = kde(xy)
    lev68, lev95 = np.percentile(dens, [32, 5])
    xg = np.linspace(x.min() - 0.08 * np.ptp(x), x.max() + 0.08 * np.ptp(x), gridn)
    yg = np.linspace(y.min() - 0.08 * np.ptp(y), y.max() + 0.08 * np.ptp(y), gridn)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return XX, YY, ZZ, [lev95, lev68]

BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 10.5, "axes.linewidth": 0.8})
fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=220)

# ---- panel (a): the ridge -------------------------------------------------
zc, lam = c['zc'], c['lam']
ax.scatter(zc, lam, s=3, color=BLUE, alpha=0.18, lw=0, rasterized=True)
XX, YY, ZZ, levs = kde_contours(zc, lam)
ax.contour(XX, YY, ZZ, levels=levs, colors=[INK, INK], linewidths=[0.9, 1.5],
           linestyles=["--", "-"])
ax.plot(np.median(zc), np.median(lam), "o", color=INK, ms=5,
        label="posterior median")
ax.plot(4.1125, 0.9026, "*", color=VERM, ms=12, mec="white", mew=0.5,
        label="best fit (Table 3)")
ax.annotate(f"sample correlation $+0.95$", xy=(0.04, 0.93),
            xycoords="axes fraction", fontsize=9.5, color=GREY)
ax.axvline(8.0, color=GREY, lw=0.8, ls=":")
ax.annotate("prior edge", xy=(7.93, 0.52), rotation=90, fontsize=9,
            color=GREY, ha="right")
ax.set_xlabel(r"activation epoch $z_c$")
ax.set_ylabel(r"growth width $\lambda$")
ax.legend(frameon=False, loc="lower right", fontsize=9.3)
ax.grid(color="#dddddd", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
ax.set_title(r"(a)  the date--rate ridge", fontsize=10.5, loc="left")

# ---- panel (b): the projections ------------------------------------------
axb.scatter(w0s, odes, s=6, color=BLUE, alpha=0.35, lw=0, rasterized=True)
XX, YY, ZZ, levs = kde_contours(w0s, odes)
axb.contour(XX, YY, ZZ, levels=levs, colors=[INK, INK], linewidths=[0.9, 1.5],
            linestyles=["--", "-"])
# profile ridge corridors
axb.add_patch(plt.Rectangle((-0.962, 2.6), 0.011, 1.7, fill=True,
              color=VERM, alpha=0.25, lw=0))
axb.annotate("profile ridge corridors", xy=(-0.9555, 4.35), xytext=(-0.935, 5.6),
             fontsize=9, color=VERM,
             arrowprops=dict(arrowstyle="-", color=VERM, lw=0.7))
axb.axvline(-1.0, color=GREY, lw=0.8, ls=":")
axb.annotate(r"$w = -1$", xy=(-0.9985, 5.95), fontsize=9, color=GREY)
axb.set_xlabel(r"$w_0$")
axb.set_ylabel(r"$\Omega_{\rm de}(2.33)$  [%]")
axb.grid(color="#dddddd", lw=0.5, alpha=0.6); axb.set_axisbelow(True)
axb.set_title(r"(b)  the ridge projections", fontsize=10.5, loc="left")

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_mcmc_posterior.png", bbox_inches="tight")
json.dump(dict(n_sub=N_SUB, idx=list(map(int, idx)),
               w0=list(map(float, w0s)), Ode233_pct=list(map(float, odes)),
               w0_q=[float(v) for v in q(w0s)],
               Ode233_q=[float(v) for v in q(odes)]),
          open("/mnt/user-data/outputs/fig_mcmc_posterior_data_v1.json", "w"), indent=1)
print("saved fig_mcmc_posterior.png + fig_mcmc_posterior_data_v1.json")
