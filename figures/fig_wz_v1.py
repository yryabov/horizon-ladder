"""
Figure: the measured equation-of-state history (Sec. 4.4) -- v1
===============================================================
w(z) of the leading Avrami form: the Table-3 best fit with the posterior
68% band (background re-solved on the same 320-sample subset as the MCMC
driver), against the sn4 CPL tangent.  The band is drawn over the window
where every subsample history has activated (z <= 2.6); the best-fit curve
continues through its phantom-side activation transient (clipped at w = -3,
as in the battery).
Inputs: mcmc_avrami_sn4_samples_v1.npy.  Output: fig_wz_posterior.png + .json
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

def wz_of(wb, wc, mnu, lam, zc, hT, n_it=170, clip=-3.0):
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
    return np.maximum(w, clip)

ZB = np.linspace(-0.95, 2.6, 260)        # band window (future: all born)
ZF = np.linspace(-0.95, 3.2, 420)        # best-fit window (below activation)
xb, xf = np.log(1 / (1 + ZB)), np.log(1 / (1 + ZF))

N_SUB = 320
idx = rng.choice(len(S), N_SUB, replace=False)
band = np.empty((N_SUB, len(ZB)))
for j, i in enumerate(idx):
    w = wz_of(c['wb'][i], c['wc'][i], c['mnu'][i], c['lam'][i], c['zc'][i], c['h'][i])
    band[j] = np.interp(xb, XG, w)
q16, q50, q84 = np.percentile(band, [16, 50, 84], axis=0)

# best fit (Table 3) and the sn4 CPL tangent
BEST = dict(wb=0.0223960224, wc=0.1200090090, mnu=0.0590788467,
            hT=0.6790651632, lam=0.9026216626, zc=4.1125308865)
WFULL = wz_of(**BEST)
wbest = np.interp(xf, XG, WFULL)
ZE = np.linspace(0.0, 4.6, 460)                 # inset window (birth at 4.11)
we = np.interp(np.log(1 / (1 + ZE)), XG, WFULL)
me = (ZE < BEST['zc']) & (we > -2.995)          # w exists only below z_c
wcpl_e = None  # filled below
W0C, WAC = -0.8725307554, -0.4881683839
wcpl = W0C + WAC * (1 - 1 / (1 + ZF))
ZE5 = np.linspace(0.0, 5.0, 300)
wcpl_e = W0C + WAC * (1 - 1 / (1 + ZE5))

print(f"w0 band at z=0: {q50[0]:.3f} [{q16[0]:.3f}, {q84[0]:.3f}]")
print(f"best-fit w(0) = {wbest[0]:.3f}; min of best-fit transient: {wbest.min():.2f}")

BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 11, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=220)

ax.axvspan(-1.0, 0, color="#f0f0f0", zorder=0)
ax.axvline(0.0, color=INK, lw=0.8, ls=":")
ax.annotate("today", xy=(-0.055, -1.525), fontsize=11, color=INK,
            rotation=90, va="bottom", ha="center")
ax.annotate("future", xy=(-0.5, -1.45), fontsize=11, color=INK, ha="center", va="center")
ax.fill_between(ZB, q16, q84, color=BLUE, alpha=0.22, lw=0, label="posterior 68% band (Avrami)")
ax.plot(ZB, q50, color=BLUE, lw=1.2, ls=":", label="posterior median")
ax.plot(ZF, wbest, color=INK, lw=2.0, label="Avrami best fit (Table 3)")
ax.plot(ZF, wcpl, color=VERM, lw=1.6, ls="--", label="CPL tangent (Table 3)")
ax.axhline(-1.0, color=INK, lw=0.9, ls=":")
ax.annotate(r"$w = -1$: de Sitter", xy=(2.28, -0.99), fontsize=11, color=INK, va="bottom", ha="right")

ix = int(np.argmax(wbest < -1.0))
zx = float(np.interp(-1.0, [wbest[ix], wbest[ix-1]], [ZF[ix], ZF[ix-1]]))
i16 = int(np.argmax(q16 < -1.0)); z16 = float(np.interp(-1.0, [q16[i16], q16[i16-1]], [ZB[i16], ZB[i16-1]]))
i84 = int(np.argmax(q84 < -1.0)); z84 = float(np.interp(-1.0, [q84[i84], q84[i84-1]], [ZB[i84], ZB[i84-1]]))
zlo, zhi = min(z16, z84), max(z16, z84)
zx_cpl = float(1.0 / (1.0 - (-1.0 - W0C) / WAC) - 1.0)
ax.annotate(rf"Avrami crossing, $z = {zx:.2f}^{{+{zhi-zx:.2f}}}_{{-{zx-zlo:.2f}}}$",
            xy=(zx, -1.0), xytext=(0.58, -0.875), va="bottom", ha="left", fontsize=11, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=1))
ax.annotate(rf"CPL crossing, $z = {zx_cpl:.2f}$",
            xy=(zx_cpl, -1.0), xytext=(0.02, -1.21), va="top", ha="left", fontsize=11, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=0, relpos=(0.38, 1.0)))
ax.axvline(2.33, color=INK, lw=0.9, ls=":")
ax.annotate(r"Lyman-$\alpha$, $z=2.33$", xy=(2.32, -1.095), fontsize=11, color=INK, ha="right")
ax.annotate("apparent phantom\n" r"en route to $z_c = 4.11^{+1.50}_{-0.50}$",
            xy=(1.95, float(np.interp(1.95, ZF, wbest))), xytext=(0.95, -2.0),
            fontsize=11, color=INK, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=2,
                            relpos=(1.0, 0.75)))
w0b = float(np.interp(0.0, ZF, wbest))
w0_up = float(np.interp(0.0, ZB, q84)) - w0b
w0_dn = w0b - float(np.interp(0.0, ZB, q16))
print(f"w0 anchored at best fit: {w0b:.3f} +{w0_up:.3f} -{w0_dn:.3f}")
ax.annotate(rf"$w_0 = {w0b:.2f}^{{+{w0_up:.2f}}}_{{-{w0_dn:.2f}}}$",
            xy=(0.0, w0b), xytext=(-0.88, -1.12),
            fontsize=11, color=INK, ha="left", va="top",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=1, relpos=(1.0, 1.0)))

ax.set_xlim(-1.0, 2.7)
ax.set_ylim(-2.05, -0.75)
ax.set_xlabel(r"redshift $z$")
ax.set_ylabel(r"$w(z)$")
_h, _l = ax.get_legend_handles_labels()
_ord = [_l.index(n) for n in ["Avrami best fit (Table 3)", "CPL tangent (Table 3)",
                              "posterior median", "posterior 68% band (Avrami)"]]
ax.legend([_h[i] for i in _ord], [_l[i] for i in _ord], frameon=False,
          loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=11)
ax.grid(color="#dddddd", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_wz_posterior.png", bbox_inches="tight")
json.dump(dict(z_band=list(map(float, ZB)),
               w_q16=list(map(float, q16)), w_q50=list(map(float, q50)),
               w_q84=list(map(float, q84)),
               z_best=list(map(float, ZF)), w_best=list(map(float, wbest)),
               w_cpl=list(map(float, wcpl))),
          open("/mnt/user-data/outputs/fig_wz_posterior_data_v1.json", "w"), indent=1)
print("saved fig_wz_posterior.png + fig_wz_posterior_data_v1.json")
