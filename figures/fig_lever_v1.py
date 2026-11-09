"""
Figure: the Lyman-alpha lever (Sec. 4.2) -- v1
==============================================
Panel (a): Omega_de(z) histories of the three classes at their Table-3 best
fits (self-consistent solver for the horizon forms; analytic LCDM), against
the two readings of the z = 2.33 point:
  (i)  the single-row inversion of D_H/r_d = 8.600 +- 0.066 (joint
       AP+BAO row) with omega_m and r_d CMB-pinned -- excludes a vanishing
       tail on its own;
  (ii) the all-probe 68% corridor Omega_de(2.33) = 2.6--4.3% from the
       profile likelihood of the full battery (Sec. 5).
Panel (b): where the Avrami lead over LCDM comes from -- Delta chi2 by data
block on the three-probe battery (CMB and BAO+SN totals from the released
verdict files; the BAO/Lya/SN split recomputed with the same solver, summing
to the verdict total within 0.3 units).
Pure numpy + matplotlib; seconds to run.  Output: fig_lya_lever.png + .json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KM = 299792.458
WR = 2.469e-5 * (1 + 0.2271 * 3.046)

# ---- Table-3 best fits (verdict JSONs, released with the repository) ----
AVRAMI = dict(wb=0.0223960224, wc=0.1200090090, mnu=0.0590788467,
              h=0.6790651632, lam=0.9026216626, zc=4.1125308865)
THERMAL = dict(wb=0.0222639081, wc=0.1218333637, mnu=0.0586124892,
               h=0.6806963877, Tc_code=0.1361112799, p=4.0)
LCDM = dict(wb=0.0225181049, wc=0.1180185449, mnu=0.0581851433,
            h=0.6826523697)

# ---- joint Lya row (DR2 Results IV values, as embedded in the battery) ----
DH_RD, DH_RD_ERR = 8.600, 0.066
Z_LYA = 2.33
SIG_WM = 0.0011            # CMB-pinned omega_m uncertainty (MCMC wc width)


def wm_of(p):
    return p['wb'] + p['wc'] + p['mnu'] / 93.14


def rd_desi(wcb, wb):
    return 147.05 * (wcb / 0.1432) ** -0.23 * (wb / 0.02236) ** -0.13


# ---- self-consistent solver (validated blocks of the campaign) ----------
def make_grid(xmax=13.0, n=3200):
    xg = np.linspace(np.log(1 / 2500), xmax, n)
    return xg, xg[1] - xg[0], np.exp(xg)


def RE_of(H, AA, DX):
    f = 1 / (AA * H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1] + f[::-1][1:]) / 2 * DX)[::-1], [0]])
    return AA * (cum + 1 / (AA[-1] * H[-1]))


def solve(alpha_fn, wm, h_target, n_it=300, mix=0.42):
    XG, DX, AA = make_grid()
    lo, hi = 0.02, 2.4
    for _ in range(44):
        mid = (lo + hi) / 2
        r = np.full_like(XG, mid)
        for _ in range(n_it):
            H = np.sqrt(wm * AA ** -3 + WR * AA ** -4 + np.maximum(r, 0))
            RE = RE_of(H, AA, DX)
            al = alpha_fn(XG, H, RE)
            r = (1 - mix) * r + mix * (al / max(al[-1], 1e-12)) / RE ** 2
        h = float(np.interp(1.0, AA, H))
        if abs(h - h_target) < 3e-4:
            break
        lo, hi = (mid, hi) if h < h_target else (lo, mid)
    Ode = r / H ** 2
    return XG, Ode


def avrami_alpha(p):
    xc = np.log(1 / (1 + p['zc']))
    return lambda XG, H, RE: np.where(XG > xc, 1 - np.exp(-((XG - xc) / p['lam']) ** 2), 0.0)


def thermal_alpha(p):
    Rc = 1 / (2 * np.pi * p['Tc_code'])
    return lambda XG, H, RE: np.maximum(1 - (Rc / np.maximum(RE, 1e-30)) ** p['p'], 0.0)


# ---- curves -------------------------------------------------------------
zgrid = np.linspace(1.5, 5.5, 400)
xq = np.log(1 / (1 + zgrid))

XG, Ode = solve(avrami_alpha(AVRAMI), wm_of(AVRAMI), AVRAMI['h'])
ode_av = np.interp(xq, XG, Ode)

XG, Ode = solve(thermal_alpha(THERMAL), wm_of(THERMAL), THERMAL['h'])
ode_th = np.interp(xq, XG, Ode)

wm_l = wm_of(LCDM)
wde_l = LCDM['h'] ** 2 - wm_l - WR
ode_lc = wde_l / (wde_l + wm_l * (1 + zgrid) ** 3 + WR * (1 + zgrid) ** 4)

# ---- single-row inversion ----------------------------------------------
def row_inversion(dh_rd, wm, wb):
    rd = rd_desi(wm - AVRAMI['mnu'] / 93.14, wb)   # r_d uses omega_cb
    Hq = C_KM / (dh_rd * rd)
    Hm2 = 1e4 * (wm * (1 + Z_LYA) ** 3 + WR * (1 + Z_LYA) ** 4)
    return 1 - Hm2 / Hq ** 2

wm0, wb0 = wm_of(AVRAMI), AVRAMI['wb']
c0 = row_inversion(DH_RD, wm0, wb0)
s_dh = abs(row_inversion(DH_RD + DH_RD_ERR, wm0, wb0) - c0)
s_wm = abs(row_inversion(DH_RD, wm0 + SIG_WM, wb0) - c0)
sig = np.hypot(s_dh, s_wm)
print(f"single-row reading: Omega_de(2.33) = {100*c0:.2f} +- {100*sig:.2f} %"
      f"  (DH: {100*s_dh:.2f}, wm: {100*s_wm:.2f})")
i233 = np.argmin(np.abs(zgrid - Z_LYA))
print(f"curves at 2.33: Avrami {100*ode_av[i233]:.2f}%, LCDM {100*ode_lc[i233]:.2f}%, "
      f"Thermal {100*ode_th[i233]:.2f}%")

# ---- block decomposition of the battery-3 lead (Avrami - LCDM) ----------
# CMB from the verdict files; BAO/Lya/SN split recomputed with this solver
BLOCKS = [("CMB (TT+TE+EE)", -4.29),
          (r"BAO, 12 numbers w/o Ly$\alpha$", -4.17),
          ("SN (Pantheon+)", -1.88),
          (r"Ly$\alpha$ joint row", +2.23)]
TOTAL_RECOMP, TOTAL_VERDICT = -8.11, -8.41

# ---- figure -------------------------------------------------------------
BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 10.5, "axes.linewidth": 0.8,
})
fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=220,
                              gridspec_kw=dict(width_ratios=[1.45, 1.0]))

ax.plot(zgrid, 100 * ode_av, color=BLUE, lw=2.0,
        label=r"early birth — Avrami best fit ($z_c = 4.11$)")
ax.plot(zgrid, 100 * ode_lc, color=INK, lw=1.6, ls="--",
        label=r"$\Lambda$CDM best fit (rigid tail)")
ax.plot(zgrid, 100 * ode_th, color=VERM, lw=2.0,
        label=r"late birth — Thermal GC best fit ($z_c = 1.96$)")

ax.axvline(Z_LYA, color=GREY, lw=0.8, ls=":")
ax.annotate(r"Lyman-$\alpha$" "\n" r"$z = 2.33$", xy=(Z_LYA, 12.4),
            ha="center", va="top", fontsize=9, color=GREY)

# (i) single-row inversion
ax.errorbar([Z_LYA], [100 * c0], yerr=[100 * sig], fmt="D", color=INK,
            ms=5.5, capsize=4, lw=1.4, zorder=5)
ax.annotate(r"single-row reading of $D_H/r_d$",
            xy=(Z_LYA + 0.07, 100 * c0 + 0.3), xytext=(2.75, 9.4),
            fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

# (ii) all-probe corridor
ax.fill_between([Z_LYA - 0.10, Z_LYA + 0.10], 2.6, 4.3,
                color=BLUE, alpha=0.25, lw=0, zorder=4)
ax.annotate("all-probe fits (68%)",
            xy=(Z_LYA + 0.11, 3.4), xytext=(2.95, 1.2),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

ax.annotate(r"identically zero above its $z_c$", xy=(2.08, 0.45),
            fontsize=9, color=VERM)

ax.set_xlim(1.5, 5.5)
ax.set_ylim(0, 13)
ax.set_xlabel(r"redshift $z$")
ax.set_ylabel(r"$\Omega_{\rm de}(z)$  [%]")
ax.legend(frameon=False, loc="upper right", fontsize=9.3)
ax.grid(color="#dddddd", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.set_title("(a)  the lever at $z=2.33$", fontsize=10.5, loc="left")

# ---- panel (b): the ledger of the lead ----------------------------------
labels = [b[0] for b in BLOCKS][::-1]
vals = np.array([b[1] for b in BLOCKS][::-1])
ypos = np.arange(len(vals))
colors = [BLUE if v < 0 else GREY for v in vals]
axb.barh(ypos, vals, height=0.58, color=colors, edgecolor="none")
axb.axvline(0, color=INK, lw=0.9)
for y, v in zip(ypos, vals):
    axb.annotate(f"{v:+.1f}", xy=(v + (0.25 if v > 0 else -0.25), y),
                 va="center", ha="left" if v > 0 else "right",
                 fontsize=9.5, color=INK)
axb.set_yticks(ypos, labels, fontsize=9.5)
axb.set_xlim(-5.6, 3.4)
axb.set_ylim(-0.95, 4.05)
axb.set_xlabel(r"$\Delta\chi^2$ (Avrami $-$ $\Lambda$CDM), three-probe battery")
axb.annotate(r"$\leftarrow$ favors the born form", xy=(-5.45, 3.62),
             fontsize=9.2, color=BLUE)
axb.annotate("price $\\rightarrow$", xy=(3.25, 3.62), fontsize=9.2,
             color=GREY, ha="right")
axb.annotate(f"total {TOTAL_RECOMP:+.1f}  (verdicts: {TOTAL_VERDICT:+.1f})",
             xy=(3.25, -0.72), fontsize=9.5, color=INK, ha="right")
axb.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.6)
axb.set_axisbelow(True)
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("(b)  where the lead comes from", fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_lya_lever.png", bbox_inches="tight")
json.dump(dict(z=list(map(float, zgrid)),
               Ode_avrami=list(map(float, ode_av)),
               Ode_lcdm=list(map(float, ode_lc)),
               Ode_thermal=list(map(float, ode_th)),
               row_reading=dict(center=float(c0), sigma=float(sig),
                                dh_component=float(s_dh), wm_component=float(s_wm)),
               corridor_68=[0.026, 0.043]),
          open("/mnt/user-data/outputs/fig_lya_lever_data_v1.json", "w"), indent=1)
print("saved fig_lya_lever.png + fig_lya_lever_data_v1.json")
