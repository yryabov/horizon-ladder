"""
Figure 1C: the Planck plik_lite TTTEEE bandpowers against the battery-3
best fits -- v2 (curves + pulls from the user-side CAMB dump,
cmb_spectra_sn4_v1.json; chi2 validated against the verdicts:
LCDM 585.57 exact, CPL 582.11 exact, Avrami 581.34 vs 581.27).
"""
import json
import numpy as np
from scipy.io import FortranFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = json.load(open('/mnt/user-data/attachments/1d847ab9-785e-596b-9eda-59e7dff91cf4/442bfd7c-cmb_spectra_sn4_v1.json'))
D = '/home/claude/pliklite/data/planck2018_plik_lite/'
bval, X, sig = np.genfromtxt(D + 'cl_cmb_plik_v22.dat', unpack=True)
blmin = np.loadtxt(D + 'blmin.dat').astype(int)
blmax = np.loadtxt(D + 'blmax.dat').astype(int)
blwin = np.loadtxt(D + 'bweight.dat')
LMIN, NTT, NTE, NEE = 30, 215, 199, 199
ELLS = np.arange(30, 2509)
sl = dict(TT=slice(0, NTT), TE=slice(NTT, NTT+NTE), EE=slice(NTT+NTE, None))

def binspec(Dl_full, nb):
    T = np.asarray(Dl_full)[LMIN:2509]*2*np.pi/(ELLS*(ELLS+1.0))
    return np.array([np.sum(blwin[blmin[i]:blmax[i]+1]*T[blmin[i]:blmax[i]+1])
                     for i in range(nb)])

def binned_cl(model):
    m = J[model]; Ap2 = m['params']['Ap']**2
    return np.concatenate([binspec(m['TT'], NTT), binspec(m['TE'], NTE),
                           binspec(m['EE'], NEE)]) / Ap2

V = {k: binned_cl(k) for k in ('LCDM', 'CPL', 'AVRAMI')}
pull_av = (V['AVRAMI'] - V['LCDM']) / sig
pull_cpl = (V['CPL'] - V['LCDM']) / sig
print(f"max |pull| per bin: Avrami {np.max(np.abs(pull_av)):.3f}, CPL {np.max(np.abs(pull_cpl)):.3f}")

def dl_data(name):
    b = bval[sl[name]]; f = b*(b+1)/(2*np.pi)
    return b, X[sl[name]]*f, sig[sl[name]]*f

ell_th = np.arange(len(J['AVRAMI']['TT']))
AP2 = J['AVRAMI']['params']['Ap']**2
TH = dict(TT=np.array(J['AVRAMI']['TT'])/AP2, TE=np.array(J['AVRAMI']['TE'])/AP2,
          EE=np.array(J['AVRAMI']['EE'])/AP2)

BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 11.5, "axes.linewidth": 0.8})
# ---- Figure 1C: the bandpowers and the best fit ---------------------------
fig, axs = plt.subplots(3, 1, figsize=(8.8, 9.2), dpi=220,
                        gridspec_kw=dict(height_ratios=[1.25, 1, 1]))
for ax, name, col, lab, lmax in [
        (axs[0], 'TT', INK,  r"$\mathcal{D}_\ell^{TT}$  [$\mu$K$^2$]", 2508),
        (axs[1], 'TE', BLUE, r"$\mathcal{D}_\ell^{TE}$  [$\mu$K$^2$]", 1996),
        (axs[2], 'EE', VERM, r"$\mathcal{D}_\ell^{EE}$  [$\mu$K$^2$]", 1996)]:
    b, d, e = dl_data(name)
    ax.plot(ell_th[30:lmax+1], TH[name][30:lmax+1], color=GREY, lw=1.0,
            zorder=2)
    ax.errorbar(b, d, yerr=e, fmt="o", color=col, ms=2.4, capsize=0,
                lw=0.9, ls="none", zorder=3)
    ax.set_ylabel(lab)
    ax.set_xlim(0, 2508)
    ax.grid(color="#dddddd", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.annotate(name, xy=(0.972, 0.86), xycoords="axes fraction",
                fontsize=12, ha="right", color=col)
axs[1].axhline(0, color=GREY, lw=0.6, ls=":")
axs[0].set_title("Planck plik_lite bandpowers (613 numbers) and the Avrami best fit",
                 fontsize=11.5, loc="left")
axs[2].set_xlabel(r"multipole $\ell$")
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_cmb_bandpowers.png", bbox_inches="tight")
print("saved fig_cmb_bandpowers.png")

# ---- Figure 2C: the fits against the data ---------------------------------
fig2, (axr, axp) = plt.subplots(2, 1, figsize=(8.8, 6.6), dpi=220,
                                gridspec_kw=dict(height_ratios=[1, 1]))
ib = np.arange(1, 614)
axr.axhline(0, color=INK, lw=0.8, ls=":")
for name, col, z, ms, mfc, mew in (
        ("LCDM", INK, 2, 5.6, "none", 0.6),
        ("CPL", VERM, 3, 4.2, VERM, 0),
        ("AVRAMI", BLUE, 4, 3.0, BLUE, 0)):
    axr.plot(ib, (X - V[name]) / sig, "o", color=col, ms=ms, mfc=mfc,
             mew=mew, zorder=z, ls="none")
for xdiv in (215.5, 414.5):
    axr.axvline(xdiv, color=GREY, lw=0.7)
for xc, name in ((108, 'TT'), (315, 'TE'), (514, 'EE')):
    axr.annotate(name, xy=(xc, 3.45), ha="center", fontsize=11, color=INK)
axr.set_xlim(1, 613)
axr.set_ylim(-4.2, 4.2)
axr.set_ylabel(r"(data $-$ fit) / $\sigma_{\rm bin}$")
axr.grid(color="#dddddd", lw=0.5, alpha=0.6)
axr.set_axisbelow(True)
axr.set_title("residuals of the three best fits", fontsize=11.5, loc="left")

axp.axhline(0, color=INK, lw=1.0)
axp.plot(ib, pull_av, color=BLUE, lw=1.0)
axp.plot(ib, pull_cpl, color=VERM, lw=1.0, ls="--")
for xdiv in (215.5, 414.5):
    axp.axvline(xdiv, color=GREY, lw=0.7)
for xc, name in ((108, 'TT'), (315, 'TE'), (514, 'EE')):
    axp.annotate(name, xy=(xc, 0.205), ha="center", fontsize=11, color=INK)
axp.set_xlim(1, 613)
axp.set_ylim(-0.25, 0.25)
axp.set_xlabel("bandpower index")
axp.set_ylabel(r"$\Delta\mathcal{D}$ / $\sigma_{\rm bin}$")
axp.grid(color="#dddddd", lw=0.5, alpha=0.6)
axp.set_axisbelow(True)
axp.set_title("the front-runners against the $\\Lambda$CDM best fit, in units of the bin error",
              fontsize=11.5, loc="left")
fig2.tight_layout()
fig2.savefig("/mnt/user-data/outputs/fig_cmb_fits.png", bbox_inches="tight")
print("saved fig_cmb_fits.png")
