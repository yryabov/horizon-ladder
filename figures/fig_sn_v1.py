"""
Figure draft: the Pantheon+ Hubble diagram in residuals (Sec. 4.3) -- v1
========================================================================
Binned Pantheon+ residuals relative to the LCDM best fit of Table 3, with
the Avrami and CPL best-fit histories drawn as offsets from the same LCDM
baseline.  Battery convention VERIFIED against the verdicts: zHD > 0.023
(1371 SNe), full STAT+SYS covariance, absolute magnitude M profiled
analytically (chi2_SN: Avrami 1210.0 vs verdict-implied 1210.1; LCDM
1211.8 vs 1212.4).  Display bins are weighted by the full covariance
restricted to each bin (cross-bin correlations not shown).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KM = 299792.458
WR = 2.469e-5 * (1 + 0.2271 * 3.046)
D = "/home/claude/pantheonplussh0es/datarelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/"

# ---- data ----------------------------------------------------------------
hdr = open(D + "Pantheon+SH0ES.dat").readline().split()
raw = np.genfromtxt(D + "Pantheon+SH0ES.dat", names=hdr, skip_header=1,
                    dtype=None, encoding=None)
COV = np.loadtxt(D + "Pantheon+SH0ES_STAT+SYS.cov", skiprows=1).reshape(len(raw), len(raw))
cut = raw['zHD'] > 0.023
z = raw['zHD'][cut]
mb = raw['m_b_corr'][cut]
sig = raw['m_b_corr_err_DIAG'][cut]
Ci = np.linalg.inv(COV[np.ix_(cut, cut)])
print(f"SNe after zHD>0.023 cut: {len(z)} of {len(raw)}")

# ---- best fits (Table 3) -------------------------------------------------
AVRAMI = dict(wb=0.0223960224, wc=0.1200090090, mnu=0.0590788467,
              h=0.6790651632, lam=0.9026216626, zc=4.1125308865)
LCDM = dict(wb=0.0225181049, wc=0.1180185449, mnu=0.0581851433,
            h=0.6826523697)
W0C, WAC, HC = -0.8725307554078238, -0.48816838385924444, 0.6773283850798444
WMC = 0.022440940100811296 + 0.11927028645549345 + 0.058357845558898586/93.14

def make_grid(xmax=13.0, n=3200):
    xg = np.linspace(np.log(1/2500), xmax, n)
    return xg, xg[1]-xg[0], np.exp(xg)

def RE_of(H, AA, DX):
    f = 1/(AA*H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*DX)[::-1],[0]])
    return AA*(cum + 1/(AA[-1]*H[-1]))

def solve_avrami(p):
    XG, DX, AA = make_grid()
    xc = np.log(1/(1+p['zc']))
    wm = p['wb']+p['wc']+p['mnu']/93.14
    al0 = np.where(XG > xc, 1-np.exp(-((XG-xc)/p['lam'])**2), 0.0)
    lo, hi = 0.02, 2.4
    for _ in range(44):
        mid = (lo+hi)/2
        r = np.full_like(XG, mid)
        for _ in range(300):
            H = np.sqrt(wm*AA**-3 + WR*AA**-4 + np.maximum(r,0))
            RE = RE_of(H, AA, DX)
            r = 0.58*r + 0.42*(al0/max(al0[-1],1e-12))/RE**2
        h = float(np.interp(1.0, AA, H))
        if abs(h-p['h']) < 3e-4: break
        lo, hi = (mid, hi) if h < p['h'] else (lo, mid)
    return XG, H*100  # H(z) in km/s/Mpc units via h*100 scaling of E

def Ez_lcdm(zz, p):
    wm = p['wb']+p['wc']+p['mnu']/93.14
    h2 = p['h']**2
    wde = h2 - wm - WR
    return np.sqrt(wm*(1+zz)**3 + WR*(1+zz)**4 + wde)/p['h']

def Ez_cpl(zz):
    a = 1/(1+zz)
    wde = HC**2 - WMC - WR
    rho = wde * a**(-3*(1+W0C+WAC)) * np.exp(-3*WAC*(1-a))
    return np.sqrt(WMC*(1+zz)**3 + WR*(1+zz)**4 + rho)/HC

def mu_from_E(zz, Efun, h):
    zg = np.linspace(0, 2.4, 2000)
    invE = 1/Efun(np.maximum(zg, 1e-12))
    invE[0] = 1.0
    dc = np.cumsum((invE[:-1]+invE[1:])/2*np.diff(zg))
    dc = np.concatenate([[0], dc]) * C_KM/(100*h)
    dL = (1+zz)*np.interp(zz, zg, dc)
    return 5*np.log10(np.maximum(dL,1e-9)) + 25

# Avrami: E(z) from solver
XG, Hav = solve_avrami(AVRAMI)
zg_av = 1/np.exp(XG) - 1
def Ez_av(zz):
    o = np.argsort(zg_av)
    return np.interp(zz, zg_av[o], (Hav/100/AVRAMI['h'])[o])

mu_l = mu_from_E(z, lambda q: Ez_lcdm(q, LCDM), LCDM['h'])
mu_a = mu_from_E(z, Ez_av, AVRAMI['h'])
mu_c = mu_from_E(z, Ez_cpl, HC)

# profile out M per model (GLS offset, full covariance)
ONE = np.ones_like(mb)
CiONE = Ci @ ONE
DEN = ONE @ CiONE
def Moff(mu): return ((mb - mu) @ CiONE) / DEN
def chi2_sn(mu):
    r = mb - mu - Moff(mu)
    return r @ Ci @ r
res_l = mb - mu_l - Moff(mu_l)
dmu_a = (mu_a + Moff(mu_a)) - (mu_l + Moff(mu_l))
dmu_c = (mu_c + Moff(mu_c)) - (mu_l + Moff(mu_l))

print(f"chi2_SN check: LCDM {chi2_sn(mu_l):.2f}, Avrami {chi2_sn(mu_a):.2f}, CPL {chi2_sn(mu_c):.2f}")
# binned residuals (equal-count bins in log z; GLS weights from the full
# covariance restricted to each bin)
NB = 24
edges = np.quantile(np.log(z), np.linspace(0, 1, NB+1))
zb, rb, eb = [], [], []
for i in range(NB):
    m = (np.log(z) >= edges[i]) & (np.log(z) <= edges[i+1])
    Cb = np.linalg.inv(COV[np.ix_(cut, cut)][np.ix_(m, m)])
    wb_ = Cb @ np.ones(m.sum())
    den = wb_.sum()
    zb.append(np.exp(np.mean(np.log(z[m]))))
    rb.append((res_l[m] @ wb_) / den)
    eb.append(1/np.sqrt(den))
zb, rb, eb = map(np.array, (zb, rb, eb))

zc_grid = np.geomspace(0.01, 2.4, 300)
mu_l_g = mu_from_E(zc_grid, lambda q: Ez_lcdm(q, LCDM), LCDM['h'])
dmu_a_g = mu_from_E(zc_grid, Ez_av, AVRAMI['h']) + Moff(mu_a) - mu_l_g - Moff(mu_l)
dmu_c_g = mu_from_E(zc_grid, Ez_cpl, HC) + Moff(mu_c) - mu_l_g - Moff(mu_l)

BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 11.5, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=220)
ax.scatter(z, res_l, s=5, color=GREY, alpha=0.8, lw=0, rasterized=True,
           label="Pantheon+ ($z_{\\rm HD} > 0.023$, 1371 SNe)")
ax.errorbar(zb, rb, yerr=eb, fmt="D", color=INK, ms=4.5, capsize=3,
            lw=1.2, ls="none", zorder=6)
ax.axhline(0, color=INK, lw=0.9, ls=":")
ax.plot(zc_grid, dmu_a_g, color=BLUE, lw=2.0,
        label="Avrami best fit $-$ $\\Lambda$CDM (Table 3)")
ax.plot(zc_grid, dmu_c_g, color=VERM, lw=1.6, ls="--",
        label="CPL tangent $-$ $\\Lambda$CDM (Table 3)")
ax.set_xscale("log")
ax.set_xlim(0.02, 2.4)
ax.set_ylim(-0.19, 0.19)
ax.set_xlabel(r"redshift $z$")
ax.set_ylabel(r"$\Delta\mu$  [mag], relative to the $\Lambda$CDM best fit")
ax.grid(color="#dddddd", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_sn_residuals.png", bbox_inches="tight")
print("saved fig_sn_residuals.png")
