"""
Dump the CMB spectra of the three sn4 (battery-3) best fits -- v1
=================================================================
Run this next to your campaign environment (needs camb only).  It computes
the lensed TT/TE/EE spectra of the Table-3 best fits -- LCDM, CPL, and the
Avrami form (w(a) table via DarkEnergyPPF, exactly as in the surf_* fit
scripts: same solver grid, relax, clip and table layout) -- and writes
cmb_spectra_sn4_v1.json (~1.5 MB) for the paper's CMB data figure.

Conventions copied verbatim from surf_avrami_unlocked_ttteee (lmax 2700,
lens_potential_accuracy=1, CMB_unit='muK', spectra['total']).
Expected check values printed at the end come from the verdict files.
"""
import json
import numpy as np
import camb

# ---- Table-3 (sn4) best-fit vectors, verbatim from the verdict JSONs ----
AVRAMI = dict(ombh2=0.022396022422168718, omch2=0.12000900897740749,
              mnu=0.05907884673878404, h=0.6790651631727402,
              lam=0.9026216625602139, zc=4.112530886485031,
              As=2.1310710093445056e-09, ns=0.9645963400742643,
              tau=0.06159802408955225, Ap=1.0004939548986043)
CPL = dict(ombh2=0.022440940100811296, omch2=0.11927028645549345,
           mnu=0.058357845558898586, h=0.6773283850798444,
           w0=-0.8725307554078238, wa=-0.48816838385924444,
           As=2.133594645786487e-09, ns=0.9662284896797569,
           tau=0.06299031195314392, Ap=1.0005598714085582)
LCDM = dict(ombh2=0.022518104920466478, omch2=0.11801854491191927,
            mnu=0.05818514327436684, h=0.6826523696508386,
            As=2.1279406488204076e-09, ns=0.9693653263792881,
            tau=0.06302062567695499, Ap=1.0007585580121172)

# ---- nucleation solver (identical to the fit scripts) --------------------
WG = 2.469e-5
WRAD = WG * (1 + 0.2271 * 3.046)
_XG = np.linspace(np.log(1 / 2500), 13.0, 3200)
_A = np.exp(_XG)
_DX = _XG[1] - _XG[0]


def _solve(wm, lam, zc, rho0=0.30, n_iter=220, relax=0.45):
    xc = np.log(1.0 / (1.0 + zc))
    t = np.maximum(_XG - xc, 0.0) / lam
    alpha = 1.0 - np.exp(-t ** 2)
    a = _A
    rho = np.full_like(_XG, rho0)
    for _ in range(n_iter):
        H = np.sqrt(wm * a ** -3 + WRAD * a ** -4 + np.maximum(rho, 0))
        f = 1 / (a * H)
        cum = np.concatenate([np.cumsum((f[::-1][:-1] + f[::-1][1:]) / 2 * _DX)[::-1], [0]])
        RE = a * (cum + 1 / (a[-1] * H[-1]))
        rho = (1 - relax) * rho + relax * alpha / RE ** 2
    H = np.sqrt(wm * a ** -3 + WRAD * a ** -4 + rho)
    return H, rho


def _solve_target(wm, lam, zc, hT):
    lo, hi = 0.02, 2.4
    H = rho = None
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        H, rho = _solve(wm, lam, zc, mid)
        h = float(np.interp(1.0, _A, H))
        if abs(h - hT) < 4e-4:
            break
        if h < hT:
            lo = mid
        else:
            hi = mid
    return H, rho


def surf_table(wm, lam, zc, hT, n_table=600):
    H, rho = _solve_target(wm, lam, zc, hT)
    rho0 = float(np.interp(0.0, _XG, rho))
    rho_f = np.maximum(rho, rho0 * 1e-12)
    w = -1 - np.gradient(np.log(rho_f), _XG) / 3
    w = np.maximum(w, -60.0)
    zlo = max(0.4, 0.5 * zc)
    zhi = min(30.0, 3.5 * (1 + zc))
    lnat = np.unique(np.concatenate([
        np.linspace(-10., 0., n_table),
        np.linspace(np.log(1 / (1 + zhi)), np.log(1 / (1 + zlo)), 4 * n_table // 3)]))
    at = np.exp(lnat)
    wt = np.interp(lnat, _XG, w)
    return at, wt


def spectra(p, de=None):
    pN = camb.CAMBparams()
    pN.set_cosmology(H0=100 * p['h'], ombh2=p['ombh2'], omch2=p['omch2'],
                     mnu=max(0.001, p['mnu']), omk=0, tau=p['tau'])
    pN.InitPower.set_params(As=p['As'], ns=p['ns'], r=0)
    if de is not None:
        pN.DarkEnergy = de
    pN.set_for_lmax(2700, lens_potential_accuracy=1)
    pN.WantTensors = False
    spec = camb.get_results(pN).get_cmb_power_spectra(pN, CMB_unit='muK')['total']
    return spec  # (n x 4): [TT, EE, BB, TE], D_ell muK^2


out = {}

# LCDM
spec = spectra(LCDM)
out['LCDM'] = dict(params=LCDM, TT=spec[:, 0].tolist(),
                   EE=spec[:, 1].tolist(), TE=spec[:, 3].tolist())

# CPL via PPF
de = camb.dark_energy.DarkEnergyPPF()
de.set_params(w=CPL['w0'], wa=CPL['wa'])
spec = spectra(CPL, de)
out['CPL'] = dict(params=CPL, TT=spec[:, 0].tolist(),
                  EE=spec[:, 1].tolist(), TE=spec[:, 3].tolist())

# Avrami via w(a) table
wm = AVRAMI['ombh2'] + AVRAMI['omch2'] + AVRAMI['mnu'] / 93.14
at, wt = surf_table(wm, AVRAMI['lam'], AVRAMI['zc'], AVRAMI['h'])
de = camb.dark_energy.DarkEnergyPPF()
de.set_w_a_table(at, wt)
spec = spectra(AVRAMI, de)
out['AVRAMI'] = dict(params=AVRAMI, TT=spec[:, 0].tolist(),
                     EE=spec[:, 1].tolist(), TE=spec[:, 3].tolist())

json.dump(out, open('cmb_spectra_sn4_v1.json', 'w'))
print('saved cmb_spectra_sn4_v1.json')
print('Expected plik_lite chi2 at these vectors (verdicts, for your own check')
print('with your chi2_ttteee): LCDM 585.57, CPL 582.11, Avrami 581.27')
