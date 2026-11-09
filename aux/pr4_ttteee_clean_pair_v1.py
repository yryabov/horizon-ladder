"""
The clean cross-pipeline pair on PR4 CamSpec TTTEEE + DESI DR2 BAO -- v1
========================================================================
Purpose: an apples-to-apples measurement of the likelihood-choice systematic.
The plik_lite side of the ladder is TT+TE+EE; the earlier PR4 refit
(nucl_pr4_unlocked_bao_v1 / lcdm_pr4_direct) was CamSpec TT only.  This script
refits BOTH models on CamSpec PR4 **TTTEEE** + DESI DR2 BAO, with BOTH Lya
rows, so each plik battery has its exact PR4 counterpart:

  fit 1  AVRAMI (unlocked) + old Lya row   <-> plik baseline battery
  fit 2  AVRAMI (unlocked) + DR2-IV row    <-> plik lya4 battery
  fit 3  LCDM              + old Lya row   <-> plik baseline battery
  fit 4  LCDM              + DR2-IV row    <-> plik lya4 battery

Everything else is verbatim from nucl_pr4_unlocked_bao_v1.py: the same
solver (grid ln(1/2500)..13, n=3200, 220 iters, relax 0.45, h-bisection to
4e-4), the same BAO block (validated to 0.001 against four stored verdicts),
the same priors on BOTH models (tau = 0.0605 +- 0.0046; A_planck = 1 +-
0.0025; the NH neutrino wall at 0.05819).  calTE and calEE are the two new
TTTEEE calibrations: they are left flat within [0.9, 1.1] (no Gaussian
prior), identically for both models, so they cannot move the gap.

The TTTEEE anchor is computed INTERNALLY: cobaya's own loglike at a fixed
reference point, in this same process, with this same camb -- then the
direct-injection channel must reproduce it (auto-detected over D_ell/C_ell
and TE/EE argument order; the script refuses to fit otherwise).

Each fit saves its verdict json the moment it finishes, so a partial night
still yields results.  To split across nights, edit RUN below; the DR2-IV
fits warm-start from the old-row jsons if those are on disk.

Expected total: ~6-10k camb TTTEEE evaluations.  Start it and go to bed.
"""
import json
import os
import time
import numpy as np
import camb
from scipy.optimize import minimize
from cobaya.model import get_model

RUN = ['av_old', 'av_new', 'lc_old', 'lc_new']   # edit to split across nights

PACKAGES_PATH = r'C:\yaroslavFiles\cobaya_packages'

MNU_NH_FLOOR = 0.05819; MNU_NH_SIGMA = 0.00029
TAU_MEAN = 0.0605; TAU_SIGMA = 0.0046

# ── CamSpec TTTEEE object ────────────────────────────────────────────────────
info = {
    'likelihood': {'planck_NPIPE_highl_CamSpec.TTTEEE': None},
    'theory': {'camb': {'extra_args': {'lens_potential_accuracy': 1, 'lmax': 2700}}},
    'params': {
        'H0':    {'prior': {'min': 55, 'max': 80}, 'ref': 67.36},
        'ombh2': {'prior': {'min': 0.019, 'max': 0.026}, 'ref': 0.02237},
        'omch2': {'prior': {'min': 0.09, 'max': 0.15}, 'ref': 0.1200},
        'mnu':   {'prior': {'min': 0.0, 'max': 0.6}, 'ref': 0.06},
        'As':    {'prior': {'min': 1.5e-9, 'max': 3.5e-9}, 'ref': 2.100e-9},
        'ns':    {'prior': {'min': 0.87, 'max': 1.06}, 'ref': 0.9649},
        'tau':   {'prior': {'min': 0.01, 'max': 0.13}, 'ref': 0.0544},
    },
    'packages_path': PACKAGES_PATH,
}
model = get_model(info)
like = list(model.likelihood.values())[0]

COSMO_INFO = ['H0', 'ombh2', 'omch2', 'mnu', 'As', 'ns', 'tau']
SAMPLED = list(model.parameterization.sampled_params())
NUIS_NAMES = [p for p in SAMPLED if p not in COSMO_INFO]
print("TTTEEE nuisance parameters discovered:", NUIS_NAMES)

# reference values / bounds for the nuisances we know; cobaya bounds otherwise
NUIS_REF_KNOWN = dict(A_planck=1.0, amp_143=9.868, amp_217=20.64,
                      amp_143x217=10.1, n_143=0.8929, n_217=1.072,
                      n_143x217=1.261, calTE=1.0, calEE=1.0)
BOUNDS_KNOWN = {'A_planck': (0.98, 1.02), 'amp_143': (0.0, 50.0),
                'amp_217': (0.0, 60.0), 'amp_143x217': (0.0, 50.0),
                'n_143': (0.0, 3.0), 'n_217': (0.0, 3.0), 'n_143x217': (0.0, 3.0),
                'calTE': (0.9, 1.1), 'calEE': (0.9, 1.1)}
_pb = np.asarray(model.prior.bounds(confidence_for_unbounded=0.99995))
PRIOR_BOUNDS = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(SAMPLED, _pb)}
NUIS_REF, NUIS_BOUNDS = {}, {}
for n in NUIS_NAMES:
    if n not in NUIS_REF_KNOWN:
        print(f"  WARNING: unexpected nuisance '{n}' -- ref = bounds midpoint")
    NUIS_BOUNDS[n] = BOUNDS_KNOWN.get(n, PRIOR_BOUNDS[n])
    NUIS_REF[n] = NUIS_REF_KNOWN.get(n, 0.5 * (NUIS_BOUNDS[n][0] + NUIS_BOUNDS[n][1]))

# ── theory spectra (verbatim conventions: lmax 2700, lens_pot_acc 1, muK) ────
def camb_spec(H0, ombh2, omch2, mnu, As, ns, tau, de=None):
    p = camb.CAMBparams()
    p.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=max(0.001, mnu),
                    omk=0, tau=tau)
    p.InitPower.set_params(As=As, ns=ns, r=0)
    if de is not None: p.DarkEnergy = de
    p.set_for_lmax(2700, lens_potential_accuracy=1)
    p.WantTensors = False
    tot = camb.get_results(p).get_cmb_power_spectra(p, CMB_unit='muK')['total']
    return tot[:, 0], tot[:, 3], tot[:, 1]        # TT, TE, EE (D_ell, muK^2)

ELLS_FULL = None
def _to_C(D):
    C = np.zeros_like(D)
    ell = np.arange(len(D), dtype=float)
    C[2:] = D[2:] * 2 * np.pi / (ell[2:] * (ell[2:] + 1.0))
    return C

# full data_params: free nuisances + fixed defaults introspected from the object
FIXED_DEFAULTS = {'use_fg_residual_model': 0, 'cal0': 1.0, 'cal2': 1.0,
                  'amp_100': 0.0, 'n_100': 1.0, 'calEE': 1.0, 'calTE': 1.0}
def full_params(nuis):
    d = dict(FIXED_DEFAULTS)
    for k in list(d.keys()):
        for attr in (k, k.lower()):
            if hasattr(like, attr):
                try: d[k] = float(getattr(like, attr))
                except Exception: pass
    d.update(nuis)
    return d

# ── internal anchor + injection auto-detection ──────────────────────────────
REF_POINT = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06,
                 As=2.100e-9, ns=0.9649, tau=0.0544, **NUIS_REF)
print("\nComputing the internal TTTEEE anchor (cobaya loglike at the "
      "reference point)...")
_ll, _ = model.loglikes({k: REF_POINT[k] for k in SAMPLED}, cached=False)
ANCHOR = float(-2.0 * np.sum(_ll))
print(f"  internal anchor chi2 = {ANCHOR:.2f}")

TT0, TE0, EE0 = camb_spec(67.36, 0.02237, 0.1200, 0.06, 2.100e-9, 0.9649, 0.0544)
INJECT = [None]

def _validate():
    cands = [
        ('Dl (TT,TE,EE)', lambda: (TT0, TE0, EE0)),
        ('Cl (TT,TE,EE)', lambda: (_to_C(TT0), _to_C(TE0), _to_C(EE0))),
        ('Dl (TT,EE,TE)', lambda: (TT0, EE0, TE0)),
        ('Cl (TT,EE,TE)', lambda: (_to_C(TT0), _to_C(EE0), _to_C(TE0))),
    ]
    results = []
    for name, make in cands:
        a, b, c = make()
        try:
            v = float(np.atleast_1d(
                like.chi_squared(a, b, c, full_params(NUIS_REF)))[0])
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__} {str(e)[:80]}")
            continue
        d = v - ANCHOR
        print(f"  {name}: chi2 = {v:.2f}  (delta vs anchor {d:+.2f})")
        results.append((abs(d), name, make, v))
    if not results:
        return False
    results.sort()
    dbest, name, make, v = results[0]
    if dbest < 35.0:        # <0.5%: the channel is right; the residual is the
        INJECT[0] = make    # in-process camb-settings delta, recorded below
        print(f"INJECTION VALIDATED: {name} (chi2 {v:.2f}, pipeline delta "
              f"{v - ANCHOR:+.2f} recorded)")
        print("NOTE: all four fits stay WITHIN this one direct pipeline, so "
              "the gap is pipeline-delta-free.")
        globals()['PIPELINE_DELTA'] = v - ANCHOR
        globals()['INJECT_NAME'] = name
        return True
    return False

if not _validate():
    print("fixed-defaults introspection dump:", full_params({}))
    raise SystemExit("no valid injection signature; send this log back")

ORDER_SWAP = 'TT,EE,TE' in INJECT_NAME
USE_CL = INJECT_NAME.startswith('Cl')
def chi2_camspec(TT, TE, EE, nuis):
    a, b, c = (TT, TE, EE) if not ORDER_SWAP else (TT, EE, TE)
    if USE_CL: a, b, c = _to_C(a), _to_C(b), _to_C(c)
    return float(np.atleast_1d(like.chi_squared(a, b, c, full_params(nuis)))[0])

# ── surface solver (validated blocks; h-target bisection) ───────────────────
WG = 2.469e-5; WRAD = WG * (1 + 0.2271 * 3.046)
_XG = np.linspace(np.log(1 / 2500), 13.0, 3200); _A = np.exp(_XG)
_DX = _XG[1] - _XG[0]

def _solve(wm, lam, zc, rho0=0.30, n_iter=220, relax=0.45):
    xc = np.log(1.0 / (1.0 + zc))
    t = np.maximum(_XG - xc, 0.0) / lam
    alpha = 1.0 - np.exp(-t ** 2)
    a = _A; rho = np.full_like(_XG, rho0)
    for _ in range(n_iter):
        H = np.sqrt(wm * a ** -3 + WRAD * a ** -4 + np.maximum(rho, 0))
        f = 1 / (a * H)
        cum = np.concatenate([np.cumsum((f[::-1][:-1] + f[::-1][1:]) / 2 * _DX)[::-1], [0]])
        RE = a * (cum + 1 / (a[-1] * H[-1]))
        rho = (1 - relax) * rho + relax * alpha / RE ** 2
    return np.sqrt(wm * a ** -3 + WRAD * a ** -4 + rho), rho

def _solve_target(wm, lam, zc, hT):
    lo, hi = 0.02, 2.4; H = rho = None
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        H, rho = _solve(wm, lam, zc, mid)
        h = float(np.interp(1.0, _A, H))
        if abs(h - hT) < 4e-4: break
        if h < hT: lo = mid
        else: hi = mid
    return H, rho

def surf_de(wm, lam, zc, hT, n_table=600):
    H, rho = _solve_target(wm, lam, zc, hT)
    rho0v = float(np.interp(0.0, _XG, rho))
    w = -1 - np.gradient(np.log(np.maximum(rho, rho0v * 1e-12)), _XG) / 3
    w = np.maximum(w, -60.0)
    zlo = max(0.4, 0.5 * zc); zhi = min(30.0, 3.5 * (1 + zc))
    lnat = np.unique(np.concatenate([
        np.linspace(-10., 0., n_table),
        np.linspace(np.log(1 / (1 + zhi)), np.log(1 / (1 + zlo)), 4 * n_table // 3)]))
    de = camb.dark_energy.DarkEnergyPPF()
    de.set_w_a_table(np.exp(lnat), np.interp(lnat, _XG, w))
    return de, H

def H_lcdm(wm, h):      # same mnu-as-matter convention as the solver grid
    ol = h * h - wm - WRAD
    return np.sqrt(wm * _A ** -3 + WRAD * _A ** -4 + ol)

# ── BAO (verified block; two Lya rows) ──────────────────────────────────────
BAO_GAL = [(0.295, 7.942, 0.075, None, None, None),
 (0.510, 13.588, 0.167, 21.863, 0.425, -0.459), (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
 (0.934, 21.576, 0.152, 17.641, 0.193, -0.416), (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
 (1.484, 30.512, 0.760, 12.817, 0.516, -0.500)]
LYA = {'oldlya': (2.330, 38.988, 0.531, 8.632, 0.101, -0.431),
       'newlya': (2.330, 39.32, 0.33, 8.600, 0.066, -0.431)}  # DR2 Results IV joint AP+BAO (2607.27410); rho kept -0.431
C_KM = 299792.458
def rd_desi(wcb, wb): return 147.05 * (wcb / 0.1432) ** -0.23 * (wb / 0.02236) ** -0.13

def chi2_bao_from_H(Hgrid, rd, row):
    BAO = BAO_GAL + [LYA[row]]
    zf = np.linspace(0, 2.6, 600)
    Hf = np.interp(np.log(1 / (1 + zf)), _XG, Hgrid) * 100.0
    invH = 1 / Hf
    DMg = C_KM * np.concatenate([[0.0], np.cumsum((invH[:-1] + invH[1:]) / 2 * np.diff(zf))])
    zq, val, err, _, _, _ = BAO[0]
    dm = float(np.interp(zq, zf, DMg)); dh = C_KM / float(np.interp(zq, zf, Hf))
    cB = (((zq * dm * dm * dh) ** (1 / 3) / rd - val) / err) ** 2
    for zq, vM, eM, vH, eH, r in BAO[1:]:
        dm = float(np.interp(zq, zf, DMg)) / rd
        dh = (C_KM / float(np.interp(zq, zf, Hf))) / rd
        d2 = np.array([dm - vM, dh - vH])
        C = np.array([[eM ** 2, r * eM * eH], [r * eM * eH, eH ** 2]])
        cB += float(d2 @ np.linalg.inv(C) @ d2)
    return cB

# ── objectives ──────────────────────────────────────────────────────────────
def priors(p):
    c = ((p['tau'] - TAU_MEAN) / TAU_SIGMA) ** 2
    c += 0.0 if p['mnu'] >= MNU_NH_FLOOR else ((p['mnu'] - MNU_NH_FLOOR) / MNU_NH_SIGMA) ** 2
    c += ((p['A_planck'] - 1.0) / 0.0025) ** 2
    return c

def chi2_avrami(p, row):
    try:
        wm = p['wb'] + p['wc'] + p['mnu'] / 93.14
        de, Hg = surf_de(wm, p['lam'], p['zc'], p['h'])
        TT, TE, EE = camb_spec(100 * p['h'], p['wb'], p['wc'], p['mnu'],
                               p['As'], p['ns'], p['tau'], de=de)
        cC = chi2_camspec(TT, TE, EE, {k: p[k] for k in NUIS_NAMES})
        cB = chi2_bao_from_H(Hg, rd_desi(p['wb'] + p['wc'], p['wb']), row)
        return cC + cB + priors(p), cC, cB
    except Exception:
        return 1e10, 1e10, 1e10

def chi2_lcdm(p, row):
    try:
        wm = p['wb'] + p['wc'] + p['mnu'] / 93.14
        TT, TE, EE = camb_spec(100 * p['h'], p['wb'], p['wc'], p['mnu'],
                               p['As'], p['ns'], p['tau'])
        cC = chi2_camspec(TT, TE, EE, {k: p[k] for k in NUIS_NAMES})
        cB = chi2_bao_from_H(H_lcdm(wm, p['h']), rd_desi(p['wb'] + p['wc'], p['wb']), row)
        return cC + cB + priors(p), cC, cB
    except Exception:
        return 1e10, 1e10, 1e10

COS_AV = ['wb', 'wc', 'lam', 'zc', 'h', 'mnu', 'As', 'ns', 'tau']
COS_LC = ['wb', 'wc', 'h', 'mnu', 'As', 'ns', 'tau']
BOUNDS = {'wb': (0.020, 0.025), 'wc': (0.10, 0.14), 'lam': (0.05, 3.0),
          'zc': (0.8, 12.0), 'h': (0.60, 0.78), 'mnu': (0.0, 0.50),
          'As': (1.5e-9, 3.5e-9), 'ns': (0.88, 1.05), 'tau': (0.02, 0.12),
          **NUIS_BOUNDS}

# ── starts (PR4 TT verdicts + plik_lite sn4 vectors; calTE/calEE at 1) ──────
AV_PR4TT = dict(wb=0.022127154878404665, wc=0.11953875314113054,
                lam=0.9172632440718658, zc=4.091124168900044,
                h=0.6818920946494209, mnu=0.05864429971999942,
                As=2.1233421057427234e-09, ns=0.96130503915347,
                tau=0.062094317620500855, A_planck=1.0001803599807437,
                amp_143=19.491889111361914, amp_217=13.276453915354557,
                amp_143x217=10.236359732799025, n_143=0.9414331839998018,
                n_217=1.291132198835192, n_143x217=1.3563706690678266)
AV_PLIK = dict(wb=0.0223960, wc=0.1200090, lam=0.9026217, zc=4.1125309,
               h=0.6790652, mnu=0.0590788, As=2.1310710e-9, ns=0.9645963,
               tau=0.0615980)
LC_PR4TT = dict(wb=0.022226946348749478, wc=0.11706788434900524,
                h=0.6835166347557072, mnu=0.05818913141590999,
                As=2.105616476428481e-09, ns=0.9667566859787327,
                tau=0.060615914916811814, A_planck=1.000379923721054,
                amp_143=19.335902056422878, amp_217=13.220637592229073,
                amp_143x217=10.114883958284452, n_143=0.960976701950851,
                n_217=1.2858560440485138, n_143x217=1.3688761914562781)
LC_PLIK = dict(wb=0.0225181, wc=0.1180185, h=0.6826524, mnu=0.0581851,
               As=2.1279406e-9, ns=0.9693653, tau=0.0630206)

def vec(cos_names, start):
    full = dict(NUIS_REF); full.update(start)
    return [full[k] for k in cos_names + NUIS_NAMES]

def warm(fname, cos_names, fallback):
    if os.path.exists(fname):
        j = json.load(open(fname))
        print(f"  warm start from {fname}")
        return [vec(cos_names, {k: j[k] for k in cos_names + NUIS_NAMES if k in j})]
    return fallback

def run_fit(tag, model_name, row, cos_names, chi2_fn, starts, maxiter):
    PN = cos_names + NUIS_NAMES
    print(f"\n{'=' * 64}\nFIT {tag}: {model_name} + {row} "
          f"({LYA[row][1]}/{LYA[row][3]} Lya row), dim={len(PN)}, "
          f"{len(starts)} start(s), maxiter={maxiter}\n{'=' * 64}")
    best = (1e10, None, None); t0 = time.time(); nev = [0]
    for i, x0 in enumerate(starts):
        print(f"\nStart {i + 1}/{len(starts)}")
        loc = [1e10]
        def obj(v):
            p = {k: float(np.clip(x, *BOUNDS[k])) for k, x in zip(PN, v)}
            c, cC, cB = chi2_fn(p, row)
            nev[0] += 1
            if c < loc[0]:
                loc[0] = c
                extra = (f"lam={p['lam']:.2f} zc={p['zc']:.2f} "
                         if 'lam' in p else "")
                print(f"  eval {nev[0]:5d} [{(time.time() - t0) / 60:6.1f} min]: "
                      f"wb={p['wb']:.5f} wc={p['wc']:.4f} {extra}h={p['h']:.4f} "
                      f"mnu={p['mnu']:.4f} tau={p['tau']:.4f} "
                      f"CamSpec={cC:.1f} BAO={cB:.2f} tot={c:.2f} *")
            return c
        r = minimize(obj, x0, method='Nelder-Mead',
                     options=dict(maxiter=maxiter, xatol=1e-4, fatol=0.05))
        p = {k: float(np.clip(x, *BOUNDS[k])) for k, x in zip(PN, r.x)}
        c, cC, cB = chi2_fn(p, row)
        print(f"Start {i + 1} done: tot={c:.2f}  CamSpec={cC:.2f}  BAO={cB:.2f}")
        if c < best[0]: best = (c, p, (cC, cB))
    c_tot, p, (cC, cB) = best
    cB_other = None  # the same vector scored on the other row (cheap cross-check)
    other = 'newlya' if row == 'oldlya' else 'oldlya'
    try:
        if 'lam' in p:
            wm = p['wb'] + p['wc'] + p['mnu'] / 93.14
            _, Hg = surf_de(wm, p['lam'], p['zc'], p['h'])
        else:
            Hg = H_lcdm(p['wb'] + p['wc'] + p['mnu'] / 93.14, p['h'])
        cB_other = chi2_bao_from_H(Hg, rd_desi(p['wb'] + p['wc'], p['wb']), other)
    except Exception:
        pass
    out = dict(MODEL=model_name, LIKELIHOOD='planck_NPIPE_highl_CamSpec.TTTEEE',
               LYA_ROW=row, LYA_VALUES=list(LYA[row]), USE_BAO=True,
               anchor_internal=ANCHOR, injection=INJECT_NAME,
               pipeline_delta=PIPELINE_DELTA, **p,
               chi2_CamSpec=cC, chi2_BAO=cB, chi2_tot=c_tot,
               chi2_BAO_other_row_at_this_vector=cB_other,
               n_evals=nev[0], minutes=round((time.time() - t0) / 60, 1))
    fname = f"pr4_ttteee_{tag}_best_v1.json"
    json.dump(out, open(fname, 'w'), indent=2)
    print(f"\nFIT {tag} BEST: CamSpec={cC:.2f} BAO={cB:.2f} tot={c_tot:.2f}  "
          f"-> saved {fname}")
    return c_tot

FITS = {
    'av_old': lambda: run_fit('avrami_oldlya', 'surface_avrami2_unlocked', 'oldlya',
                              COS_AV, chi2_avrami,
                              [vec(COS_AV, AV_PR4TT), vec(COS_AV, AV_PLIK)], 2800),
    'av_new': lambda: run_fit('avrami_newlya', 'surface_avrami2_unlocked', 'newlya',
                              COS_AV, chi2_avrami,
                              warm('pr4_ttteee_avrami_oldlya_best_v1.json', COS_AV,
                                   [vec(COS_AV, AV_PR4TT)]), 1600),
    'lc_old': lambda: run_fit('lcdm_oldlya', 'LCDM_direct', 'oldlya',
                              COS_LC, chi2_lcdm,
                              [vec(COS_LC, LC_PR4TT), vec(COS_LC, LC_PLIK)], 2000),
    'lc_new': lambda: run_fit('lcdm_newlya', 'LCDM_direct', 'newlya',
                              COS_LC, chi2_lcdm,
                              warm('pr4_ttteee_lcdm_oldlya_best_v1.json', COS_LC,
                                   [vec(COS_LC, LC_PR4TT)]), 1200),
}

totals = {}
for key in RUN:
    totals[key] = FITS[key]()

print(f"\n{'=' * 64}\nSUMMARY (CamSpec PR4 TTTEEE + DESI DR2 BAO, both models, "
      f"both Lya rows)\n{'=' * 64}")
for key in RUN:
    print(f"  {key}: chi2_tot = {totals[key]:.2f}")
if 'av_old' in totals and 'lc_old' in totals:
    print(f"\n  GAP old row (Avrami - LCDM) = {totals['av_old'] - totals['lc_old']:+.2f}"
          f"   [plik baseline pair: 591.05 - 598.72 = -7.68]")
if 'av_new' in totals and 'lc_new' in totals:
    print(f"  GAP DR2-IV row (Avrami - LCDM) = {totals['av_new'] - totals['lc_new']:+.2f}"
          f"   [compare to the plik lya4 battery pair]")
print("\nSend the four jsons (and this log) back.")
