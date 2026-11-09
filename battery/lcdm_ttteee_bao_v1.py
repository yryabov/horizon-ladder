"""
Nucleation DE on plik_lite TT (215 bins, full covariance) — v1
===============================================================
Self-contained: reads the plik_lite data files directly from your cobaya
installation, computes chi2 with the full TT covariance, calls CAMB directly
with the nucleation PPF w(a) table and DERIVED H0.

SELF-VALIDATION: before fitting, evaluates the official Planck-2018 point and
REFUSES to run unless chi2 = 206.37 +/- 0.5 (your verified cobaya number).
This pins every file-format/binning convention.

MODE = 'sfree'          : free (omega_m, s, mnu, As, ns) + tau, A_planck priors   k=5
MODE = 'selfconsistent' : free (omega_m,    mnu, As, ns) + tau, A_planck priors   k=4
Thresholds vs LCDM reference chi2=203.13 (k=6):
  sfree beats AIC if chi2 <= 205.1;  s=h beats AIC if chi2 <= 207.1 (BIC: 208.5 / 213.9)
"""
import os, json
import numpy as np
import camb
from scipy.optimize import minimize, brentq

MODE = 'sfree'                       # 'sfree' | 'selfconsistent'
PACKAGES_PATH = r'C:\yaroslavFiles\cobaya_packages'
VALIDATION_CHI2 = 206.37             # your verified cobaya number at the official point

# ── locate and load plik_lite data ───────────────────────────────────────────
def find_file(name):
    for root, _, files in os.walk(PACKAGES_PATH):
        for f in files:
            if name in f:
                return os.path.join(root, f)
    raise FileNotFoundError(f"{name} not found under {PACKAGES_PATH}")

f_cl    = find_file('cl_cmb_plik_v22')
f_cov   = find_file('c_matrix_plik_v22')
f_blmin = find_file('blmin')
f_blmax = find_file('blmax')
f_blwin = find_file('bweight') if True else None
try:
    f_blwin = find_file('bweight')
except FileNotFoundError:
    f_blwin = find_file('bl_weight')
print("plik_lite files:")
for p in (f_cl, f_cov, f_blmin, f_blmax, f_blwin): print("  ", p)

NBINTT = 215; NBINTOT = 613          # TT 215 | TE 199 | EE 199
dat    = np.loadtxt(f_cl)
D_data = dat[:NBINTT, 1]             # binned D_ell, muK^2
blmin  = np.loadtxt(f_blmin).astype(int)
blmax  = np.loadtxt(f_blmax).astype(int)
blwin  = np.loadtxt(f_blwin)         # weights over ell=30..2508 (0-indexed at ell=30)

raw = None
for offset in (4, 0, 8):             # fortran record marker variants
    try:
        arr = np.fromfile(f_cov, dtype=np.float64, offset=offset, count=NBINTOT*NBINTOT)
        if arr.size != NBINTOT*NBINTOT:
            continue
        M = arr.reshape(NBINTOT, NBINTOT)
        # file stores the LOWER TRIANGLE (upper is garbage incl. NaN): symmetrize
        M = np.tril(M) + np.tril(M, -1).T
        d = np.diag(M)
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            continue
        try:
            np.linalg.cholesky(M[:NBINTT, :NBINTT])
        except np.linalg.LinAlgError:
            continue
        raw = M
        print(f"covariance read OK (offset={offset}, lower-triangle symmetrized)")
        break
    except Exception:
        pass
if raw is None:
    raise RuntimeError("covariance read failed for offsets 4/0/8")
COV_TT  = raw[:NBINTT, :NBINTT]
CINV_TT = np.linalg.inv(COV_TT)
LMIN = 30
if blmin[0] >= LMIN:                 # absolute indices -> make relative to ell=30
    blmin = blmin - LMIN; blmax = blmax - LMIN
ELLS = np.arange(LMIN, 2509)
print(f"data column range: [{D_data.min():.3e}, {D_data.max():.3e}]  "
      f"(~1e-3..1 => C_ell; ~1e2..6e3 => D_ell)")

_CONV = {}                           # filled by auto-calibration

def bin_theory(Dl_full):
    """Dl_full: D_ell from ell=0 (muK^2). Uses the auto-calibrated convention."""
    Dl = Dl_full[LMIN:2509]
    T  = Dl if _CONV['qty'] == 'D' else Dl*2*np.pi/(ELLS*(ELLS+1.0))
    out = np.empty(NBINTT)
    for i in range(NBINTT):
        sl = slice(blmin[i], blmax[i]+1)
        w  = blwin[sl]
        if _CONV['wnorm']: w = w/np.sum(w)
        out[i] = np.sum(w * T[sl])
    return out

ELL_EFF = np.array([np.average(ELLS[blmin[i]:blmax[i]+1], weights=blwin[blmin[i]:blmax[i]+1])
                    for i in range(NBINTT)])
EXCISE_ZONE = (-2.0, -1.0)   # BAO act: NO excision (baseline corner, full 215 bins)
KEEP = ~((ELL_EFF >= EXCISE_ZONE[0]) & (ELL_EFF <= EXCISE_ZONE[1]))
COV_FULL = np.linalg.inv(CINV_TT)
CINV_EX = np.linalg.inv(COV_FULL[np.ix_(KEEP, KEEP)])
print(f"EXCISED {np.sum(~KEEP)} bins, ell_eff = {np.round(ELL_EFF[~KEEP],1).tolist()}")
print(f"fit uses {np.sum(KEEP)} bins; validation uses all {NBINTT}")

def chi2_pliklite(Dl_full, A_planck=1.0, excise=True):
    X = bin_theory(Dl_full)/A_planck**2 - D_data
    if excise:
        Xk = X[KEEP]
        return float(Xk @ CINV_EX @ Xk)
    return float(X @ CINV_TT @ X)

# ── CAMB spectra ─────────────────────────────────────────────────────────────

# ── TTTEEE extension: full 613-bin vector (TT 215 | pol 199+199), full covariance ──
NBINTE = 199; NBINEE = 199
D_FULL    = dat[:NBINTOT, 1]
CINV_FULL = np.linalg.inv(raw)
_POL = {}
def _bin_spec(Dl_full, nb):
    Dl = Dl_full[LMIN:2509]
    T  = Dl if _CONV['qty'] == 'D' else Dl*2*np.pi/(ELLS*(ELLS+1.0))
    out = np.empty(nb)
    for i in range(nb):
        sl = slice(blmin[i], blmax[i]+1)
        w  = blwin[sl]
        if _CONV['wnorm']: w = w/np.sum(w)
        out[i] = np.sum(w * T[sl])
    return out
def chi2_ttteee(spec, A_planck=1.0):
    """spec: CAMB 'total' array (n x 4): [TT, EE, BB, TE], D_ell muK^2."""
    vTT = _bin_spec(spec[:,0], NBINTT)
    vTE = _bin_spec(spec[:,3], NBINTE)
    vEE = _bin_spec(spec[:,1], NBINEE)
    v = np.concatenate([vTT,vTE,vEE]) if _POL['order']=='TE_EE' else np.concatenate([vTT,vEE,vTE])
    X = v/A_planck**2 - D_FULL
    return float(X @ CINV_FULL @ X)

def camb_Dl_lcdm(H0, ombh2, omch2, mnu, As, ns, tau):
    p = camb.CAMBparams()
    p.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=max(0.001, mnu), omk=0, tau=tau)
    p.InitPower.set_params(As=As, ns=ns, r=0)
    p.set_for_lmax(2700, lens_potential_accuracy=1)
    p.WantTensors = False
    return camb.get_results(p).get_cmb_power_spectra(p, CMB_unit='muK')['total']

# ── SELF-VALIDATION with convention auto-calibration ─────────────────────────
Dl_off = camb_Dl_lcdm(67.36, 0.02237, 0.1200, 0.06, 2.100e-9, 0.9649, 0.0544)
print(f"\nAuto-calibrating binning convention (target {VALIDATION_CHI2} +/- 0.5):")
chosen = None
for qty in ('C','D'):
    for wnorm in (False, True):
        _CONV.update(qty=qty, wnorm=wnorm)
        c = chi2_pliklite(Dl_off[:,0], 1.0, excise=False)
        tag = f"qty={qty} wnorm={wnorm}"
        print(f"  {tag:24}: chi2 = {c:.2f}")
        if abs(c - VALIDATION_CHI2) < 0.5 and chosen is None:
            chosen = dict(qty=qty, wnorm=wnorm, chi2=c)
if chosen is None:
    raise SystemExit("VALIDATION FAILED for all conventions — send this log back.")
_CONV.update(qty=chosen['qty'], wnorm=chosen['wnorm'])
print(f"VALIDATION PASSED: convention qty={chosen['qty']} wnorm={chosen['wnorm']} "
      f"(chi2={chosen['chi2']:.2f})\n")


# ── TTTEEE polarization-order auto-calibration (wrong order fails catastrophically) ──
print("TTTEEE order auto-calibration (613-bin vector):")
_best_pol=None
for _order in ('TE_EE','EE_TE'):
    _POL['order']=_order
    _c=chi2_ttteee(Dl_off,1.0)
    print(f"  order {_order}: chi2_full = {_c:.2f}")
    if _best_pol is None or _c<_best_pol[1]: _best_pol=(_order,_c)
_POL['order']=_best_pol[0]
print(f"TTTEEE VALIDATION: order={_best_pol[0]}  chi2_full(official) = {_best_pol[1]:.2f}  [TT anchor 206.37 held above]")
assert 500.0<_best_pol[1]<780.0, "TTTEEE chi2 at official point out of sane range — send the log"

# ── DESI DR2 BAO (Table III, arXiv:2503.14738; verified values + rho_MH) ─────
BAO=[(0.295,7.942,0.075,None,None,None),
 (0.510,13.588,0.167,21.863,0.425,-0.459),(0.706,17.351,0.177,19.455,0.330,-0.404),
 (0.934,21.576,0.152,17.641,0.193,-0.416),(1.321,27.601,0.318,14.176,0.221,-0.434),
 (1.484,30.512,0.760,12.817,0.516,-0.500),(2.330,38.988,0.531,8.632,0.101,-0.431)]
def rd_desi(wcb,wb): return 147.05*(wcb/0.1432)**-0.23*(wb/0.02236)**-0.13
C_KM=299792.458
def chi2_bao_from_H(Hfn, rd):
    zf=np.linspace(0,2.6,600); invH=1.0/Hfn(zf)
    DMg=C_KM*np.concatenate([[0.0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])
    zq,val,err,_,_,_=BAO[0]
    dm=float(np.interp(zq,zf,DMg)); dh=C_KM/float(Hfn(np.array([zq]))[0])
    cB=(((zq*dm*dm*dh)**(1/3)/rd-val)/err)**2
    for zq,vM,eM,vH,eH,r in BAO[1:]:
        dm=float(np.interp(zq,zf,DMg))/rd; dh=(C_KM/float(Hfn(np.array([zq]))[0]))/rd
        d2=np.array([dm-vM,dh-vH]); C=np.array([[eM**2,r*eM*eH],[r*eM*eH,eH**2]])
        cB+=float(d2@np.linalg.inv(C)@d2)
    return cB


# ── LCDM fit: plik_lite (full vector) + DESI DR2 BAO ─────────────
MNU_NH_FLOOR=0.05819; MNU_NH_SIGMA=0.00029
TAU_MEAN=0.0605; TAU_SIGMA=0.0046

def chi2_full(H0,wb,wc,mnu,As,ns,tau,Ap):
    try:
        spec = camb_Dl_lcdm(H0,wb,wc,mnu,As,ns,tau)
        c_like = chi2_ttteee(spec,Ap)
        h2=(H0/100.0)**2; wm_tot=wb+wc+mnu/93.14
        Hfn=lambda z: 100.0*np.sqrt(wm_tot*(1+z)**3 + WRAD_L*(1+z)**4 + (h2-wm_tot-WRAD_L))
        cB=chi2_bao_from_H(Hfn, rd_desi(wb+wc, wb))
        c = c_like + cB + ((tau-TAU_MEAN)/TAU_SIGMA)**2 + ((Ap-1.0)/0.0025)**2
        c += 0.0 if mnu>=MNU_NH_FLOOR else ((mnu-MNU_NH_FLOOR)/MNU_NH_SIGMA)**2
        return c, c_like, cB
    except Exception:
        return 1e10, 1e10, 1e10
WRAD_L = 2.469e-5*(1+0.2271*3.046)

PNAMES=['H0','wb','wc','mnu','As','ns','tau','Ap']
BOUNDS={'H0':(60,76),'wb':(0.020,0.025),'wc':(0.10,0.14),'mnu':(0.0,0.5),
        'As':(1.5e-9,3.5e-9),'ns':(0.88,1.05),'tau':(0.02,0.12),'Ap':(0.98,1.02)}
starts=[[66.56,0.02210,0.12142,0.0582,2.136e-9,0.9599,0.0615,1.0003],   # CMB best
        [67.60,0.02230,0.11900,0.0582,2.120e-9,0.9650,0.0605,1.0000],   # BAO-leaning
        [67.97,0.02245,0.11780,0.0582,2.105e-9,0.9680,0.0605,1.0000]]

best=(1e10,None,None)
print(f"Running {len(starts)} starts...  MODEL=LCDM on excised vector")
for i,x0 in enumerate(starts):
    print(f"\n{'─'*55}\nStart {i+1}/{len(starts)}")
    cnt=[0]; loc=[1e10]
    def obj(v):
        p=[np.clip(x,*BOUNDS[n]) for x,n in zip(v,PNAMES)]
        c,cl,cb=chi2_full(*p); cnt[0]+=1
        if c<loc[0]:
            loc[0]=c
            print(f"  iter {cnt[0]:3d}: H0={p[0]:.2f} wb={p[1]:.5f} wc={p[2]:.4f} "
                  f"mnu={p[3]:.4f} ns={p[5]:.4f} tau={p[6]:.4f} CMB={cl:.1f} BAO={cb:.1f} tot={c:.2f} *")
        return c
    r=minimize(obj,x0,method='Nelder-Mead',options=dict(maxiter=1200,xatol=1e-4,fatol=0.05))
    p=[np.clip(x,*BOUNDS[n]) for x,n in zip(r.x,PNAMES)]
    c,cl,cb=chi2_full(*p)
    print(f"Start {i+1} done: tot={c:.2f} CMB={cl:.2f} BAO={cb:.2f}")
    if c<best[0]: best=(c,p,(cl,cb))

c_tot,p,(c_like,c_bao)=best
print(f"\n{'='*60}\nLCDM on plik_lite + DESI DR2 BAO (k=6+priors)\n{'='*60}")
print(f"  H0={p[0]:.3f} ombh2={p[1]:.5f} omch2={p[2]:.5f} mnu={p[3]:.5f}")
print(f"  As={p[4]:.4e} ns={p[5]:.4f} tau={p[6]:.4f} A_planck={p[7]:.4f}")
print(f"  chi2_CMB = {c_like:.3f}   chi2_BAO = {c_bao:.3f}   chi2_tot = {c_tot:.3f}")
json.dump(dict(MODEL='LCDM_pliklite_BAO',dataset='pliklite+DESI_DR2',chi2_BAO=c_bao,
               H0=p[0],ombh2=p[1],omch2=p[2],mnu=p[3],As=p[4],ns=p[5],tau=p[6],
               A_planck=p[7],chi2_like=c_like,chi2_tot=c_tot),
          open('lcdm_bao_ttteee_best_v1.json','w'),indent=2)
print("Saved: lcdm_pliklite_best_excise_v1.json")
