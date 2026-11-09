"""
Fisher (finite-difference Hessian) uncertainties at the sn4 best fits
=====================================================================
Battery: plik_lite TTTEEE + DESI DR2 [BAO + Lya Results-IV joint row] + Pantheon+
with FULL STAT+SYS covariance — identical likelihood to the *_sn4.py scripts.

Usage:  python fisher_sn4_v1.py avrami        (one model; ~15-25 min)
        python fisher_sn4_v1.py l2 gcp        (several)
        python fisher_sn4_v1.py all           (all six; ~2 h)
Models: avrami | l2 | gcp | gc | cpl | lcdm
Output: fisher_<model>_sn4_v1.json  (sigmas, large correlations, drift check)
NOTE: Gaussian (symmetric) approximation at the minimum; the prior-pinned m_nu
is excluded (one-sided wall). Profile likelihoods supersede these for the paper
headline numbers (see profile_zc_avrami_sn4_v1.py).
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

def chi2_pliklite(Dl_full, A_planck=1.0):
    X = bin_theory(Dl_full)/A_planck**2 - D_data
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
        c = chi2_pliklite(Dl_off[:,0], 1.0)
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
 (1.484,30.512,0.760,12.817,0.516,-0.500),(2.330,39.32,0.33,8.600,0.066,-0.431)]  # Lya row: DR2 Results IV joint AP+BAO (2607.27410); rho kept -0.431 pending paper
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

# ── SN leg v4: Pantheon+ with FULL STAT+SYS covariance (offset-marginalized) ──
# NOTE: the printed/JSON "BAO" value = BAO + SN(full covariance, zHD>0.023,
# calibrators excluded, absolute offset marginalized analytically with C).
SN_PATH=r'C:\yaroslavFiles\sn\Pantheon+SH0ES.dat'
SN_COV_PATH=r'C:\yaroslavFiles\sn\Pantheon+SH0ES_STAT+SYS.cov'
_sn=np.genfromtxt(SN_PATH,names=True,dtype=None,encoding=None)
_msn=(_sn['IS_CALIBRATOR']==0)&(_sn['zHD']>=0.023)&(_sn['zHD']<=2.5)
_snz=np.asarray(_sn['zHD'][_msn],float); _snmu=np.asarray(_sn['MU_SH0ES'][_msn],float)
with open(SN_COV_PATH) as _f:
    _ncov=int(_f.readline().split()[0])
    _cv=np.fromfile(_f,sep=' ',count=_ncov*_ncov)
assert _cv.size==_ncov*_ncov, f"cov read {_cv.size} != {_ncov}^2"
assert _ncov==len(_sn), f"cov N={_ncov} vs dat rows {len(_sn)} — row orders must match"
_idx=np.where(_msn)[0]
_Csn=_cv.reshape(_ncov,_ncov)[np.ix_(_idx,_idx)]
_Cinv_sn=np.linalg.inv(_Csn)
_ones=np.ones(len(_idx)); _cs=_Cinv_sn@_ones; _cd=float(_ones@_cs)
print(f"SN STAT+SYS covariance: {_ncov}x{_ncov} -> subset {len(_idx)}x{len(_idx)}, inverted OK")
def chi2_sn_from_H(Hfn):
    zf=np.linspace(0.0,2.55,1200); invH=1.0/Hfn(zf)
    DC=C_KM*np.concatenate([[0.0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])
    mu_th=5.0*np.log10(np.maximum((1.0+_snz)*np.interp(_snz,zf,DC),1e-9))+25.0
    r=_snmu-mu_th
    b=float(_cs@r)
    return float(r@_Cinv_sn@r)-b*b/_cd
_chi2_bao_only=chi2_bao_from_H
def chi2_bao_from_H(Hfn, rd):
    return _chi2_bao_only(Hfn, rd)+chi2_sn_from_H(Hfn)

# ══════════════════════════════════════════════════════════════════════════════
# FISHER MACHINERY: solvers for all six sn4 models + finite-difference Hessian
# ══════════════════════════════════════════════════════════════════════════════
import sys
WG = 2.469e-5; WRAD = WG*(1 + 0.2271*3.046)
_XG = np.linspace(np.log(1/2500), 13.0, 3200)
_A  = np.exp(_XG); _DX = _XG[1]-_XG[0]
MNU_NH_FLOOR=0.05819; MNU_NH_SIGMA=0.00029
TAU_MEAN=0.0605; TAU_SIGMA=0.0046

def _RE_of(H):
    f=1/(_A*H)
    cum=np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*_DX)[::-1],[0]])
    return _A*(cum+1/(_A[-1]*H[-1]))

def _wtab(rho, zref):
    rho0=float(np.interp(0.0,_XG,rho))
    w=-1-np.gradient(np.log(np.maximum(rho,abs(rho0)*1e-12)),_XG)/3
    w=np.maximum(w,-60.0)
    zlo=max(0.4,0.5*zref); zhi=min(30.0,3.5*(1+zref))
    lnat=np.unique(np.concatenate([np.linspace(-10.,0.,600),
                                   np.linspace(np.log(1/(1+zhi)),np.log(1/(1+zlo)),800)]))
    return np.exp(lnat), np.interp(lnat,_XG,w)

def _solve_avrami(wm,lam,zc,hT):
    xc=np.log(1.0/(1.0+zc)); t=np.maximum(_XG-xc,0.0)/lam
    alpha=1.0-np.exp(-t**2)
    lo,hi=0.02,2.4; H=rho=None
    for _ in range(44):
        mid=0.5*(lo+hi); rho=np.full_like(_XG,mid)
        for _ in range(220):
            H=np.sqrt(wm*_A**-3+WRAD*_A**-4+np.maximum(rho,0))
            rho=0.55*rho+0.45*alpha/_RE_of(H)**2
        H=np.sqrt(wm*_A**-3+WRAD*_A**-4+rho)
        h=float(np.interp(1.0,_A,H))
        if abs(h-hT)<4e-4: break
        if h<hT: lo=mid
        else: hi=mid
    return H,rho

def _solve_L2(wm,u,zc,hT):
    xc=np.log(1.0/(1.0+zc)); born=(_XG>xc)
    lo,hi=0.02,2.4; H=rho=None
    for _ in range(44):
        mid=0.5*(lo+hi); rho=np.full_like(_XG,mid)
        for _ in range(220):
            H=np.sqrt(wm*_A**-3+WRAD*_A**-4+np.maximum(rho,0))
            RE=_RE_of(H)
            g=np.where(born,1.0/(H*RE),0.0)
            I=np.concatenate([[0.0],np.cumsum((g[:-1]+g[1:])/2*_DX)])
            alpha=np.where(born,1.0-np.exp(-np.pi*(u*I)**2),0.0)
            rho=0.55*rho+0.45*alpha/RE**2
        H=np.sqrt(wm*_A**-3+WRAD*_A**-4+rho)
        h=float(np.interp(1.0,_A,H))
        if abs(h-hT)<4e-4: break
        if h<hT: lo=mid
        else: hi=mid
    return H,rho

def _solve_gc(wm,Tc,pexp,hT):
    Ode=hT*hT-wm-WRAD
    H=np.sqrt(wm*_A**-3+WRAD*_A**-4+max(Ode,1e-8)); rho=None
    for _ in range(300):
        RE=_RE_of(H); T=1.0/(2*np.pi*RE)
        alpha=np.clip(1.0-(T/Tc)**pexp,0.0,1.0)
        raw=alpha/RE**2
        s=(hT*hT-wm-WRAD)/max(float(np.interp(0.0,_XG,raw)),1e-30)
        new=s*raw
        rho=new if rho is None else 0.5*rho+0.5*new
        H=np.sqrt(wm*_A**-3+WRAD*_A**-4+np.maximum(rho,0))
    born=alpha>1e-10
    zc=float(1/_A[np.argmax(born)]-1) if born.any() and not born.all() else 3.0
    return H,rho,zc

def _priors(mnu,tau,Ap):
    c=((tau-TAU_MEAN)/TAU_SIGMA)**2+((Ap-1.0)/0.0025)**2
    if mnu<MNU_NH_FLOOR: c+=((mnu-MNU_NH_FLOOR)/MNU_NH_SIGMA)**2
    return c

def _cmb(h,wb,wc,mnu,As,ns,tau,Ap,at=None,wt=None,w0wa=None):
    p=camb.CAMBparams()
    p.set_cosmology(H0=100*h,ombh2=wb,omch2=wc,mnu=max(0.001,mnu),omk=0,tau=tau)
    p.InitPower.set_params(As=As,ns=ns,r=0)
    if w0wa is not None:
        de=camb.dark_energy.DarkEnergyPPF(); de.set_params(w=w0wa[0],wa=w0wa[1]); p.DarkEnergy=de
    elif at is not None:
        de=camb.dark_energy.DarkEnergyPPF(); de.set_w_a_table(at,wt); p.DarkEnergy=de
    p.set_for_lmax(2700,lens_potential_accuracy=1); p.WantTensors=False
    spec=camb.get_results(p).get_cmb_power_spectra(p,CMB_unit='muK')['total']
    return chi2_ttteee(spec,Ap)

def chi2_avrami(x):
    wb,wc,lam,zc,h,mnu,As,ns,tau,Ap=x
    wm=wb+wc+mnu/93.14
    H,rho=_solve_avrami(wm,lam,zc,h)
    at,wt=_wtab(rho,zc)
    cl=_cmb(h,wb,wc,mnu,As,ns,tau,Ap,at=at,wt=wt)
    Hfn=lambda z: 100.0*np.interp(np.log(1.0/(1.0+np.asarray(z,float))),_XG,H)
    return cl+chi2_bao_from_H(Hfn,rd_desi(wb+wc,wb))+_priors(mnu,tau,Ap)

def chi2_L2(x):
    wb,wc,u,zc,h,mnu,As,ns,tau,Ap=x
    wm=wb+wc+mnu/93.14
    H,rho=_solve_L2(wm,u,zc,h)
    at,wt=_wtab(rho,zc)
    cl=_cmb(h,wb,wc,mnu,As,ns,tau,Ap,at=at,wt=wt)
    Hfn=lambda z: 100.0*np.interp(np.log(1.0/(1.0+np.asarray(z,float))),_XG,H)
    return cl+chi2_bao_from_H(Hfn,rd_desi(wb+wc,wb))+_priors(mnu,tau,Ap)

def _chi2_gc_generic(x,pfixed=None):
    if pfixed is None:
        wb,wc,Tc,pexp,h,mnu,As,ns,tau,Ap=x
    else:
        wb,wc,Tc,h,mnu,As,ns,tau,Ap=x; pexp=pfixed
    wm=wb+wc+mnu/93.14
    H,rho,zc=_solve_gc(wm,Tc,pexp,h)
    at,wt=_wtab(rho,zc)
    cl=_cmb(h,wb,wc,mnu,As,ns,tau,Ap,at=at,wt=wt)
    Hfn=lambda z: 100.0*np.interp(np.log(1.0/(1.0+np.asarray(z,float))),_XG,H)
    return cl+chi2_bao_from_H(Hfn,rd_desi(wb+wc,wb))+_priors(mnu,tau,Ap)

def chi2_gcp(x): return _chi2_gc_generic(x,None)
def chi2_gc(x):  return _chi2_gc_generic(x,4.0)

def chi2_cpl(x):
    wb,wc,w0,wa,h,mnu,As,ns,tau,Ap=x
    wm=wb+wc+mnu/93.14; Ode=h*h-wm-WRAD
    cl=_cmb(h,wb,wc,mnu,As,ns,tau,Ap,w0wa=(w0,wa))
    def Hfn(z):
        a=1.0/(1.0+np.asarray(z,float))
        return 100.0*np.sqrt(wm*a**-3+WRAD*a**-4+Ode*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a)))
    return cl+chi2_bao_from_H(Hfn,rd_desi(wb+wc,wb))+_priors(mnu,tau,Ap)

def chi2_lcdm(x):
    wb,wc,H0,mnu,As,ns,tau,Ap=x
    h=H0/100.0; wm=wb+wc+mnu/93.14; Ode=h*h-wm-WRAD
    cl=_cmb(h,wb,wc,mnu,As,ns,tau,Ap)
    def Hfn(z):
        a=1.0/(1.0+np.asarray(z,float))
        return 100.0*np.sqrt(wm*a**-3+WRAD*a**-4+Ode)
    return cl+chi2_bao_from_H(Hfn,rd_desi(wb+wc,wb))+_priors(mnu,tau,Ap)

# ── registry: sn4 best fits (from the six *_sn4_v1.json verdicts) ────────────
MODELS={
 'avrami': dict(fn=chi2_avrami, chi2_ref=1803.482,
   pnames=['wb','wc','lam','zc','h','mnu','As','ns','tau','Ap'],
   best=[0.0223960224,0.1200090090,0.9026216626,4.1125308865,0.6790651632,
         0.0590788467,2.1310710093e-09,0.9645963401,0.0615980241,1.0004939549],
   steps=[2e-4,8e-4,0.02,0.06,2.5e-3,2.5e-3,1.3e-11,1.5e-3,1.5e-3,6e-4]),
 'l2': dict(fn=chi2_L2, chi2_ref=1806.729,
   pnames=['wb','wc','u','zc','h','mnu','As','ns','tau','Ap'],
   best=[0.0223515227,0.1207079204,1.6201711256,4.0552933435,0.6702921905,
         0.0607664627,2.1364423247e-09,0.9628880061,0.0617388997,1.0006900791],
   steps=[2e-4,8e-4,0.03,0.06,2.5e-3,2.5e-3,1.3e-11,1.5e-3,1.5e-3,6e-4]),
 'gcp': dict(fn=chi2_gcp, chi2_ref=1806.401,
   pnames=['wb','wc','Tc','p','h','mnu','As','ns','tau','Ap'],
   best=[0.0224344165,0.1194333926,0.1560156799,1.9955939580,0.6750807592,
         0.0607179961,2.1351144483e-09,0.9661030869,0.0631873167,1.0005133110],
   steps=[2e-4,8e-4,1.5e-3,0.06,2.5e-3,2.5e-3,1.3e-11,1.5e-3,1.5e-3,6e-4]),
 'gc': dict(fn=chi2_gc, chi2_ref=1812.252,
   pnames=['wb','wc','Tc','h','mnu','As','ns','tau','Ap'],
   best=[0.0222639081,0.1218333637,0.1361112799,0.6806963877,
         0.0586124892,2.1315872001e-09,0.9603740954,0.0600703588,0.9999743088],
   steps=[2e-4,8e-4,1.2e-3,2.5e-3,2.5e-3,1.3e-11,1.5e-3,1.5e-3,6e-4]),
 'cpl': dict(fn=chi2_cpl, chi2_ref=1805.149,
   pnames=['wb','wc','w0','wa','h','mnu','As','ns','tau','Ap'],
   best=[0.0224409401,0.1192702865,-0.8725307554,-0.4881683839,0.6773283851,
         0.0583578456,2.1335946458e-09,0.9662284897,0.0629903120,1.0005598714],
   steps=[2e-4,8e-4,0.02,0.07,2.5e-3,2.5e-3,1.3e-11,1.5e-3,1.5e-3,6e-4]),
 'lcdm': dict(fn=chi2_lcdm, chi2_ref=1811.892,
   pnames=['wb','wc','H0','mnu','As','ns','tau','Ap'],
   best=[0.0225181049,0.1180185449,68.2652369651,
         0.0581851433,2.1279406488e-09,0.9693653264,0.0630206257,1.0007585580],
   steps=[2e-4,8e-4,0.18,2.5e-3,1.3e-11,1.5e-3,1.5e-3,6e-4]),
}

def fisher(name):
    m=MODELS[name]; fn=m['fn']
    x0=np.array(m['best'],float); st=np.array(m['steps'],float); names=list(m['pnames'])
    print(f"\n{'='*70}\nFISHER: {name}  (k_full={len(x0)})\n{'='*70}")
    f0=fn(x0)
    drift=f0-m['chi2_ref']
    print(f"chi2 at stored best: {f0:.3f}  (stored {m['chi2_ref']:.3f}; drift {drift:+.3f})")
    if abs(drift)>2.0:
        print("WARNING: drift > 2 — re-optimized best or environment mismatch; sigmas still valid locally.")
    # exclude prior-pinned mnu (one-sided wall breaks the quadratic approximation)
    act=[i for i,nm in enumerate(names) if not (nm=='mnu' and x0[i]-MNU_NH_FLOOR<1.2*st[i])]
    excl=[names[i] for i in range(len(names)) if i not in act]
    if excl: print(f"excluded (prior-pinned): {excl}  [sigma ~ wall width {MNU_NH_SIGMA}]")
    k=len(act)
    n_eval=2*k+2*k*(k-1)
    print(f"active k={k}; Hessian evals ~{n_eval} (plus center) — expect ~{n_eval*4/60:.0f}-{n_eval*7/60:.0f} min")
    fp=np.zeros(k); fm=np.zeros(k)
    H=np.zeros((k,k))
    for a_ in range(k):
        i=act[a_]; e=np.zeros_like(x0); e[i]=st[i]
        fp[a_]=fn(x0+e); fm[a_]=fn(x0-e)
        H[a_,a_]=(fp[a_]-2*f0+fm[a_])/st[i]**2
        print(f"  diag {names[i]:>4}: d2chi2={H[a_,a_]:.4g}   (f+ {fp[a_]:.2f} | f- {fm[a_]:.2f})")
    for a_ in range(k):
        for b_ in range(a_+1,k):
            i,j=act[a_],act[b_]
            ei=np.zeros_like(x0); ei[i]=st[i]
            ej=np.zeros_like(x0); ej[j]=st[j]
            fpp=fn(x0+ei+ej); fpm=fn(x0+ei-ej); fmp=fn(x0-ei+ej); fmm=fn(x0-ei-ej)
            H[a_,b_]=H[b_,a_]=(fpp-fpm-fmp+fmm)/(4*st[i]*st[j])
        print(f"  off-diag row {names[act[a_]]} done")
    ev=np.linalg.eigvalsh(H)
    print(f"Hessian eigenvalues: min={ev.min():.3g} max={ev.max():.3g}")
    if ev.min()<=0:
        print("WARNING: Hessian not positive definite (optimizer not fully converged or FD noise);")
        print("         clipping negative eigenvalues to smallest positive * 1e-2.")
        w_,V=np.linalg.eigh(H)
        w_=np.clip(w_,w_[w_>0].min()*1e-2,None)
        H=V@np.diag(w_)@V.T
    C=2.0*np.linalg.inv(H)          # chi2 = -2 lnL  =>  Cov = 2 H^-1
    sig=np.sqrt(np.diag(C))
    corr=C/np.outer(sig,sig)
    print(f"\n{'param':>6} {'best':>14} {'sigma (Fisher)':>14}")
    out={}
    for a_,i in enumerate(act):
        print(f"{names[i]:>6} {x0[i]:14.6g} {sig[a_]:14.3g}")
        out[names[i]]=dict(best=float(x0[i]),sigma=float(sig[a_]))
    json.dump(dict(MODEL=name,battery='sn4 (TTTEEE + BAO/LyaIV + SN full cov)',
                   chi2_center=float(f0),chi2_ref=m['chi2_ref'],excluded=excl,
                   params=out,
                   corr={f"{names[act[a_]]}|{names[act[b_]]}":float(corr[a_,b_])
                         for a_ in range(k) for b_ in range(a_+1,k) if abs(corr[a_,b_])>0.25}),
              open(f"fisher_{name}_sn4_v1.json","w"),indent=1)
    print(f"Saved: fisher_{name}_sn4_v1.json")

if __name__=='__main__':
    args=[a.lower() for a in sys.argv[1:]] or ['avrami']
    todo=list(MODELS) if args==['all'] else args
    for nm in todo:
        assert nm in MODELS, f"unknown model {nm}; choose from {list(MODELS)} or 'all'"
        fisher(nm)
