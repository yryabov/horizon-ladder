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

# ── nucleation solver (identical to the binned script) ───────────────────────
WG = 2.469e-5; WRAD = WG*(1 + 0.2271*3.046)
EPS_FRAC = 0.01
_XG = np.linspace(np.log(1/2500), 13.0, 3200)
_A  = np.exp(_XG); _DX = _XG[1]-_XG[0]

def _solve(wm, lam, zc, rho0=0.30, n_iter=220, relax=0.45):
    # Avrami n=2: instantaneous nucleation at criticality + 2d domain growth
    xc=np.log(1.0/(1.0+zc))
    t=np.maximum(_XG-xc,0.0)/lam
    alpha=1.0-np.exp(-t**2)
    a=_A; rho=np.full_like(_XG,rho0)
    for _ in range(n_iter):
        H=np.sqrt(wm*a**-3+WRAD*a**-4+np.maximum(rho,0))
        f=1/(a*H)
        cum=np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*_DX)[::-1],[0]])
        RE=a*(cum+1/(a[-1]*H[-1]))
        rho=(1-relax)*rho+relax*alpha/RE**2
    H=np.sqrt(wm*a**-3+WRAD*a**-4+rho)
    return H,rho

def _solve_target(wm, lam, zc, hT):
    lo,hi=0.02,2.4; H=rho=None
    for _ in range(44):
        mid=0.5*(lo+hi)
        H,rho=_solve(wm,lam,zc,mid)
        h=float(np.interp(1.0,_A,H))
        if abs(h-hT)<4e-4: break
        if h<hT: lo=mid
        else: hi=mid
    return H,rho

def surf_table(wm, lam, zc, hT, n_table=600):
    H,rho=_solve_target(wm,lam,zc,hT)
    h=float(np.interp(1.0,_A,H))
    rho0=float(np.interp(0.0,_XG,rho))
    rho_f=np.maximum(rho,rho0*1e-12)
    w=-1-np.gradient(np.log(rho_f),_XG)/3
    w=np.maximum(w,-60.0)
    zlo=max(0.4,0.5*zc); zhi=min(30.0,3.5*(1+zc))
    lnat=np.unique(np.concatenate([np.linspace(-10.,0.,n_table),
                                   np.linspace(np.log(1/(1+zhi)),np.log(1/(1+zlo)),4*n_table//3)]))
    at=np.exp(lnat); wt=np.interp(lnat,_XG,w)
    return at,wt,h

# ── 5:1 + eq(4) + priors (as always) ─────────────────────────────────────────
MNU_NH_FLOOR=0.05819; MNU_NH_SIGMA=0.00029
TAU_MEAN=0.0605; TAU_SIGMA=0.0046


# ── SN leg (Pantheon+; folded into the BAO number via wrapper) ──────────────
# NOTE: after this patch the printed/JSON "BAO" value = BAO + SN(Pantheon+,
# offset-marginalized, diagonal errors, zHD>0.023, calibrators excluded).
SN_PATH=r'C:\yaroslavFiles\sn\Pantheon+SH0ES.dat'
_sn=np.genfromtxt(SN_PATH,names=True,dtype=None,encoding=None)
_msn=(_sn['IS_CALIBRATOR']==0)&(_sn['zHD']>=0.023)&(_sn['zHD']<=2.5)
_snz=np.asarray(_sn['zHD'][_msn],float); _snmu=np.asarray(_sn['MU_SH0ES'][_msn],float)
_snw=1.0/np.asarray(_sn['MU_SH0ES_ERR_DIAG'][_msn],float)**2
def chi2_sn_from_H(Hfn):
    zf=np.linspace(0.0,2.55,1200); invH=1.0/Hfn(zf)
    DC=C_KM*np.concatenate([[0.0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])
    mu_th=5.0*np.log10(np.maximum((1.0+_snz)*np.interp(_snz,zf,DC),1e-9))+25.0
    r=_snmu-mu_th; off=np.sum(_snw*r)/np.sum(_snw)
    return float(np.sum(_snw*(r-off)**2))
_chi2_bao_only=chi2_bao_from_H
def chi2_bao_from_H(Hfn, rd):
    return _chi2_bao_only(Hfn, rd)+chi2_sn_from_H(Hfn)

def chi2_full(wb, wc, mnu, As, ns, tau, Ap):
    """Matter-only control (NO dark energy): flat => h derived from densities."""
    wm = wb + wc + mnu/93.14
    h = np.sqrt(wm + WRAD)
    try:
        pars=camb.CAMBparams()
        pars.set_cosmology(H0=100.0*h, ombh2=wb, omch2=wc, mnu=mnu, tau=tau)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_for_lmax(2700, lens_potential_accuracy=1)
        spec=camb.get_results(pars).get_cmb_power_spectra(pars,CMB_unit='muK')['total']
        cl=chi2_ttteee(spec,Ap)
        Hfn=lambda z: 100.0*h*np.sqrt(wm*(1.0+np.asarray(z,float))**3+WRAD*(1.0+np.asarray(z,float))**4)/np.sqrt(wm+WRAD)
        cB=chi2_bao_from_H(Hfn, rd_desi(wb+wc, wb))   # includes the SN leg via wrapper
        pr=((tau-0.0605)/0.0046)**2+((Ap-1.0)/0.0025)**2
        return cl+cB+pr, cl, cB, h
    except Exception:
        return 1e10,1e10,1e10,h

PNAMES=['wb','wc','mnu','As','ns','tau','Ap']
BOUNDS={'wb':(0.018,0.026),'wc':(0.08,0.30),'mnu':(0.05819,0.4),
        'As':(1.6e-9,2.8e-9),'ns':(0.90,1.05),'tau':(0.04,0.09),'Ap':(0.99,1.01)}
starts=[[0.02240,0.1180,0.0590,2.10e-9,0.9650,0.0605,1.000],
        [0.02100,0.2000,0.0600,2.00e-9,0.9800,0.0605,1.000],
        [0.02240,0.1400,0.0600,2.20e-9,0.9500,0.0605,1.000]]
def unpack(v):
    out=[]
    for x,nm in zip(v,PNAMES):
        lo,hi=BOUNDS[nm]; out.append(min(max(x,lo),hi))
    return out
from scipy.optimize import minimize
np.random.seed(11)
print(f"Running {len(starts)} starts...  NO-DARK-ENERGY control (flat matter+radiation; h DERIVED)")
best=(1e18,None,None)
it=[0]
for i0,s0 in enumerate(starts):
    print(f"Start {i0+1}/{len(starts)}")
    def obj(v):
        wb,wc,mnu,As,ns,tau,Ap=unpack(v)
        c,cl,cb,h=chi2_full(wb,wc,mnu,As,ns,tau,Ap)
        it[0]+=1
        if c<best[0]:
            print(f"  iter {it[0]:3d}: wb={wb:.5f} wc={wc:.4f} h(derived)={h:.4f} mnu={mnu:.4f} ns={ns:.4f} CMB={cl:.1f} BAO+SN={cb:.1f} tot={c:.2f} *")
        return c
    r=minimize(obj,s0,method='Nelder-Mead',
               options={'maxiter':500,'xatol':1e-5,'fatol':1e-3,'adaptive':True})
    wb,wc,mnu,As,ns,tau,Ap=unpack(r.x)
    c,cl,cb,h=chi2_full(wb,wc,mnu,As,ns,tau,Ap)
    if c<best[0]: best=(c,(wb,wc,mnu,As,ns,tau,Ap),(cl,cb,h))
c_tot,(wb,wc,mnu,As,ns,tau,Ap),(c_like,c_bao,h)=best
print("="*78)
print("GLOBAL BEST — NO-DE CONTROL (Omega_m=1 flat; k=5)")
print(f"  chi2_tot = {c_tot:.2f}   (CMB {c_like:.2f} | BAO+SN {c_bao:.2f})")
print(f"  wb={wb:.5f} wc={wc:.5f} mnu={mnu:.4f} As={As:.4e} ns={ns:.4f} tau={tau:.4f}")
print(f"  h (DERIVED from flatness) = {h:.4f}")
import json
json.dump(dict(MODEL='no_dark_energy_flat',chi2_tot=c_tot,chi2_cmb=c_like,chi2_bao_sn=c_bao,
    wb=wb,wc=wc,mnu=mnu,As=As,ns=ns,tau=tau,Ap=Ap,h_derived=h,k=5),
    open("noDE_bao_ttteee_sn3_v1.json","w"),indent=1)
print('Saved: noDE_bao_ttteee_sn3_v1.json')
