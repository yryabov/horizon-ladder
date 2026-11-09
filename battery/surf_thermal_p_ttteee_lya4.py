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

def _solve_proj(wm, Tc, pexp, hT, n_iter=300, relax=0.5):
    """Projected thermal solver: GC filling shape, amplitude projected onto H(a=1)=hT.
    Returns H, rho, alpha, s where s is the amplitude supplement (s=1 <=> exact
    self-consistency of rho = alpha/R_E^2; s!=1 quantifies the capacity correction)."""
    Ode=hT*hT-wm-WRAD
    H=np.sqrt(wm*_A**-3+WRAD*_A**-4+max(Ode,1e-8)); rho=None; s=1.0
    for _ in range(n_iter):
        f=1/(_A*H)
        cum=np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*_DX)[::-1],[0]])
        RE=_A*(cum+1/(_A[-1]*H[-1]))
        T=1.0/(2*np.pi*RE)
        alpha=np.clip(1.0-(T/Tc)**pexp, 0.0, 1.0)   # GC-p: free exponent (GC = 4)
        rho_raw=alpha/RE**2
        r0=float(np.interp(0.0,_XG,rho_raw))
        s=(hT*hT-wm-WRAD)/max(r0,1e-30)
        rho_new=s*rho_raw
        rho=rho_new if rho is None else (1-relax)*rho+relax*rho_new
        H=np.sqrt(wm*_A**-3+WRAD*_A**-4+np.maximum(rho,0))
    return H,rho,alpha,s

def surf_table(wm, Tc, pexp, hT, n_table=700):
    H,rho,al,s=_solve_proj(wm,Tc,pexp,hT)
    h=float(np.interp(1.0,_A,H))
    rho0v=float(np.interp(0.0,_XG,rho))
    w=-1-np.gradient(np.log(np.maximum(rho,abs(rho0v)*1e-12)),_XG)/3
    w=np.maximum(w,-60.0)
    born=al>1e-10
    zc=float(1/_A[np.argmax(born)]-1) if born.any() and not born.all() else np.nan
    zref=zc if np.isfinite(zc) else 3.0
    zlo=max(0.4,0.5*zref); zhi=min(30.0,3.5*(1+zref))
    lnat=np.unique(np.concatenate([np.linspace(-10.,0.,n_table),
                                   np.linspace(np.log(1/(1+zhi)),np.log(1/(1+zlo)),n_table)]))
    at=np.exp(lnat); wt=np.interp(lnat,_XG,w)
    al0=float(np.interp(0.0,_XG,al))
    return at,wt,h,zc,al0,s

MNU_NH_FLOOR=0.05819; MNU_NH_SIGMA=0.00029
TAU_MEAN=0.0605; TAU_SIGMA=0.0046

def chi2_full(wb, wc, Tc, pexp, hT, mnu, As, ns, tau, Ap):
    wm = wb + wc + mnu/93.14
    at,wt,h,zc,al0,s = surf_table(wm,Tc,pexp,hT)
    try:
        pN=camb.CAMBparams()
        pN.set_cosmology(H0=100*h,ombh2=wb,omch2=wc,mnu=max(0.001,mnu),omk=0,tau=tau)
        pN.InitPower.set_params(As=As,ns=ns,r=0)
        de=camb.dark_energy.DarkEnergyPPF(); de.set_w_a_table(at,wt); pN.DarkEnergy=de
        pN.set_for_lmax(2700,lens_potential_accuracy=1); pN.WantTensors=False
        spec=camb.get_results(pN).get_cmb_power_spectra(pN,CMB_unit='muK')['total']
        c_like=chi2_ttteee(spec,Ap)
        Hg,_,_,_=_solve_proj(wm,Tc,pexp,hT)
        Hfn=lambda z: 100.0*np.interp(np.log(1.0/(1.0+z)),_XG,Hg)
        cB=chi2_bao_from_H(Hfn, rd_desi(wb+wc, wb))
        c=c_like+(cB if USE_BAO else 0.0)+((tau-TAU_MEAN)/TAU_SIGMA)**2+((Ap-1.0)/0.0025)**2
        c+=0.0 if mnu>=MNU_NH_FLOOR else ((mnu-MNU_NH_FLOOR)/MNU_NH_SIGMA)**2
        return c,c_like,cB,h,zc,al0,s
    except Exception:
        return 1e10,1e10,1e10,h,0.0,0.0,1.0

USE_BAO = True     # True: plik_lite + DESI DR2 | False: CMB-only
BOUNDS={'wb':(0.020,0.025),'wc':(0.10,0.14),'Tc':(0.08,0.60),'pexp':(0.5,12.0),'h':(0.60,0.78),'mnu':(0.0,0.50),
        'As':(1.5e-9,3.5e-9),'ns':(0.88,1.05),'tau':(0.02,0.12),'Ap':(0.98,1.02)}
np.random.seed(42)
PNAMES=['wb','wc','Tc','pexp','h','mnu','As','ns','tau','Ap']
starts=[[0.02229,0.12133,0.1375,4.0,0.6795,0.0589,2.136e-9,0.9615,0.0614,1.000],  # thermal TTTEEE best, p=4 (GC)
        [0.02229,0.12133,0.16,2.0,0.6795,0.0589,2.136e-9,0.9615,0.0614,1.000],   # soft onset, earlier birth (BAO valley)
        [0.02229,0.12133,0.14,3.0,0.6795,0.0589,2.136e-9,0.9615,0.0614,1.000]]   # intermediate

def unpack(v):
    return tuple(np.clip(x,*BOUNDS[n]) for x,n in zip(v,PNAMES))

best=(1e10,None,None)
print(f"Running {len(starts)} starts...  SURFACE model, USE_BAO={USE_BAO}  (H0 derived, expect ~0.68)")
for i,x0 in enumerate(starts):
    print(f"\n{'─'*55}\nStart {i+1}/{len(starts)}")
    cnt=[0]; loc=[1e10]
    def obj(v):
        wb,wc,Tc,pexp,hT,mnu,As,ns,tau,Ap=unpack(v)
        c,cl,cb,h,zc,al0,s=chi2_full(wb,wc,Tc,pexp,hT,mnu,As,ns,tau,Ap)
        cnt[0]+=1
        if c<loc[0]:
            loc[0]=c
            print(f"  iter {cnt[0]:3d}: wb={wb:.5f} wc={wc:.4f} Tc={Tc:.4f} p={pexp:.2f} zc={zc:.2f} a0={al0:.3f} s={s:.3f} h={hT:.4f} "
                  f" mnu={mnu:.4f} ns={ns:.4f} "
                  f"tau={tau:.4f} CMB={cl:.1f} BAO={cb:.1f} tot={c:.2f} *")
        return c
    r=minimize(obj,x0,method='Nelder-Mead',options=dict(maxiter=1500,xatol=1e-4,fatol=0.05))
    wb,wc,Tc,pexp,hT,mnu,As,ns,tau,Ap=unpack(r.x)
    c,cl,cb,h,zc,al0,s=chi2_full(wb,wc,Tc,pexp,hT,mnu,As,ns,tau,Ap)
    print(f"Start {i+1} done: tot={c:.2f}  CMB={cl:.2f}  BAO={cb:.2f}")
    if c<best[0]: best=(c,(wb,wc,Tc,pexp,hT,mnu,As,ns,tau,Ap),(cl,cb,h,zc,al0,s))

c_tot,(wb,wc,Tc,pexp,hT,mnu,As,ns,tau,Ap),(c_like,c_bao,h,zc,al0,s)=best
k = len(PNAMES)-2
print(f"\n{'='*60}\nGLOBAL BEST — THERMAL SCREEN (Gorter-Casimir filling, k=7) ({'pliklite+BAO' if USE_BAO else 'pliklite CMB-only'}, k={k})\n{'='*60}")
print(f"  h = {hT:.4f}   T_c = {Tc:.4f}   p = {pexp:.3f} (GC=4)   z_c(DERIVED) = {zc:.3f}   alpha_0 = {al0:.3f}")
print(f"  amplitude supplement s = {s:.4f}   (s=1 <=> exact self-consistency; s-1 = capacity correction)")
print(f"  wb = {wb:.5f}   wc = {wc:.5f}   ratio = {wc/wb:.3f}")
print(f"  omega_m={wb+wc+mnu/93.14:.5f}  mnu={mnu:.5f}  As={As:.4e}  ns={ns:.4f}  tau={tau:.4f}  A_planck={Ap:.4f}")
print(f"  chi2_CMB = {c_like:.3f}   chi2_BAO = {c_bao:.3f}   chi2_tot = {c_tot:.3f}")
print(f"\nreferences: LCDM CMB-only 203.13 | LCDM CMB+BAO 218.25 | volume-nucl CMB+BAO 320.83 (all k=6-ish)")
json.dump(dict(MODEL='surface_thermal_GCp',USE_BAO=USE_BAO,
               dataset='pliklite'+('+DESI_DR2' if USE_BAO else ''),
               h_sampled=hT,ombh2=wb,omch2=wc,ratio=wc/wb,Tc_code=Tc,p_exponent=pexp,z_c_derived=zc,alpha0=al0,s_supplement=s,omega_m=wb+wc+mnu/93.14,mnu=mnu,As=As,ns=ns,tau=tau,
               A_planck=Ap,chi2_CMB=c_like,chi2_BAO=c_bao,chi2_tot=c_tot,nbin=215,k=k),
          open(f"surf_thermal_p_{'bao' if USE_BAO else 'cmb'}_ttteee_lya4_v1.json",'w'),indent=2)
print(f"Saved: surf_thermal_p_{'bao' if USE_BAO else 'cmb'}_ttteee_lya4_v1.json")
