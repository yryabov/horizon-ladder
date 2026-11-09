"""
PROFILE LIKELIHOOD over z_c — Avrami on the sn4 battery
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

def chi2_full(wb, wc, lam, zc, hT, mnu, As, ns, tau, Ap):
    wm = wb + wc + mnu/93.14          # UNLOCKED: no 5:1 chain; densities sampled directly
    at,wt,h = surf_table(wm,lam,zc,hT)
    try:
        pN=camb.CAMBparams()
        pN.set_cosmology(H0=100*h,ombh2=wb,omch2=wc,mnu=max(0.001,mnu),omk=0,tau=tau)
        pN.InitPower.set_params(As=As,ns=ns,r=0)
        de=camb.dark_energy.DarkEnergyPPF(); de.set_w_a_table(at,wt); pN.DarkEnergy=de
        pN.set_for_lmax(2700,lens_potential_accuracy=1); pN.WantTensors=False
        spec=camb.get_results(pN).get_cmb_power_spectra(pN,CMB_unit='muK')['total']
        c_like=chi2_ttteee(spec,Ap)
        Hg,_=_solve_target(wm,lam,zc,hT)
        Hfn=lambda z: 100.0*np.interp(np.log(1.0/(1.0+z)),_XG,Hg)
        cB=chi2_bao_from_H(Hfn, rd_desi(wb+wc, wb))
        c=c_like+(cB if USE_BAO else 0.0)+((tau-TAU_MEAN)/TAU_SIGMA)**2+((Ap-1.0)/0.0025)**2
        c+=0.0 if mnu>=MNU_NH_FLOOR else ((mnu-MNU_NH_FLOOR)/MNU_NH_SIGMA)**2
        return c,c_like,cB,h,0.0
    except Exception:
        return 1e10,1e10,1e10,h,0.0

USE_BAO = True

# ══════════════════════════════════════════════════════════════════════════════
# PROFILE LIKELIHOOD over z_c — Avrami leader, sn4 battery
# At each fixed z_c on the grid, re-minimize the remaining 9 parameters.
# Warm starts proceed outward from the best-fit z_c. Expect ~25-40 min/point.
# Output: profile_zc_avrami_sn4_v1.json  (grid, chi2, per-point params, interval)
# ══════════════════════════════════════════════════════════════════════════════
BOUNDS={'wb':(0.020,0.025),'wc':(0.10,0.14),'lam':(0.05,3.0),'h':(0.60,0.78),
        'mnu':(0.0,0.50),'As':(1.5e-9,3.5e-9),'ns':(0.88,1.05),'tau':(0.02,0.12),'Ap':(0.98,1.02)}
P9=['wb','wc','lam','h','mnu','As','ns','tau','Ap']
BEST9=[0.0223960224,0.1200090090,0.9026216626,0.6790651632,
       0.0590788467,2.1310710093e-09,0.9645963401,0.0615980241,1.0004939549]
ZC_BEST=4.1125308865
ZC_GRID=[round(z,2) for z in np.arange(3.6,4.81,0.1)]

def unpack9(v):
    return tuple(np.clip(x,*BOUNDS[n]) for x,n in zip(v,P9))

order=sorted(ZC_GRID,key=lambda z:abs(z-ZC_BEST))
done={}   # zc -> (chi2,params9)
print(f"Profile grid: {ZC_GRID}  ({len(ZC_GRID)} points, warm-started outward from {ZC_BEST:.2f})")
for iz,zc in enumerate(order):
    if done:
        znear=min(done,key=lambda z:abs(z-zc)); x0=list(done[znear][1])
    else:
        x0=list(BEST9)
    cnt=[0]; loc=[1e18]
    def obj(v):
        wb,wc,lam,hT,mnu,As,ns,tau,Ap=unpack9(v)
        c,cl,cb,h,_=chi2_full(wb,wc,lam,zc,hT,mnu,As,ns,tau,Ap)
        cnt[0]+=1
        if c<loc[0]:
            loc[0]=c
            print(f"  zc={zc:4.2f} iter {cnt[0]:3d}: chi2={c:9.3f}  (CMB {cl:.1f} | BAO+SN {cb:.1f})")
        return c
    r=minimize(obj,x0,method='Nelder-Mead',
               options=dict(maxiter=420,xatol=1e-4,fatol=0.04))
    p9=unpack9(r.x)
    c,cl,cb,h,_=chi2_full(p9[0],p9[1],p9[2],zc,p9[3],p9[4],p9[5],p9[6],p9[7],p9[8])
    done[zc]=(c,p9)
    print(f"[{iz+1}/{len(order)}] zc={zc:4.2f}: chi2_min = {c:.3f}  ({cnt[0]} evals)")

zs=sorted(done); cs=[done[z][0] for z in zs]
cmin=min(cs); zmin=zs[int(np.argmin(cs))]
print(f"\n{'='*64}\nPROFILE chi2(z_c) — Avrami, sn4 battery\n{'='*64}")
for z,c in zip(zs,cs):
    bar='#'*int(min(40,(c-cmin)*8))
    print(f"  zc={z:4.2f}: chi2={c:9.3f}  d={c-cmin:6.3f} {bar}")
def crossings(delta):
    out=[]
    for z1,z2,c1,c2 in zip(zs[:-1],zs[1:],cs[:-1],cs[1:]):
        if (c1-cmin-delta)*(c2-cmin-delta)<0:
            out.append(z1+(z2-z1)*(delta-(c1-cmin))/(c2-c1))
    return out
cr1=crossings(1.0); cr4=crossings(4.0)
print(f"\nminimum: zc = {zmin:.2f} (chi2 {cmin:.3f})")
print(f"dchi2=1 crossings (68%): {['%.3f'%x for x in cr1]}")
print(f"dchi2=4 crossings (95%): {['%.3f'%x for x in cr4]}")
if len(cr1)>=2:
    print(f"==> z_c = {zmin:.2f}  +{cr1[-1]-zmin:.2f} / -{zmin-cr1[0]:.2f}  (68% profile)")
json.dump(dict(MODEL='avrami_profile_zc',battery='sn4',zc_grid=zs,chi2=cs,
               zc_min=float(zmin),chi2_min=float(cmin),
               cross68=cr1,cross95=cr4,
               params={str(z):dict(zip(P9,map(float,done[z][1]))) for z in zs}),
          open("profile_zc_avrami_sn4_v1.json","w"),indent=1)
print("Saved: profile_zc_avrami_sn4_v1.json")
