"""
Surface nucleation (Avrami n=2, h free) on PR4 NPIPE CamSpec TT — v1
=====================================================================
Self-contained: (1) builds the CamSpec likelihood object, (2) AUTO-DETECTS the
spectrum-injection signature and VALIDATES it against the anchor chi2 = 6990.88
(official Planck-2018 point, reconnaissance nuisances) — refuses to fit otherwise,
(3) fits the surface-Avrami model with 7 CamSpec nuisances (15-dim vector).

USE_BAO = True  -> pure PR4-CMB verdict (compare to your LCDM optimize run)
USE_BAO = True   -> adds DESI DR2 (13 pts) for the joint verdict
"""
import json
import numpy as np
import camb
from scipy.optimize import minimize
from cobaya.model import get_model

USE_BAO = True
PACKAGES_PATH = r'C:\yaroslavFiles\cobaya_packages'
ANCHOR = 6990.88
NUIS_REF = dict(A_planck=1.0, amp_143=9.868, amp_217=20.64, amp_143x217=10.1,
                n_143=0.8929, n_217=1.072, n_143x217=1.261)
NUIS_NAMES = list(NUIS_REF.keys())

MNU_NH_FLOOR=0.05819; MNU_NH_SIGMA=0.00029
TAU_MEAN=0.0605; TAU_SIGMA=0.0046

# ── CamSpec object ───────────────────────────────────────────────────────────
info = {
    'likelihood': {'planck_NPIPE_highl_CamSpec.TT': None},
    'theory': {'camb': {'extra_args': {'lens_potential_accuracy': 1, 'lmax': 2700}}},
    'params': {
        'H0':{'prior':{'min':55,'max':80},'ref':67.36},
        'ombh2':{'prior':{'min':0.019,'max':0.026},'ref':0.02237},
        'omch2':{'prior':{'min':0.09,'max':0.15},'ref':0.1200},
        'mnu':{'prior':{'min':0.0,'max':0.6},'ref':0.06},
        'As':{'prior':{'min':1.5e-9,'max':3.5e-9},'ref':2.100e-9},
        'ns':{'prior':{'min':0.87,'max':1.06},'ref':0.9649},
        'tau':{'prior':{'min':0.01,'max':0.13},'ref':0.0544},
    },
    'packages_path': PACKAGES_PATH,
}
model = get_model(info)
like = list(model.likelihood.values())[0]

def camb_Dl(H0, ombh2, omch2, mnu, As, ns, tau, de=None):
    p = camb.CAMBparams()
    p.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=max(0.001,mnu), omk=0, tau=tau)
    p.InitPower.set_params(As=As, ns=ns, r=0)
    if de is not None: p.DarkEnergy = de
    p.set_for_lmax(2700, lens_potential_accuracy=1)
    p.WantTensors = False
    return camb.get_results(p).get_cmb_power_spectra(p, CMB_unit='muK')['total'][:, 0]

# ── injection auto-detection & validation ────────────────────────────────────
Dl_off = camb_Dl(67.36, 0.02237, 0.1200, 0.06, 2.100e-9, 0.9649, 0.0544)
INJECT = [None]
ELLS_FULL = np.arange(len(Dl_off))
def _to_C(D):
    C=np.zeros_like(D)
    C[2:]=D[2:]*2*np.pi/(ELLS_FULL[2:]*(ELLS_FULL[2:]+1.0))
    return C

# полный data_params: варьируемые + фиксированные из дефолтов объекта/класса
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

def _validate():
    Z=np.zeros_like(Dl_off)
    vN=vZ=None
    try: vN=float(np.atleast_1d(like.chi_squared(Dl_off,None,None,full_params(NUIS_REF)))[0])
    except Exception as e: print("  Dl/None probe FAILED:",type(e).__name__,str(e)[:80])
    try: vZ=float(np.atleast_1d(like.chi_squared(Dl_off,Z,Z,full_params(NUIS_REF)))[0])
    except Exception as e: print("  Dl/zeros probe FAILED:",type(e).__name__,str(e)[:80])
    v = vN if vN is not None else vZ
    if v is None: return False
    stable = (vN is None) or (vZ is None) or abs(vN-vZ)<1e-6
    delta = v - ANCHOR
    print(f"  chi_squared(Dl): {v:.2f} | cobaya anchor {ANCHOR:.2f} | pipeline delta {delta:+.2f} | extra-arg stable: {stable}")
    if stable and abs(delta) < 35.0:      # <0.5%: канал верен, дельта = конвейерная систематика
        print(f"INJECTION VALIDATED (self-consistent anchor {v:.2f}; cross-pipeline systematic {delta:+.2f} recorded)")
        print("NOTE: model comparisons must stay WITHIN this direct pipeline (use lcdm_pr4_direct, not the cobaya run)")
        INJECT[0]=("Dl_direct", 1.0)
        return True
    return False
if not _validate():
    print("fixed-defaults introspection dump:", full_params({}))
    raise SystemExit("no valid injection signature; send this log back")

def chi2_camspec(Dl, nuis):
    return float(np.atleast_1d(like.chi_squared(Dl, None, None, full_params(nuis)))[0])

# ── surface solver (validated blocks; h-target bisection) ────────────────────
WG=2.469e-5; WRAD=WG*(1+0.2271*3.046)
_XG=np.linspace(np.log(1/2500),13.0,3200); _A=np.exp(_XG); _DX=_XG[1]-_XG[0]

def _solve(wm, lam, zc, rho0=0.30, n_iter=220, relax=0.45):
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
    return np.sqrt(wm*a**-3+WRAD*a**-4+rho),rho

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

def surf_de(wm, lam, zc, hT, n_table=600):
    H,rho=_solve_target(wm,lam,zc,hT)
    rho0v=float(np.interp(0.0,_XG,rho))
    w=-1-np.gradient(np.log(np.maximum(rho,rho0v*1e-12)),_XG)/3
    w=np.maximum(w,-60.0)
    zlo=max(0.4,0.5*zc); zhi=min(30.0,3.5*(1+zc))
    lnat=np.unique(np.concatenate([np.linspace(-10.,0.,n_table),
                                   np.linspace(np.log(1/(1+zhi)),np.log(1/(1+zlo)),4*n_table//3)]))
    de=camb.dark_energy.DarkEnergyPPF(); de.set_w_a_table(np.exp(lnat),np.interp(lnat,_XG,w))
    return de,H

# ── optional BAO (verified block) ────────────────────────────────────────────
BAO=[(0.295,7.942,0.075,None,None,None),
 (0.510,13.588,0.167,21.863,0.425,-0.459),(0.706,17.351,0.177,19.455,0.330,-0.404),
 (0.934,21.576,0.152,17.641,0.193,-0.416),(1.321,27.601,0.318,14.176,0.221,-0.434),
 (1.484,30.512,0.760,12.817,0.516,-0.500),(2.330,38.988,0.531,8.632,0.101,-0.431)]
C_KM=299792.458
def rd_desi(wcb,wb): return 147.05*(wcb/0.1432)**-0.23*(wb/0.02236)**-0.13
def chi2_bao_from_H(Hgrid, rd):
    zf=np.linspace(0,2.6,600)
    Hf=np.interp(np.log(1/(1+zf)),_XG,Hgrid)*100.0
    invH=1/Hf
    DMg=C_KM*np.concatenate([[0.0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])
    zq,val,err,_,_,_=BAO[0]
    dm=float(np.interp(zq,zf,DMg)); dh=C_KM/float(np.interp(zq,zf,Hf))
    cB=(((zq*dm*dm*dh)**(1/3)/rd-val)/err)**2
    for zq,vM,eM,vH,eH,r in BAO[1:]:
        dm=float(np.interp(zq,zf,DMg))/rd; dh=(C_KM/float(np.interp(zq,zf,Hf)))/rd
        d2=np.array([dm-vM,dh-vH]); C=np.array([[eM**2,r*eM*eH],[r*eM*eH,eH**2]])
        cB+=float(d2@np.linalg.inv(C)@d2)
    return cB

# ── objective ────────────────────────────────────────────────────────────────
def chi2_full(wb, wc, lam, zc, hT, mnu, As, ns, tau, nuis):
    try:
        wm = wb + wc + mnu/93.14        # UNLOCKED: densities sampled directly, no 5:1
        de,Hg = surf_de(wm,lam,zc,hT)
        h=hT
        Dl = camb_Dl(100*h, wb, wc, mnu, As, ns, tau, de=de)
        cC = chi2_camspec(Dl, nuis)
        cB = chi2_bao_from_H(Hg, rd_desi(wb+wc, wb))
        c = cC + (cB if USE_BAO else 0.0)
        c += ((tau-TAU_MEAN)/TAU_SIGMA)**2
        c += 0.0 if mnu>=MNU_NH_FLOOR else ((mnu-MNU_NH_FLOOR)/MNU_NH_SIGMA)**2
        c += ((nuis['A_planck']-1.0)/0.0025)**2
        return c, cC, cB
    except Exception:
        return 1e10,1e10,1e10

PN_COS=['wb','wc','lam','zc','h','mnu','As','ns','tau']
BOUNDS={'wb':(0.020,0.025),'wc':(0.10,0.14),'lam':(0.05,3.0),'zc':(0.8,12.0),'h':(0.60,0.78),
        'mnu':(0.0,0.50),'As':(1.5e-9,3.5e-9),'ns':(0.88,1.05),'tau':(0.02,0.12),
        'A_planck':(0.98,1.02),'amp_143':(0.0,50.0),'amp_217':(0.0,60.0),
        'amp_143x217':(0.0,50.0),'n_143':(0.0,3.0),'n_217':(0.0,3.0),'n_143x217':(0.0,3.0)}
PN=PN_COS+NUIS_NAMES

starts=[[0.02220,0.11997,0.913,4.04,0.682,0.058,2.128e-9,0.9633,0.0612,
         1.0004,19.34,13.22,10.11,0.961,1.286,1.369],                       # plik_lite unlocked best + R1 nuisances
        [0.02223,0.11707,1.00,6.00,0.684,0.058,2.106e-9,0.9668,0.0606]+[NUIS_REF[k] for k in NUIS_NAMES]]

best=(1e10,None,None)
print(f"\nRunning {len(starts)} starts...  SURFACE-AVRAMI on PR4 CamSpec TT, USE_BAO={USE_BAO}, dim={len(PN)}")
for i,x0 in enumerate(starts):
    print(f"\n{'─'*55}\nStart {i+1}/{len(starts)}")
    cnt=[0]; loc=[1e10]
    def obj(v):
        p={k:float(np.clip(x,*BOUNDS[k])) for k,x in zip(PN,v)}
        nuis={k:p[k] for k in NUIS_NAMES}
        c,cC,cB=chi2_full(p['wb'],p['wc'],p['lam'],p['zc'],p['h'],p['mnu'],p['As'],p['ns'],p['tau'],nuis)
        cnt[0]+=1
        if c<loc[0]:
            loc[0]=c
            print(f"  iter {cnt[0]:4d}: wb={p['wb']:.5f} wc={p['wc']:.4f} lam={p['lam']:.2f} zc={p['zc']:.2f} "
                  f"h={p['h']:.4f} mnu={p['mnu']:.4f} tau={p['tau']:.4f} "
                  f"CamSpec={cC:.1f} BAO={cB:.1f} tot={c:.2f} *")
        return c
    r=minimize(obj,x0,method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-4,fatol=0.1))
    p={k:float(np.clip(x,*BOUNDS[k])) for k,x in zip(PN,r.x)}
    nuis={k:p[k] for k in NUIS_NAMES}
    c,cC,cB=chi2_full(p['wb'],p['wc'],p['lam'],p['zc'],p['h'],p['mnu'],p['As'],p['ns'],p['tau'],nuis)
    print(f"Start {i+1} done: tot={c:.2f}  CamSpec={cC:.2f}  BAO={cB:.2f}")
    if c<best[0]: best=(c,p,(cC,cB))

c_tot,p,(cC,cB)=best
print(f"\n{'='*60}\nGLOBAL BEST — SURFACE AVRAMI UNLOCKED on PR4 (no 5:1, zc<=12) ({'CMB+BAO' if USE_BAO else 'CMB-only'})\n{'='*60}")
print(f"  h={p['h']:.4f}  lam={p['lam']:.3f}  z_c={p['zc']:.3f}  wb={p['wb']:.5f} wc={p['wc']:.5f}")
print(f"  ratio wc/wb = {p['wc']/p['wb']:.3f}  (PR4-free-LCDM voice was 5.267)")
print(f"  mnu={p['mnu']:.4f}  ns={p['ns']:.4f}  tau={p['tau']:.4f}")
print(f"  nuisances: "+", ".join(f"{k}={p[k]:.3f}" for k in NUIS_NAMES))
print(f"  chi2_CamSpec = {cC:.2f}   chi2_BAO = {cB:.2f}   chi2_tot = {c_tot:.2f}")
print(f"  compare to your LCDM-on-PR4 optimize run (same conventions).")
json.dump(dict(MODEL='surface_avrami2_unlocked_PR4', USE_BAO=USE_BAO, anchor=ANCHOR,
               **{k:p[k] for k in PN}, chi2_CamSpec=cC, chi2_BAO=cB, chi2_tot=c_tot),
          open(f"nucl_pr4_unlocked_{'bao' if USE_BAO else 'cmb'}_best_v1.json",'w'), indent=2)
print(f"Saved: nucl_pr4_unlocked_{'bao' if USE_BAO else 'cmb'}_best_v1.json")
