"""
Pantheon+ seam test v2 (min N_in, survey marginalization, 22-deg spots) — directional expansion anisotropy in the screen-seam geometries
======================================================================================
Question (v) "oscillation channel" (notes 18k-18l): if the screen's seams modulate rho_de
along directions by delta, the integrated expansion along those directions differs:
dH/H ~ 0.34 delta. Seams are thin strips, so we FILTER instead of hemispheric averaging.

Pipeline:
  0) load Pantheon+SH0ES.dat (public: github.com/PantheonPlusSH0ES/DataRelease,
     Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat); drop calibrators; z>=ZMIN;
  1) isotropic flat-LCDM fit (H0, Om free; diagonal errors) -> residuals r_i = mu_i - mu_th(z_i);
  2) VALIDATION: hemispheric H0 dipole map (should reproduce the literature low-z dipole,
     ~2-4 km/s/Mpc, fading with z_min);
  3) SEAM TESTS: (a) three meridional strips about a free axis (N0=3 lunes), (b) antipodal
     pole spots, (c) tetrahedral spots; statistic T = <r>_in - <r>_out (weighted), maximized
     over orientations; null = permutations of residuals over the same orientation scan
     (equal look-elsewhere); dH/H = -T/2.17;
  4) REDSHIFT PERSISTENCE: T in z-bins at the global best orientation — local flows fade
     with z, a cosmological seam persists/grows.
Run:  python pantheon_seam_test_v1.py    (pandas, numpy, scipy)
"""
import numpy as np, pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

DATA_PATH = r'C:\yaroslavFiles\sn\Pantheon+SH0ES.dat'
ZMIN, ZMAX = 0.015, 2.5
STRIP_HALFWIDTH_DEG = 10.0     # meridional strip half-width
SPOT_RADIUS_DEG     = 22.0     # pole/tetra spot radius (v2: wider for sparse coverage)
MIN_NIN             = 60       # v2: orientations with fewer SNe inside are ignored
SURVEY_MARG         = True     # v2: subtract per-survey mean residual (footprint systematics)
N_ORIENT = 1500
N_PERM   = 200
SEED     = 7
rng = np.random.default_rng(SEED)

# ── 0. load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, sep=r'\s+', comment='#')
df = df[(df['IS_CALIBRATOR']==0)&(df['zHD']>=ZMIN)&(df['zHD']<=ZMAX)].copy()
z  = df['zHD'].to_numpy(float); mu = df['MU_SH0ES'].to_numpy(float)
err= df['MU_SH0ES_ERR_DIAG'].to_numpy(float)
ra = np.radians(df['RA'].to_numpy(float)); dec = np.radians(df['DEC'].to_numpy(float))
w  = 1.0/err**2
print(f"Pantheon+ loaded: {len(z)} SNe with {ZMIN}<=z<={ZMAX} (calibrators excluded)")

# equatorial J2000 -> galactic unit vectors (explicit rotation, no astropy needed)
def radec_to_gal(ra, dec):
    x=np.cos(dec)*np.cos(ra); y=np.cos(dec)*np.sin(ra); zz=np.sin(dec)
    R=np.array([[-0.0548755604,-0.8734370902,-0.4838350155],
                [ 0.4941094279,-0.4448296300, 0.7469822445],
                [-0.8676661490,-0.1980763734, 0.4559837762]])
    v=R@np.vstack([x,y,zz])
    return v.T
n_hat = radec_to_gal(ra, dec)                       # (N,3) galactic
def lb(v): return (np.degrees(np.arctan2(v[1],v[0]))%360, np.degrees(np.arcsin(v[2])))

# ── 1. isotropic flat-LCDM fit ───────────────────────────────────────────────
C_KM=299792.458
def mu_model(zs, H0, Om):
    E=lambda x: 1/np.sqrt(Om*(1+x)**3+(1-Om))
    zu=np.unique(zs); dc=np.array([quad(E,0,zz)[0] for zz in zu])
    dl=(1+zu)*C_KM/H0*dc
    m=5*np.log10(dl)+25
    return np.interp(zs, zu, m)
def chi2_iso(p):
    H0,Om=p
    if not (50<H0<90 and 0.05<Om<0.7): return 1e12
    return float(np.sum(w*(mu-mu_model(z,H0,Om))**2))
res=minimize(chi2_iso,[70,0.33],method='Nelder-Mead',options={'xatol':1e-3,'fatol':1e-2})
H0,Om=res.x
r = mu-mu_model(z,H0,Om)
print(f"isotropic fit: H0={H0:.2f}  Om={Om:.3f}  chi2/N={res.fun/len(z):.3f}  (diagonal errors)")
if SURVEY_MARG and 'IDSURVEY' in df.columns:
    sid=df['IDSURVEY'].to_numpy()
    for s in np.unique(sid):
        m=sid==s
        if m.sum()>=5: r[m]-=np.sum(w[m]*r[m])/np.sum(w[m])
    print(f"survey marginalization: per-survey weighted mean residual removed ({len(np.unique(sid))} surveys)")

# ── helpers ──────────────────────────────────────────────────────────────────
def wmean(x,mask):
    ww=w[mask]; return float(np.sum(ww*x[mask])/np.sum(ww)) if ww.size>3 else np.nan
def T_stat(rr, mask):
    if mask.sum()<MIN_NIN or (~mask).sum()<MIN_NIN: return np.nan
    return wmean(rr,mask)-wmean(rr,~mask)

# ── 2. VALIDATION: hemispheric H0 dipole ─────────────────────────────────────
dirs=Rotation.random(600,random_state=SEED).apply(np.array([0,0,1.0]))
best=(0,None)
for d in dirs:
    m=(n_hat@d)>0
    T=T_stat(r,m)
    if np.isfinite(T) and abs(T)>abs(best[0]): best=(T,d)
Td,dd=best
print(f"VALIDATION hemispheric dipole: dmu = {Td:+.4f} mag  -> dH/H = {-Td/2.17*100:+.2f}%  "
      f"(dH0 ~ {-Td/2.17*H0:+.2f} km/s/Mpc), axis (l,b)=({lb(dd)[0]:.0f},{lb(dd)[1]:+.0f})")
for zlo,zhi in ((0.015,0.05),(0.05,0.15),(0.15,2.5)):
    sel=(z>=zlo)&(z<zhi); m=(n_hat@dd)>0
    T=T_stat(np.where(sel,r,np.nan),m&sel) if sel.sum()>20 else np.nan
    # recompute cleanly on the subset
    rr=r[sel]; mm=m[sel]; ww=w[sel]
    if mm.sum()>5 and (~mm).sum()>5:
        T=float(np.sum(ww[mm]*rr[mm])/np.sum(ww[mm])-np.sum(ww[~mm]*rr[~mm])/np.sum(ww[~mm]))
        print(f"   z in [{zlo},{zhi}): N={sel.sum():4d}  dipole dH/H = {-T/2.17*100:+.2f}%  (expected: fades with z if local)")

# ── 3. SEAM GEOMETRIES ───────────────────────────────────────────────────────
sw=np.radians(STRIP_HALFWIDTH_DEG); sr=np.cos(np.radians(SPOT_RADIUS_DEG))
def mask_strips(R):
    # three meridional half-planes at azimuths 0,120,240 about axis e3=R[:,2]
    e1,e2,e3=R[:,0],R[:,1],R[:,2]
    x=n_hat@e1; y=n_hat@e2
    phi=np.arctan2(y,x)
    m=np.zeros(len(z),bool)
    for k in range(3):
        dphi=np.angle(np.exp(1j*(phi-2*np.pi*k/3)))
        m|=np.abs(dphi)<sw
    return m
def mask_poles(R):
    e3=R[:,2]; c=n_hat@e3
    return (c>sr)|(c<-sr)
V_tet=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
def mask_tetra(R):
    m=np.zeros(len(z),bool)
    for v in (R@V_tet.T).T: m|=(n_hat@v)>sr
    return m
geoms={'three strips (N0=3 lunes)':mask_strips,'antipodal poles':mask_poles,'tetrahedron':mask_tetra}
Rs=[Rotation.random(random_state=SEED+i).as_matrix() for i in range(N_ORIENT)]
print("\nSEAM TESTS (T = <r>_in - <r>_out, mag; dH/H = -T/2.17):")
results={}
def refine(fn, rr, R0, T0):
    """local orientation refinement: jittered rotations with shrinking scale"""
    best=(abs(T0),R0,T0)
    for scale in (8.0,4.0,2.0,1.0):
        for _ in range(25):
            Rj=R0@Rotation.from_rotvec(np.radians(rng.normal(0,scale,3))).as_matrix()
            T=T_stat(rr,fn(Rj))
            if np.isfinite(T) and abs(T)>best[0]: best=(abs(T),Rj,T)
        R0=best[1]
    return best[1],best[2]
for name,fn in geoms.items():
    masks=[fn(R) for R in Rs]
    Ts=np.array([T_stat(r,m) for m in masks])
    j=int(np.nanargmax(np.abs(Ts)))
    Rb,Tobs=refine(fn,r,Rs[j],Ts[j])
    mbest=fn(Rb)
    # permutation null: same coarse scan (subsampled x3) + same refinement
    null=np.empty(N_PERM)
    for p in range(N_PERM):
        rp=rng.permutation(r)
        Tp=np.array([T_stat(rp,m) for m in masks[::3]])
        jp=int(np.nanargmax(np.abs(Tp)))
        _,Tr=refine(fn,rp,Rs[::3][jp],Tp[jp])
        null[p]=abs(Tr)
    pte=float(np.mean(null>=abs(Tobs)))
    nin=int(mbest.sum())
    sig=1.0/np.sqrt(np.sum(w[mbest]))+1.0/np.sqrt(np.sum(w[~mbest]))
    results[name]=(Tobs,Rb,pte)
    ax=Rb[:,2]
    print(f"  {name:28}: T = {Tobs:+.4f} mag (N_in={nin}, ~sigma {sig:.4f})  -> dH/H = {-Tobs/2.17*100:+.2f}%  "
          f"-> delta_DE ~ {-Tobs/2.17/0.34:+.2f}   PTE = {pte:.3f}   axis (l,b)=({lb(ax)[0]:.0f},{lb(ax)[1]:+.0f})")
    m=mbest
    # 4. redshift persistence at the best orientation
    line="     z-bins:"
    for zlo,zhi in ((0.015,0.05),(0.05,0.15),(0.15,2.5)):
        sel=(z>=zlo)&(z<zhi); rr=r[sel]; mm=m[sel]; ww=w[sel]
        if mm.sum()>5 and (~mm).sum()>5:
            T=float(np.sum(ww[mm]*rr[mm])/np.sum(ww[mm])-np.sum(ww[~mm]*rr[~mm])/np.sum(ww[~mm]))
            s=1/np.sqrt(np.sum(ww[mm]))+1/np.sqrt(np.sum(ww[~mm]))
            line+=f"  [{zlo},{zhi}): {-T/2.17*100:+.2f}% ± {s/2.17*100:.2f}%"
        else: line+=f"  [{zlo},{zhi}): n/a"
    print(line+"   (local flows fade with z; a cosmological seam persists)")
print("\nNOTE: diagonal errors only (no STAT+SYS covariance); calibrators excluded; zHD (PV-corrected).")
