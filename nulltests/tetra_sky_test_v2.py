"""
Tetrahedral sky test v1 — four half-charge disclination imprints on the CMB large angles
=======================================================================================
Tests the tangent-director scenario of the screen condensate (session_notes_condensate_door,
sec. 10-11a): topology forces EXACTLY four +1/2 defects in a tetrahedral arrangement.
Fingerprint: C1=C2=0, octupole-dominant with pure |m|=2 in the tetra frame, C4/C3=0.37.

Statistics computed on the real low-ell sky:
  (A) octupole m-structure: max-frame |m|=2 fraction (tetrahedral) vs |m|=3 (planar),
      significance from Gaussian LCDM Monte Carlo;
  (B) matched filter: four equal Gaussian spots on a tetrahedron, free orientation
      (+ separate Cold-Spot-anchored variant), significance from the same MC.

SETUP (Windows-friendly, no healpy needed):
  pip install astropy astropy-healpix numpy scipy
  Download a Planck CMB temperature map (Commander or SMICA) from the Planck Legacy
  Archive, e.g. product "COM_CMB_IQU-commander_1024_R2.02_full.fits" (or any I map;
  higher Nside is fine, it is downgraded by the LSQ to ell<=8), and set MAP_PATH.
"""
import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
import astropy.units as u
from scipy.special import sph_harm_y
from scipy.spatial.transform import Rotation

MAP_PATH   = r'C:\yaroslavFiles\maps\COM_CMB_IQU-commander_1024_R2.02_full.fits'
FIELD      = 0          # I column
GAL_CUT    = 20.0       # |b| mask, degrees
LMAX       = 8
CORE_DEG   = 10.0       # defect core width for the matched filter
N_MC       = 800       # Gaussian MCs for significance
N_ORIENT   = 1500       # random orientations for the matched filter
SEED       = 42
D_ELL_LCDM = 1100.0     # SW-plateau fiducial, muK^2 (for MC; flat over 2..LMAX)
CS_L, CS_B = 209.0, -57.0   # Cold Spot anchor (galactic)

rng=np.random.default_rng(SEED)

# ── load map, build directions ───────────────────────────────────────────────
with fits.open(MAP_PATH) as h:
    hdr=h[1].header
    nside=hdr['NSIDE']; order=hdr.get('ORDERING','RING').strip().upper()
    Tmap=np.asarray(h[1].data.field(FIELD),dtype=float).ravel()
if np.nanmedian(np.abs(Tmap))<1e-2: Tmap=Tmap*1e6   # K -> muK
hp=HEALPix(nside=nside, order='ring' if order=='RING' else 'nested', frame=Galactic())
npix=hp.npix
lon,lat=hp.healpix_to_lonlat(np.arange(npix))
b=lat.to_value(u.deg); l=lon.to_value(u.deg)
good=np.isfinite(Tmap)&(np.abs(b)>GAL_CUT)
# thin pixels for speed (LSQ at lmax=8 needs no more than ~20k points)
idx=np.where(good)[0]
if idx.size>20000: idx=rng.choice(idx,20000,replace=False)
th=np.radians(90.0-b[idx]); ph=np.radians(l[idx])
T=Tmap[idx]-np.mean(Tmap[idx])
print(f"map: nside={nside} {order}, used pixels {idx.size}, |b|>{GAL_CUT}")

# ── LSQ harmonic fit on the cut sky (real-valued basis), ell=1..LMAX ─────────
def real_ylm_matrix(th,ph,lmax):
    cols=[]; labels=[]
    for L in range(1,lmax+1):
        for m in range(0,L+1):
            Y=sph_harm_y(L,m,th,ph)
            if m==0:
                cols.append(Y.real); labels.append((L,0,'c'))
            else:
                cols.append(np.sqrt(2)*Y.real); labels.append((L,m,'c'))
                cols.append(np.sqrt(2)*Y.imag); labels.append((L,m,'s'))
    return np.array(cols).T, labels
A,labels=real_ylm_matrix(th,ph,LMAX)
coef,*_=np.linalg.lstsq(A,T,rcond=None)
def block(L):
    v=np.array([c for c,(LL,m,t) in zip(coef,labels) if LL==L])
    return v
C_meas={L: float(np.sum(block(L)**2)/(2*L+1)) for L in range(1,LMAX+1)}
print("recovered C_ell (muK^2):", {L: round(C_meas[L],1) for L in C_meas})

# ── statistic A: octupole frame m-structure ─────────────────────────────────
def octupole_complex(coefs,labels):
    a={}
    for c,(L,m,t) in zip(coefs,labels):
        if L!=3: continue
        if m==0: a[0]=a.get(0,0)+c
        else:
            if t=='c': a[m]=a.get(m,0)+c/np.sqrt(2)
            else:      a[m]=a.get(m,0)-1j*c/np.sqrt(2)
    alm=np.zeros(7,complex)
    for m in range(0,4):
        alm[3+m]=a.get(m,0)
        if m>0: alm[3-m]=(-1)**m*np.conj(a.get(m,0))
    return alm
from scipy.special import factorial
def wigner_d3(beta):
    # d^3_{m'm}(beta) via explicit formula
    d=np.zeros((7,7))
    for i,mp in enumerate(range(-3,4)):
        for j,m in enumerate(range(-3,4)):
            s=0.0
            for k in range(max(0,m-mp),min(3+m,3-mp)+1):
                num=np.sqrt(factorial(3+mp)*factorial(3-mp)*factorial(3+m)*factorial(3-m))
                den=factorial(3+m-k)*factorial(k)*factorial(3-mp-k)*factorial(k+mp-m)
                s+=(-1)**(k+mp-m)*num/den*np.cos(beta/2)**(6+m-mp-2*k)*np.sin(beta/2)**(2*k+mp-m)
            d[i,j]=s
    return d
def frame_fractions(alm):
    best2=0.0; best3=0.0
    for alpha in np.linspace(0,2*np.pi,24,endpoint=False):
        ea=np.exp(-1j*np.arange(-3,4)*alpha)
        for beta in np.linspace(0,np.pi,25):
            d=wigner_d3(beta)
            rot=d@(ea*alm)
            P=np.abs(rot)**2; tot=P.sum()
            f2=(P[3-2]+P[3+2])/tot; f3=(P[3-3]+P[3+3])/tot
            best2=max(best2,f2); best3=max(best3,f3)
    return best2,best3
f2_obs,f3_obs=frame_fractions(octupole_complex(coef,labels))
print(f"octupole: max-frame |m|=2 fraction = {f2_obs:.3f} (tetrahedral); |m|=3 = {f3_obs:.3f} (planar)")

# ── statistic B: tetra matched filter ───────────────────────────────────────
V0=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
pix=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]).T
sig=np.radians(CORE_DEG)
def template(verts):
    t=np.zeros(len(pix))
    for v in verts:
        t+=np.exp((pix@v-1)/sig**2)     # ~exp(-theta^2/2sig^2)
    t-=t.mean(); n=np.linalg.norm(t)
    return t/n if n>0 else t
T_band=A@coef                      # v2: банд-лимит ell<=LMAX — та же полоса, что у MC
T_band-=T_band.mean()
Tn=T_band/np.linalg.norm(T_band)
Rs=Rotation.random(N_ORIENT,random_state=SEED)
best_hot=-9; best_cold=-9; best_tpl=None
for R in Rs:
    t=template(R.apply(V0))
    r=float(Tn@t)
    if r>best_hot: best_hot=r
    if -r>best_cold: best_cold=-r; best_tpl=t
best_r=max(best_hot,best_cold)
# CS-anchored family (one vertex fixed): scan azimuth
zv=np.array([np.cos(np.radians(CS_B))*np.cos(np.radians(CS_L)),
             np.cos(np.radians(CS_B))*np.sin(np.radians(CS_L)),
             np.sin(np.radians(CS_B))])
xv=np.cross([0,0,1],zv); xv/=np.linalg.norm(xv); yv=np.cross(zv,xv)
best_cs_hot=-9; best_cs_cold=-9
for phi in np.linspace(0,2*np.pi,181):
    vs=[zv]+[np.cos(np.radians(109.47))*zv+np.sin(np.radians(109.47))*(np.cos(phi+2*np.pi*k/3)*xv+np.sin(phi+2*np.pi*k/3)*yv) for k in range(3)]
    r=float(Tn@template(np.array(vs)))
    best_cs_hot=max(best_cs_hot,r); best_cs_cold=max(best_cs_cold,-r)
best_cs=max(best_cs_hot,best_cs_cold)
amp=best_cold*np.linalg.norm(T_band)/np.linalg.norm(T_band*0+1)   # грубая мК-нормировка ниже
A_cold_uK=best_cold*np.linalg.norm(T_band)/np.sqrt(len(T_band))
print(f"matched filter v2 (band-limited): hot r={best_hot:.4f}, cold r={best_cold:.4f}; CS-anchored hot/cold = {best_cs_hot:.4f}/{best_cs_cold:.4f}")
print(f"  cold-template best-fit amplitude ~ {A_cold_uK:.1f} muK-rms-equivalent")

# ── Monte Carlo significance ─────────────────────────────────────────────────
Cl={L: 2*np.pi*D_ELL_LCDM/(L*(L+1)) for L in range(2,LMAX+1)}; Cl[1]=0.0
f2_mc=np.zeros(N_MC); r_mc=np.zeros(N_MC); rcs_mc=np.zeros(N_MC)
Rs_mc=Rotation.random(N_ORIENT,random_state=SEED+1)
for i in range(N_MC):
    cmc=np.array([rng.normal(0,np.sqrt(Cl[L])) for L,(m) in [(L,0) for L in range(1,LMAX+1)]]) # placeholder
    # построить гауссово небо в тех же пикселях:
    csim=[]
    for L in range(1,LMAX+1):
        n=2*L+1
        csim.append(rng.normal(0,np.sqrt(Cl[L]),n))
    csim=np.concatenate(csim)
    Ts=A@csim; Ts-=Ts.mean(); Tsn=Ts/np.linalg.norm(Ts)
    co,*_=np.linalg.lstsq(A,Ts,rcond=None)
    f2_mc[i],_=frame_fractions(octupole_complex(co,labels)) if i<400 else (np.nan,0)
    rb=-9
    for R in Rs_mc:
        r=float(Tsn@template(R.apply(V0)))
        rb=max(rb,abs(r))          # оба знака, тот же N_ORIENT — выровненный look-elsewhere
    r_mc[i]=rb
    if i%200==0: print(f"  MC {i}/{N_MC}")
f2v=f2_mc[:400]
print("\nRESULTS")
print(f"  A: |m|=2 octupole fraction: obs {f2_obs:.3f}; MC median {np.nanmedian(f2v):.3f}; PTE = {np.nanmean(f2v>=f2_obs):.3f}")
print(f"  B(v2): free |r| both signs: obs {best_r:.4f}; MC median {np.median(r_mc):.4f}; PTE = {np.mean(r_mc>=best_r):.3f}")
print(f"     (CS-anchored r = {best_cs:.4f}: compare against the same MC as a conservative reference)")
print("NOTE: for COLD spots the template correlation should be taken with -T; run both signs.")
