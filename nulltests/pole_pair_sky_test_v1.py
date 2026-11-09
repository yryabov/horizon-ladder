"""
Pole-pair sky test v1 — the "three lunes" configuration: two antipodal triple-junction
defects on the horizon screen (N0 = 3, luminal fronts; notes sec. 18d).
Fingerprint (10-deg cores): only EVEN ell (C1=C3=0), quadrupole-dominant, C4/C2 = 0.65,
quadrupole ZONAL (m=0) in the axis frame.
Statistics on the real low-ell sky (same LSQ cut-sky harmonics as the tetra test):
  (A) quadrupole zonality: max over axes of the m=0 power fraction  (planar quadrupole
      = |m|=2 would be the anti-signature); PTE from Gaussian MC;
  (B) matched filter of an antipodal same-sign pair over all axes, both signs,
      band-limited ell<=LMAX, equal look-elsewhere MC; prints the best axis (l,b) and its
      angle to the Planck quadrupole-octupole axis (catalog value from memory, +-10 deg).
Setup identical to tetra_sky_test_v2.py (astropy + astropy-healpix; MAP_PATH below).
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
N_ORIENT   = 1200       # random orientations for the matched filter
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
def wigner_d(L,beta):
    d=np.zeros((2*L+1,2*L+1))
    for i,mp in enumerate(range(-L,L+1)):
        for j,m in enumerate(range(-L,L+1)):
            s=0.0
            for k in range(max(0,m-mp),min(L+m,L-mp)+1):
                num=np.sqrt(factorial(L+mp)*factorial(L-mp)*factorial(L+m)*factorial(L-m))
                den=factorial(L+m-k)*factorial(k)*factorial(L-mp-k)*factorial(k+mp-m)
                s+=(-1)**(k+mp-m)*num/den*np.cos(beta/2)**(2*L+m-mp-2*k)*np.sin(beta/2)**(2*k+mp-m)
            d[i,j]=s
    return d
def wigner_d3(beta): return wigner_d(3,beta)
def multipole_complex(coefs,labels,L):
    a={}
    for c,(LL,m,t) in zip(coefs,labels):
        if LL!=L: continue
        if m==0: a[0]=a.get(0,0)+c
        else:
            if t=='c': a[m]=a.get(m,0)+c/np.sqrt(2)
            else:      a[m]=a.get(m,0)-1j*c/np.sqrt(2)
    alm=np.zeros(2*L+1,complex)
    for m in range(0,L+1):
        alm[L+m]=a.get(m,0)
        if m>0: alm[L-m]=(-1)**m*np.conj(a.get(m,0))
    return alm
def zonal_fraction_q(alm2):
    """max over axes of |a_20|^2 / sum|a_2m|^2 (quadrupole zonality)"""
    best=0.0; best_ang=(0,0)
    for alpha in np.linspace(0,2*np.pi,36,endpoint=False):
        ea=np.exp(-1j*np.arange(-2,3)*alpha)
        for beta in np.linspace(0,np.pi,37):
            rot=wigner_d(2,beta)@(ea*alm2)
            P=np.abs(rot)**2; f0=P[2]/P.sum()
            if f0>best: best=f0; best_ang=(alpha,beta)
    return best,best_ang
def frame_fractions(alm):
    return zonal_fraction_q(alm)[0],0.0
alm2_obs=multipole_complex(coef,labels,2)
f0_obs,(al_b,be_b)=zonal_fraction_q(alm2_obs)
print(f"quadrupole zonality (max-axis m=0 fraction) = {f0_obs:.3f}   [pole pair predicts -> 1; planar -> low]")

# ── statistic B: tetra matched filter ───────────────────────────────────────
V0=np.array([[0,0,1.0],[0,0,-1.0]])   # antipodal pair; orientation = axis
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
best_hot=-9; best_cold=-9; best_axis=None
for R in Rs:
    verts=R.apply(V0); t=template(verts)
    r=float(Tn@t)
    if r>best_hot: best_hot=r; ax_hot=verts[0]
    if -r>best_cold: best_cold=-r; ax_cold=verts[0]
best_r=max(best_hot,best_cold)
# CS-anchored family (one vertex fixed): scan azimuth
zv=np.array([np.cos(np.radians(CS_B))*np.cos(np.radians(CS_L)),
             np.cos(np.radians(CS_B))*np.sin(np.radians(CS_L)),
             np.sin(np.radians(CS_B))])
xv=np.cross([0,0,1],zv); xv/=np.linalg.norm(xv); yv=np.cross(zv,xv)
t_cs=template(np.array([zv,-zv])); r_cs=float(Tn@t_cs)
best_cs_hot=max(r_cs,-9); best_cs_cold=max(-r_cs,-9)
best_cs=max(best_cs_hot,best_cs_cold)
amp=best_cold*np.linalg.norm(T_band)/np.linalg.norm(T_band*0+1)   # грубая мК-нормировка ниже
A_cold_uK=best_cold*np.linalg.norm(T_band)/np.sqrt(len(T_band))
def lb(v):
    return (np.degrees(np.arctan2(v[1],v[0]))%360, np.degrees(np.arcsin(v[2])))
QO=np.array([np.cos(np.radians(60))*np.cos(np.radians(250)),np.cos(np.radians(60))*np.sin(np.radians(250)),np.sin(np.radians(60))])
def axang(v,w): return np.degrees(np.arccos(min(1,abs(float(v@w)))))
print(f"pair matched filter (band-limited): hot r={best_hot:.4f} axis (l,b)=({lb(ax_hot)[0]:.0f},{lb(ax_hot)[1]:+.0f}); cold r={best_cold:.4f} axis ({lb(ax_cold)[0]:.0f},{lb(ax_cold)[1]:+.0f})")
print(f"  angle to Q-O axis (250,+60; memory +-10): hot {axang(ax_hot,QO):.0f} deg, cold {axang(ax_cold,QO):.0f} deg;  CS-anchored pair hot/cold = {best_cs_hot:.4f}/{best_cs_cold:.4f}")
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
    f2_mc[i]=zonal_fraction_q(multipole_complex(co,labels,2))[0] if i<300 else np.nan
    rb=-9
    for R in Rs_mc:
        r=float(Tsn@template(R.apply(V0)))
        rb=max(rb,abs(r))          # оба знака, тот же N_ORIENT — выровненный look-elsewhere
    r_mc[i]=rb
    if i%200==0: print(f"  MC {i}/{N_MC}")
f2v=f2_mc[:300]
print("\nRESULTS")
print(f"  A: quadrupole zonality: obs {f0_obs:.3f}; MC median {np.nanmedian(f2v):.3f}; PTE = {np.nanmean(f2v>=f0_obs):.3f}")
print(f"  B: pair |r| both signs (all axes): obs {best_r:.4f}; MC median {np.median(r_mc):.4f}; PTE = {np.mean(r_mc>=best_r):.3f}")
print(f"     (CS-anchored r = {best_cs:.4f}: compare against the same MC as a conservative reference)")
print("NOTE: for COLD spots the template correlation should be taken with -T; run both signs.")
