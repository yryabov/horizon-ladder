"""
SN cost probe v1 — Pantheon+ chi2 at the FROZEN post-Results-IV (lya4) best fits
================================================================================
No refitting: evaluates the marginal SN cost of each battery model at its stored
best-fit background. Offset (M + H0) marginalized analytically; diagonal errors;
zHD > 0.023 (PV-conservative). Companion to the Lya first-look protocol.
Run:  python sn_cost_probe_v1.py
"""
import numpy as np, pandas as pd

DATA_PATH = r'C:\yaroslavFiles\sn\Pantheon+SH0ES.dat'
ZMIN = 0.023

WR=2.469e-5*(1+0.2271*3.046)
xg=np.linspace(np.log(1/2500),13.0,3000); dx=xg[1]-xg[0]; a=np.exp(xg)
def RE_of(H):
    f=1/(a*H)
    cum=np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*dx)[::-1],[0]])
    return a*(cum+1/(a[-1]*H[-1]))
def solve_avrami(wb,wc,mnu,h,zc,lam):
    wm=wb+wc+mnu/93.14; xc=np.log(1/(1+zc))
    al=np.where(xg>xc,1-np.exp(-((xg-xc)/lam)**2),0.0)
    lo,hi=0.02,2.4
    for _ in range(44):
        mid=(lo+hi)/2; r=np.full_like(xg,mid)
        for _ in range(200):
            H=np.sqrt(wm*a**-3+WR*a**-4+np.maximum(r,0)); r=0.55*r+0.45*al/RE_of(H)**2
        H=np.sqrt(wm*a**-3+WR*a**-4+r); hh=float(np.interp(1.0,a,H))
        if abs(hh-h)<3e-4: break
        if hh<h: lo=mid
        else: hi=mid
    return H
def solve_gc(wb,wc,mnu,h,Tc,p):
    wm=wb+wc+mnu/93.14
    H=np.sqrt(wm*a**-3+WR*a**-4+max(h*h-wm-WR,1e-8)); r=None
    for _ in range(300):
        RE=RE_of(H); T=1/(2*np.pi*RE)
        al=np.clip(1-(T/Tc)**p,0,1); raw=al/RE**2
        s=(h*h-wm-WR)/max(float(np.interp(0.0,xg,raw)),1e-30)
        new=s*raw; r=new if r is None else 0.5*r+0.5*new
        H=np.sqrt(wm*a**-3+WR*a**-4+np.maximum(r,0))
    return H
def solve_L2(wb,wc,mnu,h,u,zc):
    wm=wb+wc+mnu/93.14; xc=np.log(1/(1+zc)); born=xg>xc
    lo,hi=0.02,2.4
    for _ in range(44):
        mid=(lo+hi)/2; r=np.full_like(xg,mid)
        for _ in range(220):
            H=np.sqrt(wm*a**-3+WR*a**-4+np.maximum(r,0)); RE=RE_of(H)
            g=np.where(born,1.0/(H*RE),0.0)
            I=np.concatenate([[0.0],np.cumsum((g[:-1]+g[1:])/2*dx)])
            al=np.where(born,1.0-np.exp(-np.pi*(u*I)**2),0.0)
            r=0.55*r+0.45*al/RE**2
        H=np.sqrt(wm*a**-3+WR*a**-4+r); hh=float(np.interp(1.0,a,H))
        if abs(hh-h)<3e-4: break
        if hh<h: lo=mid
        else: hi=mid
    return H

# post-Results-IV (lya4) best fits
models={}
wb,wc,mnu,h=0.02253579,0.11782994,0.05818543,0.68351477
wm=wb+wc+mnu/93.14
models['LCDM']=np.sqrt(wm*a**-3+WR*a**-4+(h*h-wm-WR))
wb,wc,mnu,h,w0,wa=0.02242902,0.11943639,0.06117861,0.66038307,-0.71236846,-0.89263283
wm=wb+wc+mnu/93.14; Ode=h*h-wm-WR
models['CPL']=np.sqrt(wm*a**-3+WR*a**-4+Ode*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a)))
models['Avrami']=solve_avrami(0.02241772,0.11967418,0.05816609,0.67916724,4.17446729,0.90408431)
models['L2']=solve_L2(0.02234804,0.12065199,0.06110282,0.67143906,1.60450814,4.01748718)
models['GC-p']=solve_gc(0.02243617,0.11940516,0.06052618,0.67424729,0.15634851,2.03412350)
models['GC']=solve_gc(0.02226013,0.12180817,0.05890034,0.68051552,0.13617586,4.0)

# Pantheon+
df=pd.read_csv(DATA_PATH,sep=r'\s+',comment='#')
df=df[(df['IS_CALIBRATOR']==0)&(df['zHD']>=ZMIN)&(df['zHD']<=2.5)]
z=df['zHD'].to_numpy(float); mu=df['MU_SH0ES'].to_numpy(float)
err=df['MU_SH0ES_ERR_DIAG'].to_numpy(float); w=1.0/err**2
print(f"Pantheon+ loaded: {len(z)} SNe (z>{ZMIN}, calibrators excluded; diagonal errors)")

C=299792.458
zf=np.linspace(0,2.55,1200)
print(f"{'model':>8} {'chi2_SN':>9} {'d vs LCDM':>10}   [offset marginalized]")
base=None
for nm,H in models.items():
    Hz=100*np.interp(np.log(1/(1+zf)),xg,H)
    DC=C*np.concatenate([[0],np.cumsum((1/Hz[:-1]+1/Hz[1:])/2*np.diff(zf))])
    mu_th=5*np.log10(np.maximum((1+z)*np.interp(z,zf,DC),1e-9))+25
    r=mu-mu_th
    off=np.sum(w*r)/np.sum(w)
    c2=float(np.sum(w*(r-off)**2))
    if base is None: base=c2
    print(f"{nm:>8} {c2:9.2f} {c2-base:+10.2f}")
print("NOTE: LCDM row is the reference; internal comparison only (diagonal errors, no STAT+SYS).")
