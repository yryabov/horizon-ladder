"""
BAO pulls — per-point deviations of the three joint best fits (platform version)
=================================================================================
Loads: lcdm_bao_best_v1.json, surf_bao_best_v1.json, nucl_bao_best_sfree_v1.json
Rebuilds each background EXACTLY as the fit scripts did, computes:
  (a) per-observable diagonal pulls (model - data)/sigma,
  (b) per-tracer correlated chi2 contributions (2x2 blocks with rho_MH).
SELF-VALIDATION: the summed correlated chi2_BAO must match each JSON's chi2_BAO
to ~0.1 — printed side by side; mismatch => background reconstruction bug.
Outputs: table to stdout, bao_pulls_v1.png, bao_pulls_v1.csv
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

J_LCDM='lcdm_bao_best_v1.json'; J_SURF='surf_bao_best_v1.json'; J_VOL='nucl_bao_best_sfree_v1.json'

WG=2.469e-5; WRAD=WG*(1+0.2271*3.046); C_KM=299792.458
_XG=np.linspace(np.log(1/2500),13.0,3200); _A=np.exp(_XG); _DX=_XG[1]-_XG[0]

def H_grid_lcdm(H0,ombh2,omch2,mnu):
    wmt=ombh2+omch2+mnu/93.14; h2=(H0/100)**2
    return 100*np.sqrt(wmt*_A**-3+WRAD*_A**-4+(h2-wmt-WRAD))

def H_grid_vol(wm,s):
    a=_A; rho=np.full_like(_XG,0.45)
    for _ in range(160):
        H=np.sqrt(wm*a**-3+WRAD*a**-4+np.maximum(rho,0))
        f=1/(a*H)
        cum=np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*_DX)[::-1],[0]])
        RE=a*(cum+1/(a[-1]*H[-1])); C=H[-1]**2+2*s*H[-1]
        rho=0.5*rho+0.5*np.maximum(C-2*s/RE,0)
    return 100*np.sqrt(wm*a**-3+WRAD*a**-4+rho)

def H_grid_surf(wm,g,zc):
    a=_A; xc=np.log(1/(1+zc))
    alpha=np.where(_XG>=xc,1-np.exp(-g*(_XG-xc)),0.0)
    rho=np.full_like(_XG,0.30)
    for _ in range(220):
        H=np.sqrt(wm*a**-3+WRAD*a**-4+np.maximum(rho,0))
        f=1/(a*H)
        cum=np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*_DX)[::-1],[0]])
        RE=a*(cum+1/(a[-1]*H[-1]))
        rho=0.55*rho+0.45*alpha/RE**2
    return 100*np.sqrt(wm*a**-3+WRAD*a**-4+rho)

BAO=[(0.295,7.942,0.075,None,None,None),
 (0.510,13.588,0.167,21.863,0.425,-0.459),(0.706,17.351,0.177,19.455,0.330,-0.404),
 (0.934,21.576,0.152,17.641,0.193,-0.416),(1.321,27.601,0.318,14.176,0.221,-0.434),
 (1.484,30.512,0.760,12.817,0.516,-0.500),(2.330,38.988,0.531,8.632,0.101,-0.431)]
def rd_desi(wcb,wb): return 147.05*(wcb/0.1432)**-0.23*(wb/0.02236)**-0.13

def wb_from_51(wm,mnu,h):
    h2=h*h; Om=wm/h2; Onu=mnu/(93.14*h2); Ob=(Om-Onu)/6.0
    return Ob*h2

def analyze(H,rd):
    zf=np.linspace(0,2.6,600)
    Hf=np.interp(np.log(1/(1+zf)),_XG,H)
    invH=1/Hf
    DMg=C_KM*np.concatenate([[0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])
    pulls=[]; labels=[]; chi_tr=[]
    zq,val,err,_,_,_=BAO[0]
    dm=np.interp(zq,zf,DMg); dh=C_KM/np.interp(zq,zf,Hf)
    pV=((zq*dm*dm*dh)**(1/3)/rd-val)/err
    pulls.append(pV); labels.append((zq,'DV')); chi_tr.append((zq,pV*pV))
    for zq,vM,eM,vH,eH,r in BAO[1:]:
        dm=np.interp(zq,zf,DMg)/rd; dh=(C_KM/np.interp(zq,zf,Hf))/rd
        pM=(dm-vM)/eM; pH=(dh-vH)/eH
        pulls+=[pM,pH]; labels+=[(zq,'DM'),(zq,'DH')]
        d2=np.array([dm-vM,dh-vH]); C=np.array([[eM**2,r*eM*eH],[r*eM*eH,eH**2]])
        chi_tr.append((zq,float(d2@np.linalg.inv(C)@d2)))
    return np.array(pulls),labels,chi_tr

jL=json.load(open(J_LCDM)); jS=json.load(open(J_SURF)); jV=json.load(open(J_VOL))
HL=H_grid_lcdm(jL['H0'],jL['ombh2'],jL['omch2'],jL['mnu'])
rdL=rd_desi(jL['ombh2']+jL['omch2'],jL['ombh2'])
HS=H_grid_surf(jS['omega_m'],jS['gamma'],jS['z_c']); hS=float(np.interp(1.0,_A,HS))/100
rdS=rd_desi(jS['omega_m']-jS['mnu']/93.14, wb_from_51(jS['omega_m'],jS['mnu'],hS))
HV=H_grid_vol(jV['omega_m'],jV['s']); hV=float(np.interp(1.0,_A,HV))/100
rdV=rd_desi(jV['omega_m']-jV['mnu']/93.14, wb_from_51(jV['omega_m'],jV['mnu'],hV))

pL,lab,cL=analyze(HL,rdL); pS,_,cS=analyze(HS,rdS); pV,_,cV=analyze(HV,rdV)
print("SELF-VALIDATION (correlated chi2_BAO vs your JSONs):")
for name,c,j in [('LCDM',cL,jL),('SURF',cS,jS),('VOL',cV,jV)]:
    tot=sum(x[1] for x in c)
    print(f"  {name:5}: recomputed {tot:7.2f}  vs json {j['chi2_BAO']:7.2f}   {'OK' if abs(tot-j['chi2_BAO'])<0.15 else 'MISMATCH!'}")

print(f"\n{'z':>6} {'obs':>4} {'LCDM':>7} {'SURF':>7} {'VOL':>7}")
for i,(zq,t) in enumerate(lab):
    print(f"{zq:6.3f} {t:>4} {pL[i]:+7.2f} {pS[i]:+7.2f} {pV[i]:+7.2f}")
print(f"\nper-tracer correlated chi2:")
print(f"{'z':>6} {'LCDM':>7} {'SURF':>7} {'VOL':>7}")
for i in range(len(cL)):
    print(f"{cL[i][0]:6.3f} {cL[i][1]:7.2f} {cS[i][1]:7.2f} {cV[i][1]:7.2f}")

with open('bao_pulls_v1.csv','w') as f:
    f.write("z,obs,pull_LCDM,pull_SURF,pull_VOL\n")
    for i,(zq,t) in enumerate(lab):
        f.write(f"{zq},{t},{pL[i]:.3f},{pS[i]:.3f},{pV[i]:.3f}\n")

zs=np.array([z for z,_ in lab]); off={'DV':0,'DM':-0.02,'DH':+0.02}
fig,ax=plt.subplots(figsize=(11,6))
for p,name,col,mk in [(pL,'ΛCDM','k','o'),(pS,'surface nucl.','#8e44ad','s'),(pV,'volume nucl. (falsified)','#c0392b','^')]:
    x=[z*(1+off[t]) for (z,t) in lab]
    ax.plot(x,p,mk,color=col,ms=7,label=name,alpha=0.85,ls='')
for y,alpha in [(1,0.15),(2,0.07)]:
    ax.axhspan(-y,y,color='gray',alpha=alpha)
ax.axhline(0,color='k',lw=0.8)
ax.set_xlabel('z'); ax.set_ylabel('pull  (model − data)/σ')
ax.set_title('DESI DR2 BAO pulls — three joint best fits (plik_lite+BAO, τ=P18)')
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('bao_pulls_v1.png',dpi=160,facecolor='white')
print("\nSaved: bao_pulls_v1.png, bao_pulls_v1.csv")
