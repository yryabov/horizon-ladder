"""
rcorr scan v1 — stability of the verdicts against the assumed Lya joint-row correlation
=======================================================================================
The joint AP+BAO Lyman-alpha row (D_M/rd = 39.32 +- 0.33, D_H/rd = 8.600 +- 0.066 at
z_eff = 2.33, arXiv:2607.27410) is used with the correlation coefficient kept at the
BAO-only DR2 value r_corr = -0.431, pending the published number for the joint pair.
This script quantifies the sensitivity: it rebuilds each model's best-fit background
(from the committed verdict JSONs), computes its (D_M/rd, D_H/rd) prediction at
z = 2.33, and evaluates the row chi^2 = (pM^2 + pH^2 - 2 r pM pH)/(1-r^2) for
r_corr in {0, -0.431, -0.6}, at FIXED best-fit parameters (no refit) -- an upper
bound on the sensitivity, since refitting absorbs part of any shift.
Paper claim backed: inter-model contrasts move by <~1 unit of chi^2 (<~2 for the
excluded late-birth form), far below every decisive gap.
Run:  python rcorr_scan_v1.py  (expects the *_sn3_v1.json verdicts in VERDICT_DIR)
"""
import json, numpy as np

VERDICT_DIR = '.'   # path to the committed verdict JSONs
FILES = {'LCDM':   "lcdm_bao_ttteee_sn3_v1.json",
         'CPL':    "cpl_bao_ttteee_sn3_v1.json",
         'Avrami': "surf_avrami_unlocked_bao_ttteee_sn3_v1.json",
         'Thermal':"surf_thermal_bao_ttteee_sn3_v1.json",
         'Front':  "surf_front_L2_bao_ttteee_sn3_v1.json"}
WR = 2.469e-5*(1+0.2271*3.046)
XG = np.linspace(np.log(1/2500), 13.0, 3200); DX = XG[1]-XG[0]; AA = np.exp(XG)

def gp(f):
    d = json.load(open(f"{VERDICT_DIR}/{f}"))
    return dict(wb=d['ombh2'], wc=d['omch2'], mnu=d['mnu'],
                h=d.get('h_sampled', d.get('H0',0)/100),
                lam=d.get('lam'), zc=d.get('z_c'), w0=d.get('w0'),
                wa=d.get('wa'), Tc=d.get('Tc_code'), u=d.get('u_front'))

def RE_of(H):
    f = 1/(AA*H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*DX)[::-1],[0]])
    return AA*(cum + 1/(AA[-1]*H[-1]))

def H_of(name, p):
    wm = p['wb']+p['wc']+p['mnu']/93.14
    if name=='LCDM':
        return np.sqrt(wm*AA**-3+WR*AA**-4+(p['h']**2-wm-WR))
    if name=='CPL':
        rho = (p['h']**2-wm-WR)*AA**(-3*(1+p['w0']+p['wa']))*np.exp(-3*p['wa']*(1-AA))
        return np.sqrt(wm*AA**-3+WR*AA**-4+rho)
    if name=='Avrami':
        xc = np.log(1/(1+p['zc']))
        al_s = np.where(XG>xc, 1-np.exp(-((XG-xc)/p['lam'])**2), 0.0)
        alpha_of = lambda H: al_s
    elif name=='Front':
        xc = np.log(1/(1+p['zc']))
        def alpha_of(H):
            integ = 1/(H*RE_of(H))
            cum = np.concatenate([[0], np.cumsum((integ[:-1]+integ[1:])/2*DX)])
            I = np.maximum(cum-np.interp(xc,XG,cum),0)*np.where(XG>xc,1,0)
            return 1-np.exp(-np.pi*p['u']**2*I**2)
    else:
        def alpha_of(H):
            T = 1/(2*np.pi*RE_of(H))
            return np.clip(1-(T/p['Tc'])**4, 0, 1)
    lo,hi = 0.02,2.4
    for _ in range(44):
        mid=(lo+hi)/2; r=np.full_like(XG,mid)
        for _ in range(200):
            H=np.sqrt(wm*AA**-3+WR*AA**-4+np.maximum(r,0))
            al=alpha_of(H)
            if name=='Thermal':
                raw=al/RE_of(H)**2
                s=(p['h']**2-wm-WR)/max(float(np.interp(0.0,XG,raw)),1e-30)
                r=0.55*r+0.45*s*raw
            else:
                r=0.55*r+0.45*al/RE_of(H)**2
        H=np.sqrt(wm*AA**-3+WR*AA**-4+r)
        hh=float(np.interp(1.0,AA,H))
        if abs(hh-p['h'])<4e-4: break
        if hh<p['h']: lo=mid
        else: hi=mid
    return H

zL,aL = 2.33, 1/3.33
DMv,sM,DHv,sH = 39.32,0.33,8.600,0.066
C = 299792.458
RS = (0.0,-0.431,-0.6)
print(f"{'model':>8} {'DM/rd':>7} {'DH/rd':>7} {'pM':>6} {'pH':>6} | chi2_row r=0 / -0.431 / -0.6")
rows={}
for name,f in FILES.items():
    p=gp(f); H=H_of(name,p)
    rd=147.05*((p['wb']+p['wc'])/0.1432)**-0.23*(p['wb']/0.02236)**-0.13
    Hp=100*H
    DH=C/float(np.interp(aL,AA,Hp))
    xs=np.linspace(np.log(aL),0.0,800)
    DM=float(np.trapezoid(C/(np.exp(xs)*np.interp(xs,XG,Hp)),xs))
    pM=(DM/rd-DMv)/sM; pH=(DH/rd-DHv)/sH
    ch=lambda r:(pM*pM+pH*pH-2*r*pM*pH)/(1-r*r)
    rows[name]=[ch(r) for r in RS]
    print(f"{name:>8} {DM/rd:7.2f} {DH/rd:7.3f} {pM:+6.2f} {pH:+6.2f} | "
          f" {rows[name][0]:5.2f} / {rows[name][1]:5.2f} / {rows[name][2]:5.2f}")
print("\nDIFFERENTIALS vs LCDM (the quantities the conclusions compare):")
out={}
for name in ('Avrami','CPL','Front','Thermal'):
    d=[rows[name][i]-rows['LCDM'][i] for i in range(3)]
    out[name]=dict(diff_r0=d[0],diff_canon=d[1],diff_r06=d[2],
                   max_shift=max(abs(d[0]-d[1]),abs(d[2]-d[1])))
    print(f"  {name:>8}: {d[0]:+6.2f} / {d[1]:+6.2f} / {d[2]:+6.2f}"
          f"   max shift vs canon: {out[name]['max_shift']:.2f}")
json.dump(out, open('rcorr_scan_v1.json','w'), indent=1)
print("\nsaved rcorr_scan_v1.json  (fixed-parameter scan = upper bound on sensitivity)")
