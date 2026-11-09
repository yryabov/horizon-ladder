"""
Low-ell forecast v1 — exact D_ell(2..29) for the four TTTEEE best-fit models
============================================================================
Computes the ell<30 TT spectrum (the zone OUTSIDE the fit battery, which starts at
ell=30) for LCDM / CPL / Avrami / Thermal at their TTTEEE+DESI best-fit points, and
prints model/LCDM ratios — the exact version (incl. SW x ISW cross term) of the
sandbox ISW audit (notes sec. 15). Pure CAMB; no plik files, no BAO block needed.

Run:  python lowl_forecast_v1.py
"""
import numpy as np
import camb

WR = 2.469e-5*(1+0.2271*3.046)

# ── TTTEEE+DESI best fits (from the four *_ttteee_best_v1.json) ──────────────
LCDM = dict(H0=68.40318, ombh2=0.02254418, omch2=0.11771886, mnu=0.05818786,
            As=2.12687907e-9, ns=0.96999144, tau=0.06311503)
CPL  = dict(H0=66.07241, ombh2=0.02243216, omch2=0.11934107, mnu=0.06236214,
            As=2.13037164e-9, ns=0.96609117, tau=0.06214467, w0=-0.70769403, wa=-0.91278420)
AVR  = dict(h=0.68274817, ombh2=0.02239573, omch2=0.12000049, mnu=0.05886540,
            As=2.13311996e-9, ns=0.96463008, tau=0.06209613, lam=0.90071542, zc=4.00668493)
THER = dict(h=0.67947180, ombh2=0.02229421, omch2=0.12133491, mnu=0.05888020,
            As=2.13583746e-9, ns=0.96151634, tau=0.06136354, Tc=0.13750400)

XG = np.linspace(np.log(1/2500), 13.0, 3200); DX = XG[1]-XG[0]; AA = np.exp(XG)

def RE_of(H):
    f = 1/(AA*H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*DX)[::-1],[0]])
    return AA*(cum + 1/(AA[-1]*H[-1]))

def w_of_rho(rho):
    w = -1 - np.gradient(np.log(np.maximum(rho,1e-30)), XG)/3
    return np.clip(w, -3.0, -1/3)

def avrami_w_table(p):
    wm = p['ombh2']+p['omch2']+p['mnu']/93.14
    xc = np.log(1/(1+p['zc']))
    al = np.where(XG>xc, 1-np.exp(-((XG-xc)/p['lam'])**2), 0.0)
    lo,hi = 0.02,2.4
    for _ in range(44):
        mid=(lo+hi)/2; r=np.full_like(XG,mid)
        for _ in range(220):
            H=np.sqrt(wm*AA**-3+WR*AA**-4+np.maximum(r,0))
            r=0.55*r+0.45*al/RE_of(H)**2
        H=np.sqrt(wm*AA**-3+WR*AA**-4+r)
        h=float(np.interp(1.0,AA,H))
        if abs(h-p['h'])<3e-4: break
        if h<p['h']: lo=mid
        else: hi=mid
    return w_of_rho(r)

def thermal_w_table(p):
    wm = p['ombh2']+p['omch2']+p['mnu']/93.14
    H = np.sqrt(wm*AA**-3+WR*AA**-4+max(p['h']**2-wm-WR,1e-8)); r=None
    for _ in range(280):
        RE=RE_of(H); T=1/(2*np.pi*RE)
        al=np.clip(1-(T/p['Tc'])**4,0,1)
        raw=al/RE**2
        s=(p['h']**2-wm-WR)/max(float(np.interp(0.0,XG,raw)),1e-30)
        new=s*raw
        r=new if r is None else 0.5*r+0.5*new
        H=np.sqrt(wm*AA**-3+WR*AA**-4+np.maximum(r,0))
    return w_of_rho(r)

def run_camb(p, wtab=None, w0wa=None):
    pars = camb.CAMBparams()
    H0 = p.get('H0', 100*p.get('h', 0))
    pars.set_cosmology(H0=H0, ombh2=p['ombh2'], omch2=p['omch2'], mnu=p['mnu'], tau=p['tau'])
    pars.InitPower.set_params(As=p['As'], ns=p['ns'])
    pars.set_for_lmax(400, lens_potential_accuracy=1)
    if w0wa is not None:
        de = camb.dark_energy.DarkEnergyPPF(); de.set_params(w=w0wa[0], wa=w0wa[1])
        pars.DarkEnergy = de
    elif wtab is not None:
        de = camb.dark_energy.DarkEnergyPPF()
        m = AA <= 1.0
        au, iu = np.unique(AA[m], return_index=True)
        wt = wtab[m][iu]
        if au[-1] < 1.0:
            au = np.append(au, 1.0); wt = np.append(wt, wt[-1])
        de.set_w_a_table(au, wt)
        pars.DarkEnergy = de
    res = camb.get_results(pars)
    return res.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]  # D_ell TT

print("running CAMB x4 (low-ell exact)...")
D = {'LCDM':    run_camb(LCDM),
     'CPL':     run_camb(CPL, w0wa=(CPL['w0'], CPL['wa'])),
     'Avrami':  run_camb(AVR,  wtab=avrami_w_table(AVR)),
     'Thermal': run_camb(THER, wtab=thermal_w_table(THER))}

ells = np.arange(2,30)
print(f"\n{'ell':>4} {'LCDM':>9} {'Avrami':>9} {'Thermal':>9} {'CPL':>9} | ratios to LCDM")
for l in ells:
    row = [D[m][l] for m in ('LCDM','Avrami','Thermal','CPL')]
    rat = [D[m][l]/D['LCDM'][l] for m in ('Avrami','Thermal','CPL')]
    print(f"{l:>4} {row[0]:9.1f} {row[1]:9.1f} {row[2]:9.1f} {row[3]:9.1f} |  {rat[0]:.4f} {rat[1]:.4f} {rat[2]:.4f}")
print("\nband-average D_ell(2..29) model/LCDM:")
for m in ('Avrami','Thermal','CPL'):
    r = np.mean([D[m][l]/D['LCDM'][l] for l in ells])
    print(f"  {m:>8}: {r:.4f}")
print("quadrupole D_2 model/LCDM:")
for m in ('Avrami','Thermal','CPL'):
    print(f"  {m:>8}: {D[m][2]/D['LCDM'][2]:.4f}")
print("\n(NB: same As/ns per model as fitted; the ratio isolates the late-ISW + geometry effect at the best-fit points.)")
