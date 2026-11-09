"""
gc_ratio_eval_v1 — the derived T_c/T_dS^0 of the equilibrium-class fits
=======================================================================
Paper object: the T_c/T_dS^0 rows of Tables 1B-3B (App. B).

Definition: T = 1/(2 pi R_E) in code units, T_dS^0 = T(z=0) on the model's
own self-consistent background, so T_c/T_dS^0 = T_c * 2 pi R_E(0).  The same
solve yields the derived crossing z_c (where T = T_c) and alpha_0 = f(z=0);
both are cross-checked against the verdict files, so the ratio inherits
their validation.  Background: rho_de = s*max(1-(T/T_c)^p, 0)/R_E^2 relaxed
on a fixed grid in x = ln a with the analytic de Sitter tail, the amplitude
s fixed by the attractor identity (s * alpha_end = 1), the seed bisected to
match the fitted h.  Vectors are committed verbatim from the verdict files
of the released archive.  Pure numpy; ~1 min:
    python gc_ratio_eval_v1.py
"""
import numpy as np, json

WR = 2.469e-5*(1+0.2271*3.046)
TOL_Z, TOL_A = 0.03, 0.01

# tag: Tc_code, p, h, wb, wc, mnu, expect_zc, expect_alpha0  (verdict files)
FITS = {
 '1B thermal': (0.13750399856392803, 4.0, 0.6794718011708125, 0.02229, 0.12133, 0.0582, 2.023, 0.800),
 '1B GC-p'   : (0.15320396336835673, 2.008282676307846, 0.6776898080445066, 0.02241, 0.11972, 0.0611, 2.604, 0.639),
 '2B thermal': (0.1361758612492365, 4.0, 0.6805155205278146, 0.02226, 0.12181, 0.0589, 1.964, 0.789),
 '2B GC-p'   : (0.15634850522820132, 2.0341235001609714, 0.6742472928942289, 0.02244, 0.11940, 0.0605, 2.747, 0.664),
 '3B thermal': (0.1361112798812846, 4.0, 0.6806963877080181, 0.02226, 0.12183, 0.0586, 1.964, 0.788),
 '3B GC-p'   : (0.16408681231342453, 0.8, 0.6796361094280353, 0.02241941635239497, 0.11957986599147097, 0.0638340720783114, 2.973, 0.364),
}

def solve(Tc, pexp, h, WM, xmax=13.0, n=4800, n_it=300, mix=0.4):
    XG = np.linspace(np.log(1/2500), xmax, n); DX = XG[1]-XG[0]; AA = np.exp(XG)
    def RE_of(H):
        f = 1/(AA*H)
        cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*DX)[::-1],[0]])
        return AA*(cum + 1/(AA[-1]*H[-1]))
    def relax(seed):
        r = np.full_like(XG, seed)
        for _ in range(n_it):
            H = np.sqrt(WM*AA**-3 + WR*AA**-4 + np.maximum(r,0))
            RE = RE_of(H)
            T  = 1/(2*np.pi*RE)
            g  = np.clip(1-(T/Tc)**pexp, 0, None)
            with np.errstate(divide='ignore', invalid='ignore'):
                s = 1/g[-1]
                r = (1-mix)*r + mix*np.nan_to_num(s*g/RE**2)
        H = np.sqrt(WM*AA**-3+WR*AA**-4+np.maximum(r,0))
        return H, RE_of(H), s
    lo,hi = 0.02,2.4
    for _ in range(46):
        mid=(lo+hi)/2
        H,RE,s = relax(mid)
        hm = float(np.interp(1.0,AA,H))
        if abs(hm-h)<2e-4: break
        lo,hi = (mid,hi) if hm<h else (lo,mid)
    T = 1/(2*np.pi*RE)
    g = np.clip(1-(T/Tc)**pexp,0,None)
    alpha0 = float(np.interp(1.0,AA, g))   # bare equilibrium filling (table convention)
    RE0 = float(np.interp(1.0,AA,RE))
    # crossing T = Tc  (RE = 1/(2 pi Tc))
    REc = 1/(2*np.pi*Tc)
    ac  = float(np.interp(REc, RE, AA))      # RE monotone in a
    zc  = 1/ac - 1
    return dict(ratio=Tc*2*np.pi*RE0, zc=zc, alpha0=alpha0, s=float(s), RE0=RE0)

out={}
print(f"{'fit':11s} {'Tc/TdS0':>8s} {'zc(check)':>10s} {'a0(check)':>10s}  flags")
for tag,(Tc,pexp,h,wb,wc,mnu,ezc,ea0) in FITS.items():
    WM = wb+wc+mnu/93.14
    r = solve(Tc,pexp,h,WM)
    f1 = "zc OK" if abs(r['zc']-ezc)<=TOL_Z else f"zc WARN ({r['zc']:.3f} vs {ezc})"
    f2 = "a0 OK" if abs(r['alpha0']-ea0)<=TOL_A else f"a0 WARN ({r['alpha0']:.3f} vs {ea0})"
    out[tag]=dict(ratio=round(r['ratio'],3), zc=round(r['zc'],3),
                  alpha0=round(r['alpha0'],3), s=round(r['s'],3))
    print(f"{tag:11s} {r['ratio']:8.3f} {r['zc']:10.3f} {r['alpha0']:10.3f}  {f1}; {f2}")
json.dump(out, open('gc_ratio_eval_v1.json','w'), indent=1)
print("saved gc_ratio_eval_v1.json")
