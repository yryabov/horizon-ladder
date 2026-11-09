"""
kinetics_arbitration_v1 — growth-law arbitration + TDGL exclusion (App. A)
==========================================================================
Paper object: the App.-A paragraph "Kinetic arbitration of the growth laws"
and the open-questions item (ii) of Sec. 6.  THIS SCRIPT IS THE SOURCE of
every number quoted there.

Five growth laws for the covered fraction alpha, each integrated on the SAME
self-consistent background (the Avrami best fit of the baseline battery,
results/surf_avrami_unlocked_bao_ttteee_best_v1.json), normalized to the
common half-filling epoch of the fitted curve, compared with the fitted
Avrami filling over the data window z < 2.6 by the effective Avrami exponent
n_eff (least-squares slope of ln(-ln(1-alpha)) against ln(x-x_c)) and the
RMS deviation in alpha.  Laws (R = ANGULAR domain radius on the screen,
i.e. lengths in units of R_E; alpha = 1 - exp(-C R^2), C fixed by matching
the fitted curve's half-filling epoch):
  L1  dR/dx   = const                  (fixed angular step per e-fold;
                                        physical front speed ∝ H R_E)
  L2  dR/dI   = const                  (constant physical front speed;
                                        I = causal clock, dI = dx/(H R_E))
  L3  dR/dI   ∝ eps                    (undercooling-driven speed;
                                        eps = 1 - T/T_c,  T ∝ 1/R_E)
  L4  R = sqrt(t - t_c)/R_E            (diffusive PHYSICAL length, converted
                                        to angular size on the growing screen)
  L5  R ∝ sqrt(eta - eta_c)            (curvature coarsening at fixed
                                        comoving scale; eta = conformal time)
Uniform (frontless) relaxational TDGL dynamics is tested separately:
  dalpha/dt = -(alpha - alpha_eq(T)) * T_c * eps / kappa     (tau = kappa/(T_c eps)),
  alpha_eq  = s [1-(T/T_c)^4],   T = 1/(2 pi R_E)   (code units),
filling irreversible.  The shape fit determines (kappa, T_c); causally honest
kappa at that T_c destroys the curve.  Pure numpy; ~1 min:
  python kinetics_arbitration_v1.py
Self-check: EXPECT = this script's own reference outputs (also quoted in the
paper); the script WARNs if a rerun drifts beyond TOL.
"""
import numpy as np, json

WR = 2.469e-5*(1+0.2271*3.046)
# baseline-battery Avrami best (results/surf_avrami_unlocked_bao_ttteee_best_v1.json)
P = dict(wb=0.02239572750860066, wc=0.12000048647445696, lam=0.9007154180762847,
         zc=4.006684926154433, h=0.6827481691086863, mnu=0.05886540384248649)
WM = P['wb']+P['wc']+P['mnu']/93.14

EXPECT = {'L1':(2.00,0.000),'L2':(2.66,0.062),'L3':(4.26,0.131),
          'L4':(0.94,0.135),'L5':(1.12,0.112)}
TOL_N, TOL_R = 0.05, 0.01
EXPECT_TDGL = dict(kappa_fit=0.036, rms_fit=0.0101, Tc_fit=0.240,
                   rms_k016=0.308, rms_bcs=0.499)   # informational reference

# ---- self-consistent background ---------------------------------------------
def solve(xmax=13.0, n=4800, n_it=260, mix=0.45):
    XG = np.linspace(np.log(1/2500), xmax, n); DX = XG[1]-XG[0]; AA = np.exp(XG)
    xc = np.log(1/(1+P['zc']))
    al = np.where(XG>xc, 1-np.exp(-((XG-xc)/P['lam'])**2), 0.0)
    def RE_of(H):
        f = 1/(AA*H)
        cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*DX)[::-1],[0]])
        return AA*(cum + 1/(AA[-1]*H[-1]))
    lo,hi = 0.02,2.4
    for _ in range(44):
        mid=(lo+hi)/2
        r = np.full_like(XG, mid)
        for _ in range(n_it):
            H = np.sqrt(WM*AA**-3 + WR*AA**-4 + np.maximum(r,0))
            r = (1-mix)*r + mix*(al/RE_of(H)**2)
        H = np.sqrt(WM*AA**-3+WR*AA**-4+r)
        h = float(np.interp(1.0,AA,H))
        if abs(h-P['h'])<3e-4: break
        lo,hi = (mid,hi) if h<P['h'] else (lo,mid)
    return XG, DX, AA, al, H, RE_of(H), xc

XG, DX, AA, AFIT, H, RE, xc = solve()
m_after = XG > xc
def cumint(f):
    g = np.where(m_after, f, 0.0)
    c = np.concatenate([[0],np.cumsum((g[:-1]+g[1:])/2*DX)])
    return c - np.interp(xc, XG, c)
I   = cumint(1/(H*RE))                 # causal clock
tt  = cumint(1/H)                      # cosmic time since birth
eta = cumint(1/(AA*H))                 # conformal time since birth
XI  = XG - xc                          # e-fold clock
RE_c = float(np.interp(xc, XG, RE))
eps_T = np.clip(1 - RE_c/RE, 0, None)  # undercooling (T ∝ 1/R_E)

laws = {
 'L1': np.where(m_after, XI, 0),
 'L2': np.where(m_after, I, 0),
 'L3': cumint(eps_T/(H*RE)),
 'L4': np.where(m_after, np.sqrt(np.clip(tt,0,None)), 0)/RE,
 'L5': np.where(m_after, np.sqrt(np.clip(eta,0,None)), 0),
}
x_half = xc + P['lam']*np.sqrt(np.log(2.0))
win = (XG > xc) & (XG <= 0.0) & (AA > 1/3.6)     # z < 2.6, past only
def alpha_of(R):
    Rh = float(np.interp(x_half, XG, R))
    return 1 - np.exp(-(np.log(2.0)/Rh**2)*R**2)
def n_eff_of(alpha):
    m = win & (alpha>0.01) & (alpha<0.99) & (XI>1e-3)
    return float(np.polyfit(np.log(XI[m]), np.log(-np.log(1-alpha[m])), 1)[0])

print("== Growth-law arbitration (window z<2.6; half-filling-normalized) ==")
out={}
for tag,R in laws.items():
    aL = alpha_of(R)
    n  = n_eff_of(aL)
    rms = float(np.sqrt(np.mean((aL[win]-AFIT[win])**2)))
    ne,re_ = EXPECT[tag]
    flag = "OK" if (abs(n-ne)<=TOL_N and abs(rms-re_)<=TOL_R) else "WARN: drifted"
    out[tag]=dict(n_eff=round(n,3), rms=round(rms,4))
    print(f"  {tag}: n_eff={n:5.2f} (ref {ne:4.2f})   RMS={rms:.3f} (ref {re_:.3f})   [{flag}]")

# ---- uniform TDGL -------------------------------------------------------------
T = 1/(2*np.pi*RE)
def tdgl(Tc, s, kappa):
    epsl=np.clip(1-T/Tc,0,None); aeq=np.where(T<Tc, s*(1-(T/Tc)**4),0.0)
    a=np.zeros_like(XG)
    for i in range(1,len(XG)):
        if T[i]>=Tc: continue
        a[i]=max(a[i-1]+DX*(Tc*epsl[i]/kappa/H[i])*(aeq[i]-a[i-1]), a[i-1])
    return a
def best_over(kappa, Tc_grid):
    best=(1e9,None,None)
    for Tc in Tc_grid:
        for s in np.linspace(0.85,1.2,15):
            a=tdgl(Tc,s,kappa)
            rms=float(np.sqrt(np.mean((a[win]-AFIT[win])**2)))
            if rms<best[0]: best=(rms,Tc,s)
    return best
print("\n== Uniform TDGL (alpha_eq = s[1-(T/Tc)^4]; tau = kappa/(Tc*eps); T=1/(2 pi R_E)) ==")
rms0,Tc0,s0 = best_over(None, [])[0:3] if False else (None,None,None)
best=(1e9,None,None,None)
for kappa in np.geomspace(0.015,0.12,13):
    r,Tc,s = best_over(kappa, np.linspace(0.12,0.40,29))
    if r<best[0]: best=(r,kappa,Tc,s)
rms_fit,k_fit,Tc_fit,s_fit = best
print(f"  shape fit: kappa={k_fit:.3f}, Tc={Tc_fit:.3f}, s={s_fit:.2f} -> RMS={rms_fit:.4f}")
r016 = best_over(0.16,[Tc_fit])[0]
rbcs = best_over(np.pi/8,[Tc_fit])[0]
print(f"  at that Tc: kappa=0.160 -> RMS={r016:.3f};  kappa=pi/8 (BCS) -> RMS={rbcs:.3f}")
print("  verdict: the shape demands kappa ~ 0.04 (several times below both the")
print("  causal floor and kappa_BCS = pi/8); causally honest kappa destroys the fit.")
out['tdgl']=dict(kappa_fit=round(k_fit,3), Tc_fit=round(Tc_fit,3),
                 rms_fit=round(rms_fit,4), rms_k016=round(r016,3), rms_bcs=round(rbcs,3))
json.dump(out, open('kinetics_arbitration_v1.json','w'), indent=1)
print("\nsaved kinetics_arbitration_v1.json")
