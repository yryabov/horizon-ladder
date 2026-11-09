"""
nucleation_camb.py — Nucleating dark energy for CAMB (background + PPF table).

Model:  rho_de(a) = smooth_max( C - 2*s/R_E(a), eps ),   C = H_inf^2 + 2*s*H_inf,
        R_E(a) = a * int_a^inf da'/(a'^2 H(a'))   (teleological fixed point),
units: H in 100 km/s/Mpc;  H^2 = wm/a^3 + wr/a^4 + rho_de.
Free parameters of the DE sector: s (surface-tension normalization); wm enters the closure.
h = H(a=1) is DERIVED (do not sample H0; feed the derived value to CAMB).

Smoothing (declared): max(X,0) -> eps*ln(1+exp(X/eps)) (softplus), eps = EPS_FRAC*C; exp tail.
Perturbations: use CAMB DarkEnergyPPF with the w(a) table (w crosses -1); c_s^2 = 1 (PPF default)
— an assumption to be stated in the paper (nucleation microphysics does not yet fix c_s).

Validation targets from the distance-prior pipeline (this file's solver, verified vs Planck sanity):
  at (wm=0.1430, s=0.8730):  h = 0.7483, z_c = 1.97   [theta*-calibrated point]
  at (wm=0.1430, s=0.7388):  h = 0.751,  z_c ~ 2.1    [track-2 normalization]
Cross-check your CAMB-derived 100theta* against your own LCDM zero point, not absolute numbers.
"""
import numpy as np

WG   = 2.469e-5                       # omega_gamma (T_CMB = 2.7255 K)
WR   = WG * (1 + 0.2271 * 3.046)      # + massless-neutrino background (adjust if using mnu)
EPS_FRAC = 0.01

_XG = np.linspace(np.log(1/2500), 13.0, 3200)
_A  = np.exp(_XG)
_DX = _XG[1] - _XG[0]

def solve_nucleation(wm, s, n_iter=160, relax=0.5, eps_frac=EPS_FRAC):
    """Teleological fixed point. Returns dict with a-grid, H(a) [100 km/s/Mpc],
    rho_de(a), h, z_c, w(a) (finite thanks to smoothing)."""
    a = _A
    rho = np.full_like(_XG, 0.45)
    for _ in range(n_iter):
        H = np.sqrt(wm*a**-3 + WR*a**-4 + np.maximum(rho, 0))
        f = 1/(a*H)
        cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*_DX)[::-1], [0]])
        Hinf = H[-1]
        RE = a*(cum + 1/(a[-1]*Hinf))
        C = Hinf**2 + 2*s*Hinf
        X = C - 2*s/RE
        eps = eps_frac*C
        rho = (1-relax)*rho + relax*(eps*np.logaddexp(0, X/eps))       # softplus smooth max
    H = np.sqrt(wm*a**-3 + WR*a**-4 + rho)
    h = float(np.interp(1.0, a, H))
    # w(a) = -1 - (1/3) dln rho / dln a   (finite, large positive near z_c)
    lnr = np.log(rho)
    w = -1 - np.gradient(lnr, _XG)/3
    # z_c: where rho falls to eps-level going back
    zc = np.nan
    thr = 2*eps_frac*(H[-1]**2 + 2*s*H[-1])
    below = rho < thr
    if below.any() and not below.all():
        i = np.where(below)[0][-1]
        zc = float(1/a[i] - 1)
    return dict(a=a.copy(), H=H, rho=rho, h=h, zc=zc, w=w)

def camb_params_nucleation(wm_cdm_plus_b, wb, s, As=2.1e-9, ns=0.965, tau=0.054,
                           a_min_table=4.54e-5, n_table=600, w_cap=3.0):
    """Build CAMBparams with the nucleation DE via PPF w(a) table.
    wm_cdm_plus_b = omega_c + omega_b (total matter incl. baryons, excl. neutrinos)."""
    import camb
    sol = solve_nucleation(wm_cdm_plus_b, s)
    h = sol['h']
    # table on the CAMB side: cover a in [a_min_table, 1]; above z_c the density is eps-frozen
    at = np.geomspace(a_min_table, 1.0, n_table)          # from lna=-10 (session-v7 lesson), INCREASING a
    wt = np.minimum(np.interp(np.log(at), _XG, sol['w']), w_cap)   # cap: rho ~ a^-12 above z_c = numerically tame zero
    pars = camb.set_params(H0=100*h, ombh2=wb, omch2=wm_cdm_plus_b - wb,
                           As=As, ns=ns, tau=tau,
                           dark_energy_model='ppf')
    pars.DarkEnergy.set_w_a_table(at, wt)
    return pars, sol

if __name__ == "__main__":
    for wm, s, tag in [(0.1430, 0.8730, "theta*-calibrated"),
                       (0.1430, 0.7388, "track-2 normalization"),
                       (0.1487, 1.2182, "joint-fit best (distance-prior level)")]:
        r = solve_nucleation(wm, s)
        print(f"{tag:34}: wm={wm:.4f} s={s:.4f} -> h={r['h']:.4f}, z_c={r['zc']:.2f}, "
              f"w(a=1)={np.interp(0.0,_XG,r['w']):+.4f}")
    # export the calibrated-point w(a) table for direct use
    r = solve_nucleation(0.1430, 0.8730)
    at = np.geomspace(1e-3, 1.0, 400)
    wt = np.interp(np.log(at), _XG, r['w'])
    np.savetxt("nucleation_w_a_table_calibrated.csv",
               np.column_stack([at, wt]), delimiter=",",
               header="a,w(a)  [nucleation model, wm=0.1430, s=0.8730, eps_frac=0.01]")
    print("table written: nucleation_w_a_table_calibrated.csv")
