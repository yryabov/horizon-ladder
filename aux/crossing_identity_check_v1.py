"""Crossing-identity check v1 -- Eq. (9) of the paper: for the equilibrium
class (alpha a state function of R_E via T_dS ~ 1/R_E) the w = -1 crossing
sits at alpha = p/(p+2), independent of T_c and s; kinetic forms have no
such identity. Pure numpy; seconds to run."""
import numpy as np
WR = 2.469e-5*(1+0.2271*3.046)
WM = 0.0223960224+0.1200090090+0.0590788467/93.14
H0T = 0.6790651632

def make_grid(xmax=25.0, n=5600):
    xg = np.linspace(np.log(1/2500), xmax, n); return xg, xg[1]-xg[0], np.exp(xg)
def RE_of(H, AA, DX):
    f = 1/(AA*H)
    cum = np.concatenate([np.cumsum((f[::-1][:-1]+f[::-1][1:])/2*DX)[::-1],[0]])
    return AA*(cum + 1/(AA[-1]*H[-1]))
def solve(alpha_fn, n_it=400, mix=0.4):
    XG, DX, AA = make_grid()
    lo, hi = 0.02, 2.4
    for _ in range(44):
        mid = (lo+hi)/2
        r = np.full_like(XG, mid)
        for _ in range(n_it):
            H = np.sqrt(WM*AA**-3 + WR*AA**-4 + np.maximum(r,0))
            RE = RE_of(H, AA, DX)
            al = alpha_fn(XG, H, RE)
            al_inf = max(al[-1], 1e-12)
            new = (al/al_inf)/RE**2
            r = (1-mix)*r + mix*new
        h = float(np.interp(1.0, AA, H))
        if abs(h-H0T) < 3e-4: break
        if h < H0T: lo = mid
        else: hi = mid
    return XG, AA, al, r, H, RE, h

def crossings(XG, al, r, H):
    w = -1 - np.gradient(np.log(np.maximum(r,1e-30)), XG)/3
    Ode = r/H**2; out=[]
    for i in range(1, len(XG)):
        if Ode[i] > 1e-3 and (w[i-1]+1)*(w[i]+1) < 0:
            t = -(w[i-1]+1)/(w[i]-w[i-1])
            out.append((XG[i-1]+t*(XG[i]-XG[i-1]), al[i-1]+t*(al[i]-al[i-1])))
    return out

# Thermal with CORRECT T_dS = 1/(2 pi R_E): tau = Rc/RE, alpha=[1-(Rc/RE)^p]_+
def mk_thermal(Rc, p):
    return lambda XG,H,RE: np.maximum(1-(Rc/np.maximum(RE,1e-30))**p, 0.0)
for Rc,p in [(0.8,4.0),(0.5,4.0),(0.8,2.0),(0.8,0.8),(1.1,6.0)]:
    XG,AA,al,r,H,RE,h = solve(mk_thermal(Rc,p))
    cr = crossings(XG,al,r,H)
    pred = p/(p+2)
    line = "  ".join(f"x={x:+.2f}: alpha={a:.5f}" for x,a in cr) or "none"
    print(f"Thermal Rc={Rc} p={p} (h={h:.4f}): predicted alpha_x=p/(p+2)={pred:.5f} | {line}")
