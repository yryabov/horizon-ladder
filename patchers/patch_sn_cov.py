"""Upgrade any battery *_sn3.py script to the FULL Pantheon+ STAT+SYS covariance -> *_sn4.py.
Usage: python patch_sn_cov.py file1_sn3.py [file2_sn3.py ...]
Requires Pantheon+SH0ES_STAT+SYS.cov at C:\\yaroslavFiles\\sn\\ (same release as the .dat)."""
import sys, ast
NEWBLOCK='# ── SN leg v4: Pantheon+ with FULL STAT+SYS covariance (offset-marginalized) ──\n# NOTE: the printed/JSON "BAO" value = BAO + SN(full covariance, zHD>0.023,\n# calibrators excluded, absolute offset marginalized analytically with C).\nSN_PATH=r\'C:\\yaroslavFiles\\sn\\Pantheon+SH0ES.dat\'\nSN_COV_PATH=r\'C:\\yaroslavFiles\\sn\\Pantheon+SH0ES_STAT+SYS.cov\'\n_sn=np.genfromtxt(SN_PATH,names=True,dtype=None,encoding=None)\n_msn=(_sn[\'IS_CALIBRATOR\']==0)&(_sn[\'zHD\']>=0.023)&(_sn[\'zHD\']<=2.5)\n_snz=np.asarray(_sn[\'zHD\'][_msn],float); _snmu=np.asarray(_sn[\'MU_SH0ES\'][_msn],float)\nwith open(SN_COV_PATH) as _f:\n    _ncov=int(_f.readline().split()[0])\n    _cv=np.fromfile(_f,sep=\' \',count=_ncov*_ncov)\nassert _cv.size==_ncov*_ncov, f"cov read {_cv.size} != {_ncov}^2"\nassert _ncov==len(_sn), f"cov N={_ncov} vs dat rows {len(_sn)} — row orders must match"\n_idx=np.where(_msn)[0]\n_Csn=_cv.reshape(_ncov,_ncov)[np.ix_(_idx,_idx)]\n_Cinv_sn=np.linalg.inv(_Csn)\n_ones=np.ones(len(_idx)); _cs=_Cinv_sn@_ones; _cd=float(_ones@_cs)\nprint(f"SN STAT+SYS covariance: {_ncov}x{_ncov} -> subset {len(_idx)}x{len(_idx)}, inverted OK")\ndef chi2_sn_from_H(Hfn):\n    zf=np.linspace(0.0,2.55,1200); invH=1.0/Hfn(zf)\n    DC=C_KM*np.concatenate([[0.0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])\n    mu_th=5.0*np.log10(np.maximum((1.0+_snz)*np.interp(_snz,zf,DC),1e-9))+25.0\n    r=_snmu-mu_th\n    b=float(_cs@r)\n    return float(r@_Cinv_sn@r)-b*b/_cd\n_chi2_bao_only=chi2_bao_from_H\ndef chi2_bao_from_H(Hfn, rd):\n    return _chi2_bao_only(Hfn, rd)+chi2_sn_from_H(Hfn)\n'
def patch(src):
    i0=src.find("# \u2500\u2500 SN leg")
    if i0<0: return None
    key="return _chi2_bao_only(Hfn, rd)+chi2_sn_from_H(Hfn)"
    i1=src.find(key)
    if i1<i0: return None
    i1=src.find("\n",i1)+1
    return (src[:i0]+NEWBLOCK+src[i1:]).replace("sn3_v1.json","sn4_v1.json")
for t in sys.argv[1:]:
    src=open(t,encoding='utf-8',errors='replace').read()
    out=patch(src)
    if out is None:
        print(f"SKIP {t}: sn3 SN block not found"); continue
    name=t.replace("_sn3.py","_sn4.py")
    open(name,"w",encoding='utf-8').write(out)
    try: ast.parse(out); print(f"OK: {name}")
    except SyntaxError as e: print(f"SYNTAX FAIL {name}: {e}")
