"""Add the SN leg (Pantheon+) to any battery *_lya4.py script -> *_sn3.py.
Usage: python patch_sn_leg.py file1_lya4.py [file2_lya4.py ...]
After the patch the "BAO" number = BAO + SN (offset-marginalized, diag, z>0.023)."""
import sys, ast
BLOCK='\n# ── SN leg (Pantheon+; folded into the BAO number via wrapper) ──────────────\n# NOTE: after this patch the printed/JSON "BAO" value = BAO + SN(Pantheon+,\n# offset-marginalized, diagonal errors, zHD>0.023, calibrators excluded).\nSN_PATH=r\'C:\\yaroslavFiles\\sn\\Pantheon+SH0ES.dat\'\n_sn=np.genfromtxt(SN_PATH,names=True,dtype=None,encoding=None)\n_msn=(_sn[\'IS_CALIBRATOR\']==0)&(_sn[\'zHD\']>=0.023)&(_sn[\'zHD\']<=2.5)\n_snz=np.asarray(_sn[\'zHD\'][_msn],float); _snmu=np.asarray(_sn[\'MU_SH0ES\'][_msn],float)\n_snw=1.0/np.asarray(_sn[\'MU_SH0ES_ERR_DIAG\'][_msn],float)**2\ndef chi2_sn_from_H(Hfn):\n    zf=np.linspace(0.0,2.55,1200); invH=1.0/Hfn(zf)\n    DC=C_KM*np.concatenate([[0.0],np.cumsum((invH[:-1]+invH[1:])/2*np.diff(zf))])\n    mu_th=5.0*np.log10(np.maximum((1.0+_snz)*np.interp(_snz,zf,DC),1e-9))+25.0\n    r=_snmu-mu_th; off=np.sum(_snw*r)/np.sum(_snw)\n    return float(np.sum(_snw*(r-off)**2))\n_chi2_bao_only=chi2_bao_from_H\ndef chi2_bao_from_H(Hfn, rd):\n    return _chi2_bao_only(Hfn, rd)+chi2_sn_from_H(Hfn)\n\n'
for t in sys.argv[1:]:
    src=open(t,encoding='utf-8',errors='replace').read()
    if src.count("def chi2_full")!=1 or src.count("def chi2_bao_from_H")!=1:
        print(f"SKIP {t}: anchors not found"); continue
    i=src.find("def chi2_full")
    out=src[:i]+BLOCK+src[i:]
    out=out.replace("lya4_v1.json","sn3_v1.json")
    name=t.replace("_lya4.py","_sn3.py")
    open(name,"w",encoding='utf-8').write(out)
    try: ast.parse(out); print(f"OK: {name}")
    except SyntaxError as e: print(f"SYNTAX FAIL {name}: {e}")
