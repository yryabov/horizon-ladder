"""Patch any battery script to the DR2 Results IV joint Lya row and _lya4 json names.
Usage: python patch_lya_row.py script1.py [script2.py ...] -> writes script1_lya4.py etc."""
import sys, ast
OLD="(2.330,38.988,0.531,8.632,0.101,-0.431)"
NEW="(2.330,39.32,0.33,8.600,0.066,-0.431)"
for t in sys.argv[1:]:
    src=open(t).read()
    n=src.count(OLD)
    if n!=1:
        print(f"SKIP {t}: old Lya row found {n} times"); continue
    src=src.replace(OLD,NEW).replace("best_v1.json","lya4_v1.json")
    out=t.replace(".py","_lya4.py") if "_v1.py" not in t else t.replace("_v1.py","_lya4.py")
    open(out,"w").write(src)
    try: ast.parse(src); print(f"OK: {out}")
    except SyntaxError as e: print(f"SYNTAX FAIL {out}: {e}")
