"""orchestrator: runs 01..10 sequentially, aggregates summary.json. Verifies every claim by execution."""
import argparse, json, traceback, pathlib
from .common import RESULTS

MODS=[
 ("01_single_weight","src.exp01_single_weight"),
 ("02_valley","src.exp02_valley"),
 ("03_spirals26","src.exp03_spirals26"),
 ("04_backprop","src.exp04_backprop"),
 ("05_cnn","src.exp05_cnn"),
 ("06_vectors","src.exp06_vectors"),
 ("07_attention","src.exp07_attention"),
 ("08_next_token","src.exp08_next_token"),
 ("09_scaling","src.exp09_scaling"),
 ("10_live_market","src.exp10_live_market"),
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fast",action="store_true"); a=ap.parse_args()
    import importlib
    summary={"fast":a.fast,"experiments":{},"status":"ok"}
    for tag,mod in MODS:
        try:
            m=importlib.import_module(mod); out=m.main(fast=a.fast)
            summary["experiments"][tag]={"ok":True,"keys":list(out.keys())[:8]}
            print(f"[PASS] {tag}")
        except Exception as e:
            summary["experiments"][tag]={"ok":False,"error":str(e)[:500],"trace":traceback.format_exc()[-2000:]}
            summary["status"]="partial_fail"; print(f"[FAIL] {tag}: {e}")
    (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2))
    print("== summary ==", summary["status"])
    return summary

if __name__=="__main__":
    main()
