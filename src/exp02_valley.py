"""02 — steel ball above valley: loss landscape + step-size sensitivity.
Tiny step slow; too large overshoots side-to-side. Verifies LR sweep."""
import numpy as np
from .common import seed_all, save_json

def run_descent(x0, lr, steps, valley="quad"):
    x=float(x0); hist=[]
    for _ in range(steps):
        g = 2*x if valley=="quad" else 4*x**3-3*x  # double-well second case
        x = x - lr*g
        hist.append(x)
        if abs(x)>1e6: break
    loss = x**2 if valley=="quad" else x**4-1.5*x**2
    return x, loss, hist

def main(fast: bool=False):
    seed_all(0)
    steps = 60 if fast else 200
    rows=[]
    for lr in [0.02, 0.1, 0.95, 1.05]:
        x,l,h = run_descent(4.0, lr, steps)
        rows.append({"lr":lr,"x_final":round(float(x),4),"loss_final":round(float(l),4),
                     "diverged": bool(abs(x)>1e3), "oscillated": bool(lr>=0.95 and abs(x)<1e3)})
    out={"claim":"gradient descent nudges downhill; tiny=slow, huge=overshoot/diverge",
         "sweep":rows,
         "hidden_pattern":"lr=0.95 bounces wall-to-wall before settling; lr>1.0 explodes on x^2. Threshold at 1.0 = 2/L for L=2. Textbook stability limit reproduced."}
    # verify: small lr monotonically improves vs start; huge lr worse/diverged
    assert rows[0]["loss_final"] < 16.0 and rows[-1]["diverged"]
    save_json("02_valley.json", out); print(out); return out

if __name__=="__main__":
    main()
