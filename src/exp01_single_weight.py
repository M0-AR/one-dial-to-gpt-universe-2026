"""01 — one brass dial: single weight + 4-param toy neuron (3w+1b).
Maps to transcript: glass pipe, brass dial, bead adds weighted inputs + bias,
threshold switches output on. Rosenblatt 1958.
Verifies: examples shape numbers; loss falls; 4 params counted.
"""
import numpy as np
from .common import seed_all, save_json

def main(fast: bool = False):
    seed_all(0)
    # Toy cat(1) vs fox(0): features = [pointed_ears, fluffy_tail, whiskers]
    # Rules on ears alone fail (fox also has pointed ears) -> need learned weights.
    X = np.array([
        [1,0,1],[1,0,1],[1,1,1],[0,0,1],  # cats
        [1,1,0],[1,1,0],[1,0,0],[0,1,0],  # foxes
    ], dtype=float)
    y = np.array([1,1,1,1,0,0,0,0], dtype=float)
    w = np.array([0.1,-0.1,0.05]); b = 0.0  # 4 parameters
    def sigmoid(z): return 1/(1+np.exp(-z))
    lr = 0.5
    steps = 200 if fast else 800
    losses=[]
    for t in range(steps):
        p = sigmoid(X@w+b)
        eps=1e-9
        loss = -(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)).mean()
        losses.append(float(loss))
        grad_p = (p-y)/len(y)
        grad_z = grad_p*p*(1-p)*0 + (p-y)/len(y)  # BCE-with-logits grad simplified
        # correct grad for sigmoid+BCE: dL/dz = (p-y)/N
        gw = X.T@grad_z; gb = grad_z.sum()
        w -= lr*gw; b -= lr*gb
    p = sigmoid(X@w+b)
    acc = ((p>0.5)==y).mean()
    out = {
        "claim": "single dial + 4-param neuron learns from examples, no hand rules",
        "n_params": int(w.size+1),
        "w": [round(float(v),4) for v in w], "b": round(float(b),4),
        "loss_start": round(losses[0],4), "loss_end": round(losses[-1],4),
        "acc": round(float(acc),4),
        "interpretation": "pointed_ears weight stays small (fox shares it); whiskers+tail disambiguate. Rules->examples trade verified."
    }
    assert out["n_params"]==4 and out["loss_end"]<out["loss_start"] and out["acc"]>=0.875
    save_json("01_single_weight.json", out)
    print(out)
    return out

if __name__=="__main__":
    main()
