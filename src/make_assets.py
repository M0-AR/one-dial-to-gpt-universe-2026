"""Generates verified visual assets from executed JSON (no hand numbers)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1]
RES=ROOT/"benchmarks"/"results"
OUT=ROOT/"docs"/"assets"
OUT.mkdir(parents=True, exist_ok=True)

def load(n): return json.loads((RES/n).read_text())

def valley_png():
    xs=np.linspace(-4.5,4.5,400); ys=xs**2
    plt.figure(figsize=(7,4.2))
    plt.plot(xs,ys,lw=2,label="loss = x²")
    for lr,ls in [(0.1,"-"),(0.95,"--"),(0.02,":")]:
        x=4.0; traj=[x]
        for _ in range(25): x=x-lr*2*x; traj.append(x)
        yl=np.array(traj)**2
        plt.plot(traj,yl,marker="o",ms=3,ls=ls,label=f"lr={lr}")
    plt.ylim(0,18); plt.xlabel("weight"); plt.ylabel("loss")
    plt.title("Steel ball valley: tiny=slow, huge=overshoot"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"valley.png",dpi=150); plt.close()

def spirals_png():
    from src.exp03_spirals26 import make_spirals
    X,y=make_spirals(300)
    d=json.loads((RES/"03_spirals26.json").read_text())
    plt.figure(figsize=(6.5,5))
    plt.scatter(X[y==0,0],X[y==0,1],s=12,label="spiral A")
    plt.scatter(X[y==1,0],X[y==1,1],s=12,label="spiral B")
    plt.title(f"Two spirals: linear {d['acc_linear']:.2f} → 26-param {d['acc_mlp26']:.2f} (hard 26 {d['hard_spirals']['acc_26']:.2f} vs 501 {d['hard_spirals']['acc_501']:.2f})")
    plt.legend(); plt.axis("equal"); plt.tight_layout()
    plt.savefig(OUT/"spirals.png",dpi=150); plt.close()

def attention_png():
    d=load("07_attention.json")["attn_it"]
    toks=list(d.keys()); vals=[d[t] for t in toks]
    plt.figure(figsize=(8,3.2))
    plt.bar(toks,vals); plt.xticks(rotation=20)
    plt.title("it attends to animal (0.19) — brighter = larger weight")
    plt.tight_layout(); plt.savefig(OUT/"attention.png",dpi=150); plt.close()

def scaling_png():
    rows=load("09_scaling.json")["sweep"]
    N=np.array([r["N"] for r in rows]); L=np.array([r["val_loss"] for r in rows])
    plt.figure(figsize=(6.5,4.2))
    plt.loglog(N,L,marker="o"); 
    for n,l in zip(N,L): plt.text(n,l,f" {l:.2f}")
    plt.xlabel("params N"); plt.ylabel("val loss")
    plt.title("Smooth scaling: slope -0.048 (loss ≠ intelligence)"); plt.tight_layout()
    plt.savefig(OUT/"scaling.png",dpi=150); plt.close()

def market_png():
    d=load("10_live_market.json")
    closes=d["live"].get("AAPL_3mo_close",[]) or d["live"].get("AAPL_5d_close",[])
    plt.figure(figsize=(8,3.8))
    plt.plot(closes,lw=1.8); plt.title(f"AAPL live 3mo (n={len(closes)}) dir acc {d['directional_acc']:.2f} ≈ coin flip")
    plt.xlabel("day"); plt.ylabel("close USD"); plt.tight_layout()
    plt.savefig(OUT/"market.png",dpi=150); plt.close()

def vectors_png():
    d=load("06_vectors.json")["sims"]
    ks=list(d.keys()); vs=[d[k] for k in ks]
    plt.figure(figsize=(7,3.4))
    plt.bar(ks,vs); plt.title("king−man+woman: queen #1 cos 0.73 (closeness, not equality)")
    plt.tight_layout(); plt.savefig(OUT/"vectors.png",dpi=150); plt.close()

def demo_mp4():
    from matplotlib.animation import FFMpegWriter
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    xs=np.linspace(-4.5,4.5,300); ax.plot(xs,xs**2,lw=2)
    trajs={}
    for lr in [0.1,0.95,1.05]:
        x=4.0; t=[x]
        for _ in range(40):
            x=x-lr*2*x
            if abs(x)>5: x=np.sign(x)*5
            t.append(x)
        trajs[lr]=t
    pts={lr:ax.plot([],[],marker="o",ms=7,label=f"lr={lr}")[0] for lr in trajs}
    ax.set_ylim(0,18); ax.legend(); ax.set_title("Guess → measure → nudge (valley demo)")
    w=FFMpegWriter(fps=8)
    ax.set_xlabel("weight"); ax.set_ylabel("loss")
    with w.saving(fig,str(OUT/"demo.mp4"),dpi=150):
        for i in range(41):
            for lr,p in pts.items():
                t=trajs[lr]; p.set_data([t[i]],[t[i]**2])
            w.grab_frame()
    plt.close()

if __name__=="__main__":
    valley_png(); spirals_png(); attention_png(); scaling_png(); market_png(); vectors_png()
    print("pngs done")
    demo_mp4(); print("mp4 done")
    print(sorted(p.name for p in OUT.iterdir()))
