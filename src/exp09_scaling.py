"""09 — scaling: more params/data/compute -> smooth test-loss trends (Kaplan 2020, Hoffmann/Chinchilla 2022).
Sweeps 3 tiny GPT sizes on same Shakespeare slice; fits log-log power law.
Anchors transcript jumps: AlexNet ~60M, GPT-2 largest 1.5B, GPT-3 175B. Counter stops (sizes now secret)."""
import math
import numpy as np
from .common import seed_all, save_json

def train_size(n_embd,n_layer,block,iters,fast):
    import torch, torch.nn as nn
    from .exp06_vectors import get_text
    text=get_text()[:60000]
    chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
    data=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    n=int(0.9*len(data)); tr,va=data[:n],data[n:]; V=len(chars)
    class B(nn.Module):
        def __init__(self):
            super().__init__(); self.ln1=nn.LayerNorm(n_embd)
            self.at=nn.MultiheadAttention(n_embd,2,batch_first=True)
            self.ln2=nn.LayerNorm(n_embd); self.m=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.GELU(),nn.Linear(4*n_embd,n_embd))
            self.register_buffer("mk",torch.triu(torch.ones(block,block),1).bool())
        def forward(self,x):
            T=x.size(1); a,_=self.at(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=self.mk[:T,:T]); x=x+a; return x+self.m(self.ln2(x))
    class G(nn.Module):
        def __init__(self):
            super().__init__(); self.t=nn.Embedding(V,n_embd); self.p=nn.Embedding(block,n_embd)
            self.b=nn.Sequential(*[B() for _ in range(n_layer)]); self.l=nn.LayerNorm(n_embd); self.h=nn.Linear(n_embd,V,bias=False)
        def forward(self,i,t=None):
            x=self.t(i)+self.p(torch.arange(i.size(1))); x=self.b(x); lo=self.h(self.l(x))
            return (lo, nn.functional.cross_entropy(lo.view(-1,V),t.view(-1)) if t is not None else (lo,None))
    m=G(); opt=torch.optim.AdamW(m.parameters(),lr=6e-4)
    for _ in range(iters):
        i=torch.randint(0,len(tr)-block-1,(16,)); xb=torch.stack([tr[j:j+block] for j in i]); yb=torch.stack([tr[j+1:j+block+1] for j in i])
        _,lo=m(xb,yb); opt.zero_grad(); lo.backward(); opt.step()
    with torch.no_grad():
        losses=[]
        for _ in range(10):
            i=torch.randint(0,len(va)-block-1,(8,)); xb=torch.stack([va[j:j+block] for j in i]); yb=torch.stack([va[j+1:j+block+1] for j in i])
            losses.append(float(m(xb,yb)[1]))
        lv=float(np.mean(losses))
    N=sum(p.numel() for p in m.parameters())
    flops=6*N*(iters*16*block)
    return N,lv,flops

def main(fast=False):
    seed_all(0)
    iters=250 if fast else 600; block=32
    cfgs=[(32,2),(64,2),(64,4)]
    rows=[]
    for e,l in cfgs:
        N,lv,fl=train_size(e,l,block,iters,fast)
        rows.append({"n_embd":e,"n_layer":l,"N":N,"val_loss":round(lv,4),"flops":fl})
    xs=np.log([r["N"] for r in rows]); ys=np.log([r["val_loss"] for r in rows])
    slope=float(np.polyfit(xs,ys,1)[0])
    out={"claim":"test loss follows smooth scaling trend; loss!=intelligence",
         "sweep":rows,"power_law_exponent":round(slope,4),
         "anchors":{"alexnet_params":60_000_000,"gpt2_largest":1_500_000_000,"gpt3":175_000_000_000,
                     "note":"same loop predict->measure->nudge; newer sizes secret so counter stops at GPT-3"},
         "hidden_pattern":f"log(val_loss) slope {slope:.3f} <0: bigger tiny-model still wins at fixed tokens; Chinchilla would demand tokens scale with N (~20 tok/param) — our sweep is compute-mismatched on purpose to show smooth trend even off-optimal."}
    assert rows[-1]["val_loss"]<rows[0]["val_loss"] and slope<0
    save_json("09_scaling.json", out); print(out); return out

if __name__=="__main__":
    main()
