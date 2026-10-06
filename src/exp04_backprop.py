"""04 — backprop audit: error at output -> gradients for every connection.
Compares analytic backprop vs numerical finite differences. Rumelhart, Hinton & Williams 1986.
Pass criterion: max relative error < 1e-4."""
import numpy as np
from .common import seed_all, save_json

def main(fast=False):
    seed_all(1)
    rng=np.random.RandomState(1)
    X=rng.randn(8,3); y=rng.randn(8,1)
    W1=rng.randn(3,4)*0.7; b1=np.zeros(4); W2=rng.randn(4,1)*0.7; b2=np.zeros(1)
    def loss_fn(W1,b1,W2,b2):
        h=np.tanh(X@W1+b1); out=h@W2+b2; return ((out-y)**2).mean(), (h,out)
    L,(h,out)=loss_fn(W1,b1,W2,b2)
    # analytic backprop
    dOut=2*(out-y)/len(y); dW2=h.T@dOut; db2=dOut.sum(0)
    dh=dOut@W2.T; dz=dh*(1-h**2); dW1=X.T@dz; db1=dz.sum(0)
    ana={"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}
    # numerical
    eps=1e-5; maxrel=0.0; details={}
    for name,P in [("W1",W1),("b1",b1),("W2",W2),("b2",b2)]:
        num=np.zeros_like(P)
        # check subset for speed — pick indices valid for each shape
        if P.ndim==2:
            idxs=[(0,0),(1,0)] if P.shape[1]==1 else [(0,0),(0,1)]
        else:
            idxs=[0,1] if P.size>1 else [0]
        for ix in idxs:
            orig=P[ix]
            P[ix]=orig+eps; Lp,_=loss_fn(W1,b1,W2,b2)
            P[ix]=orig-eps; Lm,_=loss_fn(W1,b1,W2,b2)
            P[ix]=orig; num[ix]=(Lp-Lm)/(2*eps)
        a=ana[name]
        for ix in idxs:
            rel=abs(num[ix]-a[ix])/max(1e-8,abs(num[ix])+abs(a[ix]))
            maxrel=max(maxrel,float(rel)); details[f"{name}{ix}"]={"analytic":float(a[ix]),"numeric":float(num[ix]),"rel":float(rel)}
    out={"claim":"backprop delivers per-connection gradient; optimizer then nudges","max_rel_err":maxrel,"checks":details,
         "hidden_pattern":"tanh saturation: gradients shrink for |z|>2 — deep stacks attenuate without skip/shortcut connections (Lang & Witbrock 1988 two-spirals trick)."}
    assert maxrel<1e-4, maxrel
    save_json("04_backprop.json", out); print(out); return out
if __name__=="__main__":
    main()
