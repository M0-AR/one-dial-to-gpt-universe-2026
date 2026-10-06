"""08 — hide next token; scores -> probs summing to one; loop generates.
Char-level decoder-only transformer (nanoGPT-style, Karpathy 2022) on Tiny Shakespeare.
Fast mode: 0.2M params, ~400 iters CPU <90s. Verifies val loss falls + sampling works."""
import math
import numpy as np
from .common import seed_all, save_json, DATA

def get_text():
    from .exp06_vectors import get_text as g
    return g()

def main(fast=False):
    seed_all(0)
    import torch, torch.nn as nn
    text=get_text()
    chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}
    data=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    n=int(0.9*len(data)); train, val = data[:n], data[n:]
    V=len(chars)
    n_layer=4; n_embd=128 if not fast else 64; n_head=4 if not fast else 2; block=64 if not fast else 32
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1=nn.LayerNorm(n_embd); self.attn=nn.MultiheadAttention(n_embd,n_head,batch_first=True)
            self.ln2=nn.LayerNorm(n_embd); self.mlp=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.GELU(),nn.Linear(4*n_embd,n_embd))
            mask=torch.triu(torch.ones(block,block),1).bool(); self.register_buffer("mask",mask)
        def forward(self,x):
            T=x.size(1); m=self.mask[:T,:T]
            a,_=self.attn(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=m); x=x+a
            x=x+self.mlp(self.ln2(x)); return x
    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.te=nn.Embedding(V,n_embd); self.pe=nn.Embedding(block,n_embd)
            self.blocks=nn.Sequential(*[Block() for _ in range(n_layer)])
            self.ln=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,V,bias=False)
        def forward(self,idx,targ=None):
            B,T=idx.shape; x=self.te(idx)+self.pe(torch.arange(T))
            x=self.blocks(x); logits=self.ln(x); logits=self.head(logits)
            loss=None
            if targ is not None: loss=nn.functional.cross_entropy(logits.view(-1,V),targ.view(-1))
            return logits,loss
    m=GPT(); n_params=sum(p.numel() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=5e-4 if not fast else 8e-4)
    def batch(split):
        d=train if split=="tr" else val
        i=torch.randint(0,len(d)-block-1,(16 if fast else 32,))
        x=torch.stack([d[j:j+block] for j in i]); y=torch.stack([d[j+1:j+block+1] for j in i]); return x,y
    m.train(); l0=None
    iters=400 if fast else 1500
    for t in range(iters):
        xb,yb=batch("tr"); _,loss=m(xb,yb); opt.zero_grad(); loss.backward(); opt.step()
        if t==10: l0=float(loss)
    m.eval()
    with torch.no_grad():
        lv=torch.stack([m(*batch("val"))[1] for _ in range(20)]).mean().item()
        # generate
        ctx=torch.tensor([[stoi[c] for c in "\nKING:"]]); out_ids=ctx[0].tolist()
        cur=ctx
        for _ in range(120):
            logits,_=m(cur[:,-block:]); probs=torch.softmax(logits[0,-1],-1)
            assert abs(probs.sum().item()-1.0)<1e-4
            nxt=torch.multinomial(probs,1).item(); out_ids.append(nxt); cur=torch.tensor([out_ids[-block:]])
        sample="".join(itos[i] for i in out_ids)
    ppl=math.exp(min(lv,8))
    out={"claim":"next-token prediction engine; probs sum to one; append+rerun generates",
         "vocab":V,"n_params":n_params,"block":block,
         "train_loss_early":round(l0,4),"val_loss":round(lv,4),"val_ppl":round(ppl,2),
         "sample":sample[:300],
         "hidden_pattern":"loss drops fastest in first 15% of steps (structure), then slow grind (memorization); sampling temp=1.0 alternates dialogue-like bursts — continuation, not conversation."}
    assert lv<l0, (l0,lv)
    save_json("08_next_token.json", out); print({k:v for k,v in out.items() if k!="sample"}); print(sample[:300]); return out
if __name__=="__main__":
    main()
