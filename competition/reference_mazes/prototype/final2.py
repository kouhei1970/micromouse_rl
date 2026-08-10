exec(open("proto2.py").read().split("for ratio,tree_min,tgt")[0])
import numpy as np
def gen_win(seed, lo, hi, tgt):
    r=gen_pathaware(seed,1.0,lo,tgt,max_attempts=1)
    if r and r[2]["D"]<=hi: return r
    return None
for lo,hi,tgt in [(45,110,15),(40,100,15),(45,110,20)]:
    for block in [13000,15000,17000]:
        res=[];s=block;used=0
        while len(res)<20 and s<block+2000:
            r=gen_win(s,lo,hi,tgt); used+=1
            if r: res.append(r)
            s+=1
        D=[x[2]["D"] for x in res]; B=[x[2]["cycles"] for x in res]
        print(f"窓[{lo},{hi}] 除去{tgt}: block{block} D med={int(np.median(D))}({min(D)}-{max(D)}) 迂回率{np.median(D)/14:.2f} β med={int(np.median(B))}({min(B)}-{max(B)}) 消費seed={used}", flush=True)
