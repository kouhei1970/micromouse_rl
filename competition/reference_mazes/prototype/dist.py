import sys, statistics as st
from collections import deque
import numpy as np
sys.path.insert(0,"/Users/kouhei/tmp/github/micromouse_rl")
from competition import maze_gen_v2 as G
W=H=16; GOAL={(7,7),(7,8),(8,7),(8,8)}
def bfs(v,h):
    d=-np.ones((W,H),int); d[0,0]=0; dq=deque([(0,0)])
    while dq:
        c=dq.popleft()
        for dx,dy in((0,1),(0,-1),(1,0),(-1,0)):
            n=(c[0]+dx,c[1]+dy)
            if 0<=n[0]<W and 0<=n[1]<H and d[n]<0 and G.cells_open(v,h,c,n):
                d[n]=d[c]+1; dq.append(n)
    return min(d[g] for g in GOAL if d[g]>=0)
for target in [0,5,8,10,15]:
    ds=[];bs=[]
    for seed in range(5000,5300):
        v,h,info=G.generate_maze(seed,extra_open_target=target)
        ds.append(bfs(v,h)); bs.append(info["cycles"])
    ds=np.array(ds)
    q=lambda p: int(np.percentile(ds,p))
    print(f"target={target} n={len(ds)} D med={int(np.median(ds))} p25={q(25)} p75={q(75)} p90={q(90)} max={ds.max()} "
          f"P(D>=40)={np.mean(ds>=40):.3f} P(D>=50)={np.mean(ds>=50):.3f} P(D>=60)={np.mean(ds>=60):.3f} P(D>=70)={np.mean(ds>=70):.3f} "
          f"beta med={int(np.median(bs))}", flush=True)
