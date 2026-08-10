import sys, json, statistics as st
from collections import deque
import numpy as np
sys.path.insert(0, "/Users/kouhei/tmp/github/micromouse_rl")
from competition import maze_gen_v2 as G

W=H=16
GOAL={(7,7),(7,8),(8,7),(8,8)}

def bfs(v,h):
    d=-np.ones((W,H),int); d[0,0]=0; dq=deque([(0,0)])
    while dq:
        c=dq.popleft()
        for dx,dy in((0,1),(0,-1),(1,0),(-1,0)):
            n=(c[0]+dx,c[1]+dy)
            if 0<=n[0]<W and 0<=n[1]<H and d[n]<0 and G.cells_open(v,h,c,n):
                d[n]=d[c]+1; dq.append(n)
    vals=[d[g] for g in GOAL if d[g]>=0]
    return min(vals) if vals else -1, d

def degs(v,h):
    dd=[]
    for x in range(W):
        for y in range(H):
            k=0
            for dx,dy in((0,1),(0,-1),(1,0),(-1,0)):
                n=(x+dx,y+dy)
                if 0<=n[0]<W and 0<=n[1]<H and G.cells_open(v,h,(x,y),n): k+=1
            dd.append(k)
    return dd

rows=[]
for target in [0,3,5,8,10,15,20,30,40]:
    ds=[];bs=[];de=[];at=[];fails=0
    for seed in range(2000,2040):
        try:
            v,h,info=G.generate_maze(seed, extra_open_target=target, max_attempts=400)
        except RuntimeError:
            fails+=1; continue
        d,_=bfs(v,h)
        ds.append(d); bs.append(info["cycles"]); at.append(info["attempts"])
        dd=degs(v,h); de.append(sum(1 for x in dd if x==1))
    rows.append(dict(target=target,n=len(ds),fail=fails,
        d_med=st.median(ds),d_min=min(ds),d_max=max(ds),
        detour_med=st.median(ds)/14,
        b_med=st.median(bs),b_min=min(bs),b_max=max(bs),
        dead_med=st.median(de),
        att_med=st.median(at),att_max=max(at)))
    print(rows[-1], flush=True)
json.dump(rows, open("/private/tmp/claude-501/-Users-kouhei-tmp-github-micromouse-rl/3ea48c6c-9f45-41ca-8aed-d1c591c0688d/scratchpad/contest_mazes/exp/sweep.json","w"), indent=1)
