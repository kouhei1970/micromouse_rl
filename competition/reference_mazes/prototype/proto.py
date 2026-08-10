import sys, time, statistics as st
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
    return min(d[g] for g in GOAL if d[g]>=0), d
def deg1(v,h):
    c=0
    for x in range(W):
        for y in range(H):
            k=sum(1 for dx,dy in((0,1),(0,-1),(1,0),(-1,0))
                  if 0<=x+dx<W and 0<=y+dy<H and G.cells_open(v,h,(x,y),(x+dx,y+dy)))
            if k==1: c+=1
    return c
def npaths(v,h,d,D):
    cnt=np.zeros((W,H),float); cnt[0,0]=1
    order=sorted(((d[x,y],x,y) for x in range(W) for y in range(H) if d[x,y]>=0))
    for dist,x,y in order:
        if dist==0: continue
        s=0
        for dx,dy in((0,1),(0,-1),(1,0),(-1,0)):
            nx,ny=x+dx,y+dy
            if 0<=nx<W and 0<=ny<H and d[nx,ny]==dist-1 and G.cells_open(v,h,(x,y),(nx,ny)):
                s+=cnt[nx,ny]
        cnt[x,y]=s
    return int(sum(cnt[g] for g in GOAL if d[g]==D))
configs=[("A", 8, 50, 130),("B",15,40,130),("C",20,30,130),("D",5,60,200),("E",12,45,130)]
for name,tgt,lo,hi in configs:
    t0=time.time(); ds=[];bs=[];de=[];npz=[];seeds=0;got=[]
    s=7000
    while len(ds)<20 and s<7000+20000:
        v,h,info=G.generate_maze(s,extra_open_target=tgt); seeds+=1
        D,d=bfs(v,h)
        if lo<=D<=hi:
            ds.append(D); bs.append(info["cycles"]); de.append(deg1(v,h)); npz.append(npaths(v,h,d,D)); got.append(s)
        s+=1
    el=time.time()-t0
    if len(ds)<20:
        print(name,"FAILED to get 20"); continue
    print(f"案{name}: extra_open={tgt}, 受理窓 D∈[{lo},{hi}] → 20面: "
          f"D med={int(np.median(ds))}({min(ds)}-{max(ds)}) 迂回率 med={np.median(ds)/14:.2f}({min(ds)/14:.2f}-{max(ds)/14:.2f}) "
          f"β med={int(np.median(bs))}({min(bs)}-{max(bs)}) 行止り med={int(np.median(de))}({min(de)}-{max(de)}) "
          f"最短路本数 med={int(np.median(npz))}({min(npz)}-{max(npz)}) "
          f"消費seed={seeds} 受理率={20/seeds:.3f} 所要={el:.1f}s", flush=True)
