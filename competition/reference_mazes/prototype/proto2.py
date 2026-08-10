import sys, time, random
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
    vals=[d[g] for g in GOAL if d[g]>=0]
    return (min(vals) if vals else -1), d
def deg1(v,h):
    return sum(1 for x in range(W) for y in range(H)
               if sum(1 for dx,dy in((0,1),(0,-1),(1,0),(-1,0))
                      if 0<=x+dx<W and 0<=y+dy<H and G.cells_open(v,h,(x,y),(x+dx,y+dy)))==1)

def gen_pathaware(seed, d_floor_ratio=0.85, d_tree_min=55, extra_target=30, max_attempts=2000):
    """DFS全域木の時点で D0>=d_tree_min を要求し、その後
       『開けても最短距離が d_floor 未満に落ちない壁』だけを開けて閉路を作る。"""
    rng=random.Random(seed)
    for attempt in range(1,max_attempts+1):
        v=np.ones((W+1,H),dtype=int); h=np.ones((W,H+1),dtype=int)
        G._spanning_tree(rng,v,h)
        for e in G.GOAL_INNER: G._set(v,h,e,0)
        for e in G.RING_EDGES: G._set(v,h,e,1)
        gateway=rng.choice(G.RING_EDGES); G._set(v,h,gateway,0)
        for e in G.FORCED_OPEN: G._set(v,h,e,0)
        v[0,0]=1; h[0,0]=1; v[1,0]=1; h[0,1]=0
        protected_open=set(G.FORCED_OPEN)|{gateway}|set(G.GOAL_INNER)|{("h",0,1)}
        protected_wall=(set(G.RING_EDGES)-{gateway})|{("v",1,0)}
        D0,_=bfs(v,h)
        if D0 < d_tree_min: continue
        floor=int(d_floor_ratio*D0)
        internal=[("v",x,y) for x in range(1,W) for y in range(H)]+[("h",x,y) for x in range(W) for y in range(1,H)]
        rng.shuffle(internal); opened=0
        for e in internal:
            if opened>=extra_target: break
            if e in protected_wall or G._get(v,h,e)==0: continue
            G._set(v,h,e,0)
            k,x,y=e
            posts=((x,y),(x,y+1)) if k=="v" else ((x,y),(x+1,y))
            bad=any(p!=G.CENTER_POST and not any(G._get(v,h,pe)==1 for pe in G.post_walls(*p)) for p in posts)
            if bad: G._set(v,h,e,1); continue
            D,_=bfs(v,h)
            if D<floor: G._set(v,h,e,1)
            else: opened+=1
        ok=True
        for (px,py) in G.isolated_posts(v,h):
            cands=[e for e in G.post_walls(px,py) if e not in protected_open]
            if not cands: ok=False; break
            G._set(v,h,rng.choice(cands),1)
        if not ok: continue
        if sum(1 for e in G.RING_EDGES if G._get(v,h,e)==0)!=1: continue
        if not G.all_cells_reachable(v,h): continue
        if G.isolated_posts(v,h): continue
        if any(G._get(v,h,e)==1 for e in G.post_walls(*G.CENTER_POST)): continue
        if G.wall_follow_reaches_goal(v,h,"left") or G.wall_follow_reaches_goal(v,h,"right"): continue
        cyc,oe=G.independent_cycles(v,h)
        D,_=bfs(v,h)
        return v,h,dict(seed=seed,attempts=attempt,cycles=int(cyc),D0=int(D0),D=int(D),opened=opened)
    return None

for ratio,tree_min,tgt in [(0.85,55,30),(0.9,60,30),(0.8,50,40),(1.0,60,30)]:
    t0=time.time(); res=[]; s=9000; used=0
    while len(res)<20 and s<9600:
        r=gen_pathaware(s,ratio,tree_min,tgt,max_attempts=1); used+=1
        if r: res.append(r)
        s+=1
    if len(res)<20: print("insufficient",ratio,tree_min,tgt,len(res)); continue
    D=[r[2]["D"] for r in res]; B=[r[2]["cycles"] for r in res]; O=[r[2]["opened"] for r in res]
    de=[deg1(r[0],r[1]) for r in res]
    print(f"経路保護版 floor={ratio}*D0, tree_min={tree_min}, 目標除去={tgt}: "
          f"D med={int(np.median(D))}({min(D)}-{max(D)}) 迂回率 med={np.median(D)/14:.2f} "
          f"β med={int(np.median(B))}({min(B)}-{max(B)}) 実除去 med={int(np.median(O))} "
          f"行止り med={int(np.median(de))} 消費seed={used} 受理率={20/used:.3f} 所要={time.time()-t0:.1f}s", flush=True)

print("---- 追加: β 目標を変えた掃引（floor=1.0=最短距離を1区画も縮めない） ----")
def npaths(v,h):
    D,d=bfs(v,h)
    cnt=np.zeros((W,H),float); cnt[0,0]=1
    for dist,x,y in sorted(((d[x,y],x,y) for x in range(W) for y in range(H) if d[x,y]>=0)):
        if dist==0: continue
        cnt[x,y]=sum(cnt[x+dx,y+dy] for dx,dy in((0,1),(0,-1),(1,0),(-1,0))
                     if 0<=x+dx<W and 0<=y+dy<H and d[x+dx,y+dy]==dist-1 and G.cells_open(v,h,(x,y),(x+dx,y+dy)))
    return int(sum(cnt[g] for g in GOAL if d[g]==D))
for tree_min,tgt in [(55,10),(55,15),(55,20),(55,25),(45,20),(65,20)]:
    res=[]; s=11000; used=0
    while len(res)<20 and s<12000:
        r=gen_pathaware(s,1.0,tree_min,tgt,max_attempts=1); used+=1
        if r: res.append(r)
        s+=1
    if len(res)<20: print("insufficient",tree_min,tgt,len(res)); continue
    D=[r[2]["D"] for r in res]; B=[r[2]["cycles"] for r in res]; O=[r[2]["opened"] for r in res]
    de=[deg1(r[0],r[1]) for r in res]; NP=[npaths(r[0],r[1]) for r in res]
    print(f"tree_min={tree_min} 目標除去={tgt}: D med={int(np.median(D))}({min(D)}-{max(D)}) 迂回率 {np.median(D)/14:.2f}({min(D)/14:.2f}-{max(D)/14:.2f}) "
          f"β med={int(np.median(B))}({min(B)}-{max(B)}) 実除去 med={int(np.median(O))}({min(O)}-{max(O)}) 行止り med={int(np.median(de))}({min(de)}-{max(de)}) "
          f"最短路本数 med={int(np.median(NP))}({min(NP)}-{max(NP)}) 消費seed={used}", flush=True)
