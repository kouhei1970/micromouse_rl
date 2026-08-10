import sys, random, numpy as np
from collections import deque
sys.path.insert(0,"/Users/kouhei/tmp/github/micromouse_rl")
from competition import maze_gen_v2 as G
W=H=16; GOAL={(7,7),(7,8),(8,7),(8,8)}
def bfs(v,h):
    d=-np.ones((W,H),int); d[0,0]=0; dq=deque([(0,0)])
    while dq:
        cc=dq.popleft()
        for dx,dy in((0,1),(0,-1),(1,0),(-1,0)):
            n=(cc[0]+dx,cc[1]+dy)
            if 0<=n[0]<W and 0<=n[1]<H and d[n]<0 and G.cells_open(v,h,cc,n):
                d[n]=d[cc]+1; dq.append(n)
    vv=[d[g] for g in GOAL if d[g]>=0]
    return min(vv) if vv else -1
S=[[],[],[],[],[]]
for seed in range(20000,20300):
    rng=random.Random(seed)
    v=np.ones((W+1,H),int); h=np.ones((W,H+1),int)
    G._spanning_tree(rng,v,h); S[0].append(bfs(v,h))
    for e in G.GOAL_INNER: G._set(v,h,e,0)
    for e in G.RING_EDGES: G._set(v,h,e,1)
    gw=rng.choice(G.RING_EDGES); G._set(v,h,gw,0)
    S[1].append(bfs(v,h))
    for e in G.FORCED_OPEN: G._set(v,h,e,0)
    v[0,0]=1;h[0,0]=1;v[1,0]=1;h[0,1]=0
    S[2].append(bfs(v,h))
    pw=(set(G.RING_EDGES)-{gw})|{("v",1,0)}
    internal=[("v",x,y) for x in range(1,W) for y in range(H)]+[("h",x,y) for x in range(W) for y in range(1,H)]
    rng.shuffle(internal); opened=0
    for e in internal:
        if opened>=30: break
        if e in pw or G._get(v,h,e)==0: continue
        G._set(v,h,e,0); k,x,y=e
        posts=((x,y),(x,y+1)) if k=="v" else ((x,y),(x+1,y))
        if any(p!=G.CENTER_POST and not any(G._get(v,h,pe)==1 for pe in G.post_walls(*p)) for p in posts):
            G._set(v,h,e,1)
        else:
            opened+=1
            if opened==10: S[3].append(bfs(v,h))
    S[4].append(bfs(v,h))
lbl=["① DFS全域木のみ","② +ゴールリング(入口1)","③ +リング島化 強制開放12枚","④ +内壁ランダム除去10枚","⑤ +同 30枚(現行仕様)"]
prev=None
for i,(l,arr) in enumerate(zip(lbl,S)):
    a=np.array(arr)
    s=f"{l}: D med={np.median(a):.0f} (p25={np.percentile(a,25):.0f}, p75={np.percentile(a,75):.0f}, 範囲{a.min()}-{a.max()}) 平均={a.mean():.1f}"
    if prev is not None:
        d=prev-a
        s+=f"  ← 対応差 中央値 -{np.median(d):.0f} / 平均 -{d.mean():.1f}"
    print(s); prev=np.array(arr)
tot=np.array(S[0])-np.array(S[4])
print("①→⑤ 総短縮: 中央値 -%.0f / 平均 -%.1f"%(np.median(tot),tot.mean()))
