import sys, random, numpy as np
from collections import deque
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
    vv=[d[g] for g in GOAL if d[g]>=0]
    return min(vv) if vv else -1
a=[];b=[];c=[]
for seed in range(20000,20300):
    rng=random.Random(seed)
    v=np.ones((W+1,H),int); h=np.ones((W,H+1),int)
    G._spanning_tree(rng,v,h)
    a.append(bfs(v,h))                      # 純粋な全域木
    for e in G.GOAL_INNER: G._set(v,h,e,0)  # ゴール内壁開放(NTF規定)のみ
    b.append(bfs(v,h))
    for e in G.RING_EDGES: G._set(v,h,e,1)
    G._set(v,h,rng.choice(G.RING_EDGES),0)
    for e in G.FORCED_OPEN: G._set(v,h,e,0) # ゴール周囲の強制開放
    v[0,0]=1;h[0,0]=1;v[1,0]=1;h[0,1]=0
    c.append(bfs(v,h))
a=np.array(a);b=np.array(b);c=np.array(c)
print("純DFS全域木        : D med=%d (%d-%d)"%(np.median(a),a.min(),a.max()))
print("+ゴール内壁4枚開放 : D med=%d (%d-%d)"%(np.median(b),b.min(),b.max()))
print("+リング島化(強制開放12枚)+入口1: D med=%d (%d-%d)"%(np.median(c),c.min(),c.max()))
print("強制開放による中央値の短縮:", int(np.median(b)-np.median(c)), " 対応差の中央値:", int(np.median(b-c)))
