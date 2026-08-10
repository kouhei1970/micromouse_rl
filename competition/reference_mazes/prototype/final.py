exec(open("proto2.py").read().split("for ratio,tree_min,tgt")[0])
def npaths(v,h):
    D,d=bfs(v,h); cnt=np.zeros((W,H),float); cnt[0,0]=1
    for dist,x,y in sorted(((d[x,y],x,y) for x in range(W) for y in range(H) if d[x,y]>=0)):
        if dist==0: continue
        cnt[x,y]=sum(cnt[x+dx,y+dy] for dx,dy in((0,1),(0,-1),(1,0),(-1,0))
                     if 0<=x+dx<W and 0<=y+dy<H and d[x+dx,y+dy]==dist-1 and G.cells_open(v,h,(x,y),(x+dx,y+dy)))
    return int(sum(cnt[g] for g in GOAL if d[g]==D))
for tree_min,tgt in [(45,15),(50,15),(45,20)]:
    res=[];s=13000;used=0
    while len(res)<20 and s<14000:
        r=gen_pathaware(s,1.0,tree_min,tgt,max_attempts=1); used+=1
        if r: res.append(r)
        s+=1
    D=[r[2]["D"] for r in res]; B=[r[2]["cycles"] for r in res]
    de=[deg1(r[0],r[1]) for r in res]; NP=[npaths(r[0],r[1]) for r in res]
    # 6項目監査
    ok=0
    for v,h,_ in res:
        c1=sum(1 for e in G.RING_EDGES if G._get(v,h,e)==0)==1
        c2=not (G.wall_follow_reaches_goal(v,h,"left") or G.wall_follow_reaches_goal(v,h,"right"))
        c3=G.independent_cycles(v,h)[0]>0
        c4=len(G.isolated_posts(v,h))==0
        c5=bool((v[0,:]==1).all() and (v[16,:]==1).all() and (h[:,0]==1).all() and (h[:,16]==1).all())
        c6=(v[0,0]==1 and h[0,0]==1 and v[1,0]==1 and h[0,1]==0)
        ok+= all([c1,c2,c3,c4,c5,c6])
    print(f"[推奨] tree_min={tree_min} 除去{tgt}枚(経路保護): D med={int(np.median(D))}({min(D)}-{max(D)}) "
          f"迂回率 {np.median(D)/14:.2f}({min(D)/14:.2f}-{max(D)/14:.2f}) β med={int(np.median(B))}({min(B)}-{max(B)}) "
          f"行止り {int(np.median(de))}({min(de)}-{max(de)}) 最短路本数 {int(np.median(NP))}({min(NP)}-{max(NP)}) "
          f"消費seed={used} 6項目全適合={ok}/20", flush=True)
