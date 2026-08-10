import sys, glob, numpy as np
sys.path.insert(0,"/Users/kouhei/tmp/github/micromouse_rl")
from competition import maze_gen_v2 as G
print("FORCED_OPEN 枚数:", len(G.FORCED_OPEN))
print(sorted(G.FORCED_OPEN))
W=H=16
def opencnt(v,h,cells):
    tot=0;n=0
    for (x,y) in cells:
        for dx,dy in((0,1),(0,-1),(1,0),(-1,0)):
            nx,ny=x+dx,y+dy
            if 0<=nx<W and 0<=ny<H:
                tot+= 1 if G.cells_open(v,h,(x,y),(nx,ny)) else 0
        n+=1
    return tot/n
ring_cells=[(x,y) for x in range(6,10) for y in range(6,10) if (x,y) not in {(7,7),(7,8),(8,7),(8,8)}]
for tag,pat in [("eval","/Users/kouhei/tmp/github/micromouse_rl/competition/mazes/eval/maze_1*.npz"),
                ("contest","/private/tmp/claude-501/-Users-kouhei-tmp-github-micromouse-rl/3ea48c6c-9f45-41ca-8aed-d1c591c0688d/scratchpad/contest_mazes/contest/*.npz")]:
    vals=[];allv=[]
    for f in sorted(glob.glob(pat)):
        d=np.load(f, allow_pickle=True); v=d["v_walls"]; h=d["h_walls"]
        vals.append(opencnt(v,h,ring_cells))
        allv.append(opencnt(v,h,[(x,y) for x in range(W) for y in range(H)]))
    print(tag, "n=",len(vals), "ゴール周囲12区画の平均次数 med=%.2f (%.2f-%.2f)"%(np.median(vals),min(vals),max(vals)),
          " 盤面全体平均次数 med=%.2f"%np.median(allv))
