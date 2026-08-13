"""裁定 R38 の独立確認 — 打ち切り条文の発火を生データから再現する（AUDIT_013 §2）。"""
import json, subprocess, sys
REPO="/Users/kouhei/tmp/github/micromouse_rl"
D="experiments/exp_012_continuous_potential/design.md"
def line(rev, key):
    out=subprocess.run(["git","-C",REPO,"show",f"{rev}:{D}"],capture_output=True,text=True).stdout
    return [l for l in out.splitlines() if key in l]
print("=== R38-1 条文の逐語比較 ===")
for key,reg in (("打ち切りの基準","84facf4"),("**支持**: **3 seed の中央値","2e4ec33")):
    a,b=line(reg,key),line("HEAD",key)
    print(f"  {key[:12]:<14} 登録版({reg}) 対 HEAD: {'一致' if a==b else '🔴 不一致'}")
print("=== R38-2 発火の再現 ===")
for s in (1,2,3):
    rows=json.load(open(f"{REPO}/logs/exp_012_condE_seed{s}/validation_history.json"))
    pts=[(r["total_timesteps"],r["goal_rate"]) for r in rows if r["total_timesteps"]<=1_000_000]
    fire = len(pts)>=10 and all(g<0.05 for _,g in pts)
    print(f"  seed{s}: {len(pts)} 点  最大 {max(g for _,g in pts)}  → {'発火' if fire else '発火せず'}")
