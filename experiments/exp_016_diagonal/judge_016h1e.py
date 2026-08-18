"""016-H1e の判定。PREREG_016h1e.md の条文どおりに計算する（教授セッションが直接書いた）。"""
import json, statistics as st, math, pathlib

LEVELS = ["0.45","0.5","0.55","0.6","0.65","0.7"]
LBL = {"0.45":0.45,"0.5":0.50,"0.55":0.55,"0.6":0.60,"0.65":0.65,"0.7":0.70}
NEW = pathlib.Path("outputs/exp_016_diagonal/016h1e")
OLD = pathlib.Path("outputs/exp_016_diagonal/016h1")

def e_in_out(path, key_idx):
    """§3-1: 当該水準の全円弧（20 面連結）の入口/出口 |e_y| の中央値 [mm]。collided も含める。"""
    d = json.load(open(path))
    vals=[]; collided=0
    for r in d["rows"]:
        if r.get("collided"): collided += 1
        for arc in r["entry_exit"]:
            vals.append(abs(arc[key_idx])*1000.0)   # m -> mm
    return st.median(vals), len(vals), collided, d

print("="*78); print("L3 決定性: 健常の再走行 対 保存済み一次記録（entry_exit）"); print("="*78)
det_ok = True
for lv in LEVELS:
    new_med,new_n,_,_ = e_in_out(NEW/f"healthy_sf0.75_v{lv}.json", 0)
    old_med,old_n,_,_ = e_in_out(OLD/f"d1d2_sf0.75_v{lv}.json", 0)
    same = (abs(new_med-old_med) < 1e-9) and (new_n == old_n)
    det_ok &= same
    print(f"  v={LBL[lv]:.2f}: 再走行 {new_med:.6f} mm(n={new_n}) / 保存済 {old_med:.6f} mm(n={old_n}) "
          f"{'🟢 一致' if same else '🔴 不一致'}")
print(f"  → L3 {'合格' if det_ok else '不合格'}")

print(); print("="*78); print("主判定量 q（§3-1）"); print("="*78)
res={}
for grp, folder, pat in (("健常(一次記録)", OLD, "d1d2_sf0.75_v{}.json"), ("凍結", NEW, "frozen_sf0.75_v{}.json")):
    row=[]
    for lv in LEVELS:
        med,n,col,d = e_in_out(folder/pat.format(lv), 0)
        row.append((LBL[lv], med, n, col))
    res[grp]=row
    print(f"\n  --- {grp} E_in(v) [mm] ---")
    for v,med,n,col in row:
        print(f"    v={v:.2f}: {med:.4f}  (連結弧数 {n}, collided {col}/20)")

h = {v:m for v,m,_,_ in res["健常(一次記録)"]}
f = {v:m for v,m,_,_ in res["凍結"]}
num = f[0.70]-f[0.45]; den = h[0.70]-h[0.45]
print(f"\n  分子 E_in^凍結(0.70)-E_in^凍結(0.45) = {f[0.70]:.4f} - {f[0.45]:.4f} = {num:.4f} mm")
print(f"  分母 E_in^健常(0.70)-E_in^健常(0.45) = {h[0.70]:.4f} - {h[0.45]:.4f} = {den:.4f} mm")
if den < 0.20:
    print("  → 分母 < 0.20 mm ⇒ K5（上乗せが再現しない）"); raise SystemExit
q_raw = num/den
q = round(q_raw, 3)
print(f"  q = {q_raw:.6f} → 丸め(小数第3位) q = {q:.3f}")
if q < 0.25:   k = "K1 的中（是正の設計へ）" + ("　※ q<0 = 速度依存が反転" if q < 0 else "")
elif q < 0.60: k = "K2 部分的（残差の機構の同定を続ける）"
elif q <= 1.15:k = "K3 外れ（この経路は使われていない → 第四の容疑へ）"
else:          k = "K4 想定外（凍結で悪化）"
print(f"  ⇒ 判定: {k}")

print(); print("="*78); print("副 W1（模型の量的な当否）: 健常の減衰長 L(v) の回帰の傾き"); print("="*78)
def lslope(pat):
    xs=[];ys=[];info=[]
    for lv in LEVELS:
        d=json.load(open(NEW/pat.format(lv)))
        Ls=[]; vs=[]
        for r in d["rows"]:
            for sgm in r["entry_straights"]:
                if sgm.get("excluded_short") or sgm.get("ratio_undefined"): continue
                Ls.append(sgm["l_seg_m"]); vs.append(sgm["v_act_med"])
        if not Ls: continue
        xs.append(st.median(vs)); ys.append(st.median(Ls))
        info.append((LBL[lv], st.median(vs), st.median(Ls), len(Ls),
                     d["entry_straight_n_total"], d["entry_straight_n_excluded"],
                     d["entry_straight_n_increasing"]))
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    a=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    return a, my-a*mx, info
a,c,info = lslope("healthy_sf0.75_v{}.json")
print(f"  {'v指令':>6} {'v_act中央':>10} {'L中央値[m]':>11} {'採用':>5} {'全体':>5} {'短小除外':>8} {'増加区間':>8}")
for row in info:
    print(f"  {row[0]:6.2f} {row[1]:10.4f} {row[2]:11.4f} {row[3]:5d} {row[4]:5d} {row[5]:8d} {row[6]:8d}")
print(f"  実測 a = {a:.4f} s（横軸は進入直線の v_act 中央値）  切片 c = {c:.4f} m")
print(f"  模型 a = 1.1103 s  |差| = {abs(a-1.1103):.4f} → "
      f"{'🟢 模型は量まで当たっている（±0.33 以内）' if abs(a-1.1103)<=0.33 else '🔴 模型は量としては外れ（主判定は動かさない）'}")

print(); print("="*78); print("母集団の報告（L6・限界 1）"); print("="*78)
for grp,pat in (("健常","healthy_sf0.75_v{}.json"),("凍結","frozen_sf0.75_v{}.json")):
    for lv in LEVELS:
        d=json.load(open(NEW/pat.format(lv)))
        print(f"  {grp} v={LBL[lv]:.2f}: 進入直線 全{d['entry_straight_n_total']} / "
              f"短小除外 {d['entry_straight_n_excluded']}({d['entry_straight_excluded_rate']*100:.1f}%) / "
              f"比不定 {d['entry_straight_n_ratio_undefined']} / 採用 {d['entry_straight_n_usable']} / "
              f"増加区間 {d['entry_straight_n_increasing']}")
