#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-F0 診断 D0 — **速度ループの超過を層に分解して測る**（実装は 1 行も変えない）。

カード `card_016f0.md` §0 で分かっているのは「速度計画（参照）も指令も正しく、
**行き過ぎているのは実速度だけ**」までである。**どの層で行き過ぎているのかは未特定**。
本スクリプトは**測るだけ**で、制御には一切触れない（`SegSpeedPolicy` を継承して
`_wheel_targets_to_voltage` を**同じ値を返すまま記録つきに包む**だけ）。

--------------------------------------------------------------------------
層の分解（**定義上の恒等式**。r = 車輪半径）
--------------------------------------------------------------------------
    v_act − v_cmd = [ v_act − r·(ω_L_act + ω_R_act)/2 ]   … 層K（運動学・滑り・実効半径）
                  + [ r·(ω_L_act + ω_R_act)/2 − v_cmd ]   … 層W（車輪ループの追従誤差）

**恒等式なので残差は丸め誤差だけのはず**であり、これ自体は検査にならない。
実際に検査になるのは**指令側の運動学**:

    r·(ω_L_des + ω_R_des)/2 − v_cmd  ≡ 0        … 監査 A（指令の変換に誤りが無いこと）

--------------------------------------------------------------------------
反証形式の 3 仮説（**結論を先に置かない**。どの観測がどう見えたら偽かを先に書く）
--------------------------------------------------------------------------
| 仮説 | 真なら見えるもの | 偽なら |
|---|---|---|
| H_ff  | **定常**標本で層W が負に偏る（前置補償が過大＝定常で行き過ぎ） | 定常で層W ≈ 0 |
| H_lag | 超過が**減速**標本に集中し、**定常**標本では消える | 減速と定常で差が無い |
| H_kin | **層K** が定常で非零に偏る（実効半径・滑り） | 層K ≈ 0 |

**標本の層別は指令の時間微分で行う**（結論と同じ量で層別しないこと）:
`plateau` = |d v_cmd/dt| ≤ EPS_A が **HOLD_S 秒continuously**／`accel`／`decel`。

--------------------------------------------------------------------------
設定
--------------------------------------------------------------------------
- **設計帯のみ**（`competition/mazes/design_v4`・20 面）。安全弁 `common.seed_bands` を通す
- 面の選定・経路の切り出し・判定の幾何は **016-C と同一**（`run_016c` から import。R23）
- 梯子も 016-C と同一（`LADDER`）。カード §0 の診断条件（指令 0.60）はこの中に含まれる

使い方:
    .venv/bin/python -u experiments/exp_016_diagonal/run_016f0_diag.py
    .venv/bin/python -u experiments/exp_016_diagonal/run_016f0_diag.py --speeds 0.6
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.seed_bands import (assert_seeds_allowed,  # noqa: E402
                                describe_seeds)
from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import value_field  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

from diag_path import build_diagonal_path  # noqa: E402
from diagonal_model import (DELTA8, DiagonalGridModel, cell_center_node,  # noqa: E402
                            descend)
from geometry import git_rev  # noqa: E402
from route_model import connects_true, load_maze  # noqa: E402
from run_016b import cut_segment, longest_diagonal_run  # noqa: E402
from competition.reference_interp import ReferenceInterpMixin  # noqa: E402
from competition.velocity_loop import VelocityLoopMixin  # noqa: E402
from competition.control_2dof import TwoDofControlMixin  # noqa: E402
from run_016c import LADDER, R_ARC_M, SegSpeedPolicy  # noqa: E402

# 標本の層別（**結論とは独立な量＝指令の時間微分**で切る）
EPS_A = 0.05      # 定常とみなす |d v_cmd/dt| の上限 [m/s^2]
HOLD_S = 0.10     # 定常判定に要する継続時間 [s]

KIND_CODE = {"straight": 0, "arc": 1, "diagonal": 2}
KIND_NAME = {0: "直進", 1: "円弧", 2: "斜め"}
CLASS_NAME = {0: "定常", 1: "加速", 2: "減速"}


class ProbedPolicy(TwoDofControlMixin, ReferenceInterpMixin, VelocityLoopMixin,
                   SegSpeedPolicy):
    """**016-C の方策そのまま**。`_wheel_targets_to_voltage` を包んで記録するだけ。

    親を呼んで返り値をそのまま返すので、**電圧も軌跡も 1 ビットも変わらない**。
    ω_des は親と同じ式で再計算している（記録用。監査 A でこの再計算の妥当性を検査する）。

    **2026-08-14 追記**: F0 の是正後を同じ計装で測れるよう `VelocityLoopMixin` を
    混ぜ込んだ。**既定 `k_acc_ff = 0.0` では混ぜ込みが親へそのまま委譲する**ので、
    **基準スナップショットを取ったときと 1 ビットも変わらない**
    （`tests/test_velocity_loop.py` の test2 が全走行のビット一致で確認済み）。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._probe = None

    def _wheel_targets_to_voltage(self, v_cmd, omega_cmd, obs):
        r, tread = self.wheel_radius, self.tread
        wl_des = v_cmd / r - omega_cmd * tread / (2.0 * r)
        wr_des = v_cmd / r + omega_cmd * tread / (2.0 * r)
        wl_act = float(obs[self._i_wheel])
        wr_act = float(obs[self._i_wheel + 1])
        vl, vr = super()._wheel_targets_to_voltage(v_cmd, omega_cmd, obs)
        self._probe = (v_cmd, omega_cmd, wl_des, wr_des, wl_act, wr_act, vl, vr)
        return vl, vr


def run_one(xml_path, params, nodes, dirs, v_diag, v_walls, h_walls, max_s=40.0,
            single_cap=False, k_acc_ff=0.0, ref_interp=False,
            tau_la=0.0, k_r=0.0):
    """1 面 1 速度を走らせ、**制御周期ごとの生の記録**を返す。判定はしない。"""
    path, kinds, idxs = build_diagonal_path(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml_path), params=params)
    start_cell = (nodes[0][0] // 2, nodes[0][1] // 2)
    heading = math.degrees(math.atan2(DELTA8[dirs[0]][1], DELTA8[dirs[0]][0]))
    sim.full_reset(cell=start_cell, heading_deg=heading)

    if single_cap:
        # **016-B と同一**（経路全体に 1 つの上限）。カード §0 の診断条件を再現するため。
        v_arr = np.full(len(kinds), float(v_diag))
    else:
        v_arr = np.where(kinds == "straight", 1e9, v_diag)   # 016-C と同一（直進は v_cap）
    pol = ProbedPolicy(path, v_arr, k_acc_ff=k_acc_ff, ref_interp=ref_interp,
                       tau_la=tau_la, k_r=k_r)
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))

    px, py = path.x, path.y
    rec, collided = [], False
    for _ in range(int(max_s / params.control_dt)):
        obs = sim.observation()
        # **指令と同時刻の状態**を取る（層の分解は同時刻でないと意味を持たない）
        x, y, yaw = sim.privileged_pose()
        v_act, wz = sim.privileged_velocity()
        pol._probe = None
        vl, vr = pol.act(obs)
        if pol._probe is None:        # IDLE 等、電圧生成を通らなかったティック
            if pol.finished:
                break
            continue
        v_ref = float(path.speed[pol._cursor]) if pol._path is not None else float("nan")
        k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        (v_cmd, omega_cmd, wl_des, wr_des, wl_act, wr_act, vl_r, vr_r) = pol._probe
        rec.append((sim.sim_time, x, y, yaw, v_act, wz, v_ref, v_cmd, omega_cmd,
                    wl_des, wr_des, wl_act, wr_act, vl_r, vr_r,
                    KIND_CODE[str(kinds[k])], float(path.curvature[pol._cursor]),
                    # 016-F の E6 用: **制御が実際に潰している横偏差**（親と同じ式）
                    ((x - float(path.x[pol._cursor])) * (-math.sin(float(path.heading[pol._cursor])))
                     + (y - float(path.y[pol._cursor])) * math.cos(float(path.heading[pol._cursor]))),
                    # D1（016-F0-b）: カーソルの添字と、そこまでの弧長。
                    # **d v_cmd/dt = (dv/ds)·(ds_cursor/dt) の右因子**を測るために要る
                    float(pol._cursor), float(path.s[pol._cursor])))
        out = sim.step_control(vl, vr)
        if out.get("collision"):
            collided = True
            break
        if pol.finished:
            break
    return np.array(rec, dtype=float), collided


COLS = ("t", "x", "y", "yaw", "v_act", "omega_z", "v_ref", "v_cmd", "omega_cmd",
        "wl_des", "wr_des", "wl_act", "wr_act", "volt_l", "volt_r", "kind", "kappa",
        "e_y_cursor", "cursor", "s_cur")
COL = {c: i for i, c in enumerate(COLS)}


def derive(rec, r, dt, voltage_limit):
    """層の分解と標本の層別を計算して返す（**記録は変更しない**）。"""
    v_act = rec[:, COL["v_act"]]
    v_cmd = rec[:, COL["v_cmd"]]
    wl_act, wr_act = rec[:, COL["wl_act"]], rec[:, COL["wr_act"]]
    wl_des, wr_des = rec[:, COL["wl_des"]], rec[:, COL["wr_des"]]

    v_wheel = r * (wl_act + wr_act) / 2.0          # 車輪から復元した機体速度
    layer_k = v_act - v_wheel                      # 層K（運動学・滑り）
    layer_w = v_wheel - v_cmd                      # 層W（車輪ループ）
    e_omega = ((wl_des - wl_act) + (wr_des - wr_act)) / 2.0
    audit_a = r * (wl_des + wr_des) / 2.0 - v_cmd  # 監査 A（恒等的に 0 のはず）
    sat = (np.maximum(np.abs(rec[:, COL["volt_l"]]), np.abs(rec[:, COL["volt_r"]]))
           >= voltage_limit - 1e-9)

    # 標本の層別: 指令の時間微分（後退差分）
    d = np.zeros_like(v_cmd)
    d[1:] = (v_cmd[1:] - v_cmd[:-1]) / dt
    hold = max(int(round(HOLD_S / dt)), 1)
    quiet = np.abs(d) <= EPS_A
    steady = np.zeros_like(quiet)
    run = 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        steady[i] = run >= hold
    cls = np.where(steady, 0, np.where(d > 0.0, 1, 2))

    # ---- D1（016-F0-b）: 参照の時間微分を 2 因子へ分ける ----
    #     d v_cmd/dt = (dv/ds) · (ds_cursor/dt)
    #     **計画が保証しているのは左だけ。右は誰も保証していない**（カード §1）
    s_cur = rec[:, COL["s_cur"]]
    ds_cursor = np.zeros_like(s_cur)
    ds_cursor[1:] = s_cur[1:] - s_cur[:-1]
    ds_travel = np.zeros_like(s_cur)
    ds_travel[1:] = np.hypot(np.diff(rec[:, COL["x"]]), np.diff(rec[:, COL["y"]]))

    # ---- 終端の窓（カード §4-2。**D1 の前に固定した定義**） ----
    #     k_end = min{ k : v_ref[k] = 0 }（無ければ ∞）／W_end = { k : k >= k_end }
    v_ref = rec[:, COL["v_ref"]]
    zero = np.where(v_ref == 0.0)[0]
    k_end = int(zero[0]) if zero.size else len(v_ref)
    in_end = np.arange(len(v_ref)) >= k_end

    # ---- 区分の境目（H_jump の徴候を見るため。±2 ティック） ----
    kind = rec[:, COL["kind"]]
    chg = np.zeros(len(kind), dtype=bool)
    chg[1:] = kind[1:] != kind[:-1]
    near = chg.copy()
    for sh in (1, 2):
        near[sh:] |= chg[:-sh]
        near[:-sh] |= chg[sh:]

    # ---- E8（016-F）: 円弧出口からの弧長。斜め標本で「持ち越し」が減衰するかを見る ----
    s_since_exit = np.full(len(kind), np.nan)
    last_exit = None
    for i in range(len(kind)):
        if i > 0 and kind[i - 1] == KIND_CODE["arc"] and kind[i] != KIND_CODE["arc"]:
            last_exit = s_cur[i]
        if last_exit is not None and kind[i] == KIND_CODE["diagonal"]:
            s_since_exit[i] = s_cur[i] - last_exit

    return dict(layer_k=layer_k, layer_w=layer_w, e_omega=e_omega, audit_a=audit_a,
                sat=sat, cls=cls, excess=v_act - v_cmd, dvdt=d,
                s_since_exit=s_since_exit,
                excess_ref=v_act - v_ref, ds_cursor=ds_cursor, ds_travel=ds_travel,
                k_end=k_end, in_end=in_end, near_boundary=near)


def _stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), med=float(np.median(v)), mean=float(np.mean(v)),
                p95=float(np.percentile(v, 95)), max=float(np.max(v)),
                min=float(np.min(v)))


def summarize(rec, der):
    """(kind, class) ごとの要約。**中央値と最大値の両方を出す**（G1 は中央値・G2 は最大値）。"""
    out = {}
    for kc, kname in KIND_NAME.items():
        for cc, cname in CLASS_NAME.items():
            m = (rec[:, COL["kind"]] == kc) & (der["cls"] == cc)
            if not m.any():
                continue
            out[f"{kname}/{cname}"] = dict(
                excess=_stats(der["excess"][m]),
                layer_k=_stats(der["layer_k"][m]),
                layer_w=_stats(der["layer_w"][m]),
                e_omega=_stats(der["e_omega"][m]),
                v_ref=_stats(rec[m, COL["v_ref"]]),
                v_cmd=_stats(rec[m, COL["v_cmd"]]),
                v_act=_stats(rec[m, COL["v_act"]]),
                a_lat=_stats(rec[m, COL["v_act"]] ** 2 * np.abs(rec[m, COL["kappa"]])),
                e_y_abs=_stats(np.abs(rec[m, COL["e_y_cursor"]])),
                sat_rate=float(np.mean(der["sat"][m])),
            )
    return out


def summarize_d1(rec, der, a_max, ds_path):
    """D1（016-F0-b）の量。**窓の定義はカード §4-2 で D1 の前に固定済み。**"""
    n = len(rec)
    dvdt = np.abs(der["dvdt"])
    straight = rec[:, COL["kind"]] == 0
    ex = der["excess_ref"]                       # 実速度 − 参照（G2 / B3 の量）
    body = ~der["in_end"]

    # 停止性能（**判定には使わない参考記録**。カード §4-4）
    k_end, stop = der["k_end"], {}
    if k_end < n:
        va = np.abs(rec[k_end:, COL["v_act"]])
        below = np.where(va < 0.01)[0]
        if below.size:
            j = int(below[0])
            stop = dict(t_stop=float(rec[k_end + j, COL["t"]] - rec[k_end, COL["t"]]),
                        d_stop=float(np.sum(der["ds_travel"][k_end + 1:k_end + j + 1])),
                        v_end_max=float(va.max()), stopped=True)
        else:
            stop = dict(t_stop=None, d_stop=float(np.sum(der["ds_travel"][k_end + 1:])),
                        v_end_max=float(va.max()), stopped=False)

    # ds_cursor は経路点の刻みで数えると読みやすい（1 点 / 2 点の交番が見える）
    mv = der["ds_travel"][1:] > 1e-9
    pts = der["ds_cursor"][1:][mv] / ds_path
    ratio = der["ds_cursor"][1:][mv] / der["ds_travel"][1:][mv]
    nb = der["near_boundary"][1:][mv]

    # **反証テスト（H_quant）**: カーソルが m 点進んだティックの |dv/dt| ÷ a_max。
    # H_quant が真なら m にほぼ比例し、偽なら m によらず一定になる。
    # **終端の窓は除く**（そこは参照が構成上 0 へ落ちる別の機構）
    strat = {}
    mv2 = (der["ds_travel"] > 1e-9) & (~der["in_end"]) & (dvdt > 1e-9)
    mv2[0] = False
    pts_all = np.rint(der["ds_cursor"] / ds_path).astype(int)
    for mm in (0, 1, 2, 3):
        sel = mv2 & (pts_all == mm)
        if sel.sum() >= 3:
            strat[str(mm)] = dict(n=int(sel.sum()),
                                  med=float(np.median(dvdt[sel] / a_max)),
                                  max=float((dvdt[sel] / a_max).max()))

    # ---- E8（016-F）: 斜め標本で |e_y| が円弧出口からの距離に対して減衰するか ----
    diag = (rec[:, COL["kind"]] == KIND_CODE["diagonal"]) & np.isfinite(der["s_since_exit"])
    e8 = {}
    if diag.sum() >= 5:
        xx = der["s_since_exit"][diag]
        yy = np.abs(rec[diag, COL["e_y_cursor"]])
        if np.ptp(xx) > 1e-6:
            sl, ic = np.polyfit(xx, yy, 1)
            r = float(np.corrcoef(xx, yy)[0, 1])
            e8 = dict(n=int(diag.sum()), slope_m_per_m=float(sl), intercept_m=float(ic),
                      corr=r, span_m=float(np.ptp(xx)))

    return dict(
        n_ticks=int(n), k_end=(int(k_end) if k_end < n else None),
        dvdt_by_advance=strat, e8_decay=e8,
        # 参照の時間微分（B1・B2 の量）
        dvdt_p99=float(np.percentile(dvdt, 99)), dvdt_max=float(dvdt.max()),
        dvdt_p99_over_amax=float(np.percentile(dvdt, 99) / a_max),
        # 2 つの量に別の名前を付ける（カード §4-3・R39-3 の族）
        excess_with_end=(float(ex[straight].max()) if straight.any() else None),
        excess_no_end=(float(ex[straight & body].max()) if (straight & body).any() else None),
        # 右因子: カーソルが 1 ティックに進む弧長
        cursor_pts_med=float(np.median(pts)), cursor_pts_p99=float(np.percentile(pts, 99)),
        cursor_pts_max=float(pts.max()),
        cursor_over_travel_med=float(np.median(ratio)),
        cursor_over_travel_p99=float(np.percentile(ratio, 99)),
        # H_jump: 区分の境目とそれ以外で分布が違うか
        cursor_pts_p99_boundary=(float(np.percentile(pts[nb], 99)) if nb.any() else None),
        cursor_pts_p99_elsewhere=(float(np.percentile(pts[~nb], 99)) if (~nb).any() else None),
        stop=stop)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--speeds", nargs="*", type=float, default=list(LADDER))
    ap.add_argument("--k-acc-ff", type=float, default=0.0,
                    help="加速度前置補償の係数（0 = 是正前・1.0 = 物理から導いた全量）")
    ap.add_argument("--ref-interp", action="store_true",
                    help="参照を弧長で内挿して読む（016-F0-b の是正）")
    ap.add_argument("--tau-la", type=float, default=0.0,
                    help="前方注視時間 [s]（016-F）")
    ap.add_argument("--k-r", type=float, default=0.0,
                    help="レートダンピングの係数（016-F）")
    ap.add_argument("--single-cap", action="store_true",
                    help="経路全体に 1 つの速度上限を掛ける（016-B と同一。カード §0 の診断条件）")
    ap.add_argument("--trace-face", default="maze_41038",
                    help="生の記録を npz で残す面（カード §0 の診断面）")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                              / "016f0" / "baseline"))
    args = ap.parse_args()

    seeds = [int(q.stem.split("_")[1])
             for q in sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    a, b = load_time_model()
    params = RobotParams()
    r, dt = params.wheel_radius, params.control_dt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 016-C と同一の面の切り出し
    faces = []
    for f in sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                    key=lambda p: int(p.stem.split("_")[1])):
        v, h, start, goals = load_maze(str(f))
        conn = connects_true(v, h)
        model = DiagonalGridModel(a, b, r=1.0)
        field = value_field([tuple(g) for g in goals], 16, 16, conn, model)
        p = descend(field, model, cell_center_node(tuple(start)), "N", 16, 16, conn)
        s0, e0 = longest_diagonal_run(p["dirs"])
        i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
        xml = f.with_suffix(".xml")
        if not xml.exists():
            build_maze_robot_xml(v, h, str(xml), model_name=f"m_{f.stem}", params=params)
        faces.append(dict(maze=f.stem, xml=xml, nodes=p["nodes"][i:j + 1],
                          dirs=p["dirs"][i:j], v=v, h=h))
    print(f"面: {len(faces)}／速度: {args.speeds}\n")

    # 車輪 PI の電圧上限は方策の既定値（記録のためだけに 1 度読む）
    voltage_limit = ProbedPolicy(build_diagonal_path(faces[0]["nodes"], faces[0]["dirs"],
                                                     params.cell_size, R_ARC_M)[0],
                                 np.array([0.3])).voltage_limit
    _probe0 = ProbedPolicy(build_diagonal_path(faces[0]["nodes"], faces[0]["dirs"],
                                               params.cell_size, R_ARC_M)[0], np.array([0.3]))
    a_max_plan = _probe0.a_max            # = a_max_measured × 安全率（方策から読む）
    ds_path_med = float(np.median(np.diff(
        build_diagonal_path(faces[0]["nodes"], faces[0]["dirs"],
                            params.cell_size, R_ARC_M)[0].s)))
    print(f"計画の a_max = {a_max_plan:.3f} m/s^2／経路点の刻み 中央値 {ds_path_med*1000:.2f} mm")
    if args.k_acc_ff or args.ref_interp:
        print(f"⚠️ k_acc_ff = {args.k_acc_ff}（F0）／ref_interp = {args.ref_interp}（F0-b）\n")

    rows, audit_max = [], 0.0
    for q in faces:
        for v_d in args.speeds:
            rec, collided = run_one(q["xml"], params, q["nodes"], q["dirs"], v_d,
                                    q["v"], q["h"], single_cap=args.single_cap,
                                    k_acc_ff=args.k_acc_ff, ref_interp=args.ref_interp,
                                    tau_la=args.tau_la, k_r=args.k_r)
            if rec.size == 0:
                continue
            der = derive(rec, r, dt, voltage_limit)
            audit_max = max(audit_max, float(np.max(np.abs(der["audit_a"]))))
            rows.append(dict(maze=q["maze"], v_diag=float(v_d), collided=bool(collided),
                             n_ticks=int(rec.shape[0]), by_class=summarize(rec, der),
                             d1=summarize_d1(rec, der, a_max_plan, ds_path_med)))
            if q["maze"] == args.trace_face:
                tag = "_cap1" if args.single_cap else ""
                np.savez_compressed(out_dir / f"trace_{q['maze']}_v{v_d:g}{tag}.npz",
                                    rec=rec, cols=np.array(COLS),
                                    layer_k=der["layer_k"], layer_w=der["layer_w"],
                                    e_omega=der["e_omega"], cls=der["cls"],
                                    dvdt=der["dvdt"], sat=der["sat"])
        print(f"{q['maze']} 完了", flush=True)

    # ---- 全面をまとめた表（面ごとの標本を素直に連結する。面の重みづけはしない） ----
    print("\n【監査 A】 r·(ω_des_L+ω_des_R)/2 − v_cmd の最大絶対値 = "
          f"{audit_max:.3e}  （恒等的に 0 のはず）")

    agg = {}
    for row in rows:
        for key, s in row["by_class"].items():
            k = (row["v_diag"], key)
            agg.setdefault(k, []).append(s)
    print("\n【超過と層の分解】 単位 m/s（med = 中央値・max = 最大値。n = ティック数）")
    print(f"{'v指令':>6} {'区分':<9} {'n':>6} {'超過med':>9} {'超過max':>9} "
          f"{'層W med':>9} {'層K med':>9} {'eω med':>8} {'飽和率':>7}")
    for (v_d, key) in sorted(agg):
        ss = agg[(v_d, key)]
        n = sum(s["excess"]["n"] for s in ss)

        def wm(field, stat):
            vals = [s[field][stat] for s in ss if s[field]]
            return float(np.median(vals)) if vals else float("nan")
        print(f"{v_d:>6.2f} {key:<9} {n:>6} "
              f"{wm('excess','med'):>9.4f} {wm('excess','max'):>9.4f} "
              f"{wm('layer_w','med'):>9.4f} {wm('layer_k','med'):>9.4f} "
              f"{wm('e_omega','med'):>8.3f} "
              f"{float(np.median([s['sat_rate'] for s in ss])):>7.3f}")

    # ---- D1 の表（016-F0-b） ----
    print("\n【D1】参照の時間微分と、その右因子（カーソルが 1 ティックに進む弧長）")
    print(f"{'v指令':>6}{'|dv/dt| p99':>12}{'÷a_max':>8}{'進み 中央値':>12}{'進み p99':>10}"
          f"{'境目 p99':>10}{'境目以外 p99':>13}{'超過(終端込)':>13}{'超過(終端抜)':>13}")
    for v_d in args.speeds:
        rs = [r["d1"] for r in rows if r["v_diag"] == v_d]
        if not rs:
            continue
        def M(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return float(np.median(vals)) if vals else float("nan")
        print(f"{v_d:>6.2f}{M('dvdt_p99'):>12.3f}{M('dvdt_p99_over_amax'):>8.2f}"
              f"{M('cursor_pts_med'):>12.2f}{M('cursor_pts_p99'):>10.2f}"
              f"{M('cursor_pts_p99_boundary'):>10.2f}{M('cursor_pts_p99_elsewhere'):>13.2f}"
              f"{M('excess_with_end'):>13.4f}{M('excess_no_end'):>13.4f}")

    out = dict(git_rev=git_rev(), maze_dir=args.maze_dir, speeds=list(args.speeds),
               a_max_plan=a_max_plan, ds_path_med=ds_path_med,
               single_cap=bool(args.single_cap), k_acc_ff=float(args.k_acc_ff),
               ref_interp=bool(args.ref_interp),
               tau_la=float(args.tau_la), k_r=float(args.k_r),
               eps_a=EPS_A, hold_s=HOLD_S, wheel_radius=r, control_dt=dt,
               voltage_limit=voltage_limit, audit_a_max_abs=audit_max, rows=rows)
    (out_dir / ("d0_diag_cap1.json" if args.single_cap else "d0_diag.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
