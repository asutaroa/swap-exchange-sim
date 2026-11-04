"""
swap_sim.py  -  잉여 소재 다자간 순환 교환 시뮬레이션 (합성 데이터)

연구 질문: 장기교환식 순환 청산(2자·3자·4자)이 양자 교환/가격 장터보다
          더 많은 거래를 만들고, 물류비·이탈 위험을 반영해도 참여 유인이 있는가?

실행:  python swap_sim.py            (전체 실험, 약 5~15분)
       python swap_sim.py --quick    (빠른 점검, 1분 내)
       python swap_sim.py --only A,B (일부 실험만)

모든 숫자는 '가정'입니다. 파일 맨 위 DEFAULTS 에서 바꿀 수 있습니다.
데이터는 실측이 아니라 난수로 만든 합성 데이터입니다 (README 의 한계 참고).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds, linear_sum_assignment
from scipy.sparse import coo_matrix

# ----------------------------------------------------------------------------
# 1. 가정(파라미터).  단위: 금액 = 만원, 무게 = kg, 거리 = km
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    N=60,               # 참여 기업 수 (기업마다 잉여 로트 1개 보유)
    T=12,               # 소재 유형 수 (면 우븐, 니트, 데님, 울, 실크 ...)
    d=3.0,              # 기업당 평균 '수용 가능한 로트 수' (수용 그래프 밀도)
    fit=0.6,            # 유형이 맞는 로트 중 색상·품질까지 받아들이는 비율
    median_v=100.0,     # 로트 정가 가치 중앙값
    sigma_v=1.0,        # 로트 가치의 산포 (로그정규)
    sigma_u=0.35,       # 받는 사람의 주관적 가치 / 정가 의 산포 (로그정규)
    theta_lo=0.3,       # 외부 선택지(헐값 처분) = theta * 정가, theta ~ U(lo, hi)
    theta_hi=0.7,
    c_fixed=3.0,        # 건당 고정 비용(포장·검증·핸들링)
    c_km=0.015,         # 50kg 기준 km당 운송비
    cost_scale=1.0,     # 물류비 배율 (민감도 분석용)
    p_fail=0.05,        # 참여자 1곳당 이탈·검증 실패 확률
    fail_cost_share=0.5,  # 실패 시 이미 발생한 물류비 비율(매몰비용)
    credit_discount=0.8,  # 크레딧을 현금 대비 몇 %로 평가하는지
    ask_ratio=1.0,      # 장터에서 판매자 호가 = 정가 * ask_ratio (1.0 이면 정가 호가)
    ask_sigma=0.35,     # 호가의 산포 (가격 발견 실패 정도)
    visibility=1.0,     # 장터에서 구매자가 판매자 로트를 발견할 확률 (1.0=전부 보임, 유리한 가정)
)

# 한국 지리를 흉내 낸 가정 좌표(서울 기준 km) 와 비중
CITIES = {
    "Seoul": ((0, 0), 0.55),      # 동대문 등 수도권 집중
    "Daegu": ((150, -190), 0.20),
    "Busan": ((200, -270), 0.10),
    "Daejeon": ((10, -140), 0.08),
    "Gwangju": ((-60, -250), 0.07),
}


# ----------------------------------------------------------------------------
# 2. 합성 데이터 생성
# ----------------------------------------------------------------------------
def make_instance(rng, P):
    """기업 N곳의 로트, 위치, 수용 집합, 주관적 가치, 물류비를 난수로 생성."""
    N, T = P["N"], P["T"]
    typ = rng.integers(0, T, N)
    v = P["median_v"] * np.exp(rng.normal(0, P["sigma_v"], N))
    theta = rng.uniform(P["theta_lo"], P["theta_hi"], N)
    S = theta * v                                   # 외부 선택지 가치
    weight = 50.0 * (v / P["median_v"]) ** 0.8      # kg

    # 위치
    names = list(CITIES)
    probs = np.array([CITIES[c][1] for c in names])
    city = rng.choice(len(names), N, p=probs / probs.sum())
    base = np.array([CITIES[names[c]][0] for c in city], float)
    pos = base + rng.normal(0, 8, (N, 2))
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    C0 = P["c_fixed"] + P["c_km"] * dist * weight[:, None] / 50.0   # C0[i,j]: i가 j에게 보내는 비용

    # 수용 그래프: A[j,i]=True 이면 j 가 i 의 로트를 받아들임
    p_type = min(1.0, P["d"] / ((N - 1) * (1 - 1.0 / T) * P["fit"]))
    acc_type = rng.random((N, T)) < p_type
    acc_type[np.arange(N), typ] = False             # 자기 유형은 원하지 않음
    A = acc_type[:, typ] & (rng.random((N, N)) < P["fit"])
    np.fill_diagonal(A, False)
    W = v[None, :] * np.exp(rng.normal(0, P["sigma_u"], (N, N)))   # W[j,i]: j 가 보는 i 로트 가치
    return dict(N=N, typ=typ, v=v, S=S, A=A, W=W, C0=C0, C=C0 * P["cost_scale"])


def with_cost_scale(inst, scale):
    out = dict(inst)
    out["C"] = inst["C0"] * scale
    return out


# ----------------------------------------------------------------------------
# 3. 순환 열거 + 청산(정수계획)
# ----------------------------------------------------------------------------
def enumerate_cycles(out_nbrs, N, L):
    """길이 2..L 인 방향 순환을 모두 열거. 가장 작은 번호 노드에서 시작해 중복 제거."""
    cycles = {k: [] for k in range(2, L + 1)}
    for s in range(N):
        stack = [(s, [s])]
        while stack:
            u, path = stack.pop()
            for v in out_nbrs[u]:
                if v == s:
                    if len(path) >= 2:
                        cycles[len(path)].append(tuple(path))
                elif v > s and v not in path and len(path) < L:
                    stack.append((v, path + [v]))
    return {k: np.array(c, dtype=int).reshape(-1, k) for k, c in cycles.items()}


def graph_cycles(inst, L):
    A = inst["A"]
    out_nbrs = [np.nonzero(A[:, i])[0].tolist() for i in range(inst["N"])]  # i 가 로트를 주는 대상 j
    return enumerate_cycles(out_nbrs, inst["N"], L)


def cycle_stats(inst, cyc):
    nodes = cyc
    prev = np.roll(nodes, 1, axis=1)     # 나에게 로트를 주는 사람
    nxt = np.roll(nodes, -1, axis=1)     # 내가 로트를 주는 사람
    w = inst["W"][nodes, prev]
    cc = inst["C"][nodes, nxt]
    s = inst["S"][nodes]
    net = w - s - cc                     # 각 참여자의 순이득
    credit_paid = inst["v"][prev] - inst["v"][nodes]   # 정가 차이만큼 내가 낼 크레딧(+) 또는 받을 크레딧(-)
    return net, cc, credit_paid


def candidates(inst, cycles, L, regime, P):
    """regime: 'ntu'(크레딧 없음, 전원 개별 이득 필요),
               'credit_list'(정가 차이를 크레딧으로 정산),
               'credit_surplus'(순환 총이득>=0 이면 성사, 이득 균등 분배 - 이상적 상한)"""
    keep_nodes, wts, ks, ccs = [], [], [], []
    for k in range(2, L + 1):
        cyc = cycles.get(k)
        if cyc is None or len(cyc) == 0:
            continue
        net, cc, x = cycle_stats(inst, cyc)
        if regime == "ntu":
            ok = (net >= 0).all(1)
            surplus = net.sum(1)
        elif regime == "credit_list":
            dsc = P["credit_discount"]
            net2 = net - np.maximum(x, 0) + dsc * np.maximum(-x, 0)
            ok = (net2 >= 0).all(1)
            surplus = net2.sum(1)
        elif regime == "credit_surplus":
            surplus = net.sum(1)
            ok = surplus >= 0
        else:
            raise ValueError(regime)
        succ = (1 - P["p_fail"]) ** k
        fail_cost = P["fail_cost_share"] * cc.sum(1)
        w = succ * surplus - (1 - succ) * fail_cost       # 기대 순이익
        m = ok & (w > 0)
        keep_nodes.append(cyc[m])
        wts.append(w[m])
        ks.append(np.full(m.sum(), k))
    if not keep_nodes:
        return np.empty((0, 0), int), np.array([]), np.array([], int)
    return keep_nodes, np.concatenate(wts), np.concatenate(ks)


def clear_cycles(inst, cycles, L, regime, P):
    """겹치지 않는 순환 조합 중 기대 순이익 합이 최대인 것을 정수계획으로 선택."""
    N = inst["N"]
    cand_nodes, w, ks = candidates(inst, cycles, L, regime, P)
    res = dict(lots_share=0.0, surplus_per_firm=0.0, planned_share=0.0, share_by_k={2: 0.0, 3: 0.0, 4: 0.0})
    if len(w) == 0:
        return res
    rows, cols, col = [], [], 0
    for arr in cand_nodes:
        for cy in arr:
            rows.extend(cy.tolist())
            cols.extend([col] * len(cy))
            col += 1
    M = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, col)).tocsr()
    sol = milp(c=-w, constraints=LinearConstraint(M, -np.inf, 1), integrality=np.ones(col),
               bounds=Bounds(0, 1), options={"time_limit": 30})
    x = np.round(sol.x).astype(bool)
    succ = (1 - P["p_fail"]) ** ks
    res["lots_share"] = float((ks[x] * succ[x]).sum() / N)
    res["planned_share"] = float(ks[x].sum() / N)
    res["surplus_per_firm"] = float(w[x].sum() / N)
    for k in (2, 3, 4):
        res["share_by_k"][k] = float((ks[x & (ks == k)] * succ[x & (ks == k)]).sum() / N)
    return res


def market_clear(inst, P, rng, reservation_ask=False):
    """가격 장터: 판매자가 호가 a_i 를 부르고, 구매자 j 가 (가치 - 운송비) >= 호가 이면 구매.
    장터 운영자가 최적 배정(판매자 1곡·구매자 1곡)을 해주는 '유리한' 기준선."""
    N = inst["N"]
    if reservation_ask:
        a = inst["S"].copy()                         # 호가=판매자 최저 수용가(가격을 완벽히 아는 장터)
    else:
        a = inst["v"] * P["ask_ratio"] * np.exp(rng.normal(0, P["ask_sigma"], N))
    Wm = inst["W"].T                                 # Wm[i,j]: 구매자 j 의 i 로트 가치
    C = inst["C"]
    seen = rng.random((N, N)) < P["visibility"]
    feas = seen & inst["A"].T & (Wm - C >= a[:, None]) & (a[:, None] >= inst["S"][:, None] - 1e-9)
    surplus = Wm - inst["S"][:, None] - C
    succ = (1 - P["p_fail"]) ** 2
    w = succ * surplus - (1 - succ) * P["fail_cost_share"] * C
    mask = feas & (w > 0)
    cost = np.where(mask, -w, 1e6)
    r, c = linear_sum_assignment(cost)
    sel = mask[r, c]
    return dict(lots_share=float(succ * sel.sum() / N), planned_share=float(sel.sum() / N),
                surplus_per_firm=float(w[r, c][sel].sum() / N), share_by_k={2: 0.0, 3: 0.0, 4: 0.0})


# ----------------------------------------------------------------------------
# 4. 실험 보조 함수
# ----------------------------------------------------------------------------
def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def seeds_rng(tag, r):
    return np.random.default_rng([sum(map(ord, tag)), r])


def summarize(df, by, vals):
    g = df.groupby(by)
    out = g[vals].mean()
    for v in vals:
        out[v + "_ci"] = g[v].apply(ci95)
    return out.reset_index()


# ----------------------------------------------------------------------------
# 5. 실험 A~F
# ----------------------------------------------------------------------------
def exp_A(R, outdir):
    """A. 메커니즘 비교 (RQ1): 가격 장터 vs 2자 교환 vs 순환(<=3, <=4) vs 크레딧"""
    P = dict(DEFAULTS)
    rows = []
    for r in range(R):
        rng = seeds_rng("A", r)
        inst = make_instance(rng, P)
        cyc = graph_cycles(inst, 4)
        runs = {
            "Market: ask=list price": market_clear(inst, P, rng),
            "Market: ask=reservation (ideal)": market_clear(inst, P, rng, reservation_ask=True),
            "Barter 2-way": clear_cycles(inst, cyc, 2, "ntu", P),
            "Barter cycles <=3": clear_cycles(inst, cyc, 3, "ntu", P),
            "Barter cycles <=4": clear_cycles(inst, cyc, 4, "ntu", P),
            "Credit(list) <=3": clear_cycles(inst, cyc, 3, "credit_list", P),
            "Credit(surplus-split) <=3": clear_cycles(inst, cyc, 3, "credit_surplus", P),
        }
        for name, res in runs.items():
            rows.append(dict(seed=r, mechanism=name, lots_share=res["lots_share"],
                             surplus_per_firm=res["surplus_per_firm"],
                             share_k2=res["share_by_k"][2], share_k3=res["share_by_k"][3], share_k4=res["share_by_k"][4]))
    df = pd.DataFrame(rows)
    tab = summarize(df, "mechanism", ["lots_share", "surplus_per_firm", "share_k2", "share_k3", "share_k4"])
    order = list(dict.fromkeys(df["mechanism"]))
    tab = tab.set_index("mechanism").loc[order].reset_index()
    tab.to_csv(os.path.join(outdir, "A_mechanisms.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].barh(tab["mechanism"], tab["lots_share"] * 100, xerr=tab["lots_share_ci"] * 100, color="#4c78a8")
    ax[0].set_xlabel("Lots moved per lot held (%)"); ax[0].invert_yaxis()
    ax[1].barh(tab["mechanism"], tab["surplus_per_firm"], xerr=tab["surplus_per_firm_ci"], color="#59a14f")
    ax[1].set_xlabel("Expected net surplus per firm (10k KRW)"); ax[1].invert_yaxis(); ax[1].set_yticklabels([])
    fig.suptitle(f"A. Mechanism comparison (N={P['N']}, mean degree d={P['d']}, {R} runs, 95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A_mechanisms.png"), dpi=140); plt.close(fig)
    return tab


def exp_B(R, outdir):
    """B. 밀도 스윕 (RQ2): 순환 이득이 밀도에 따라 어떻게 변하는가 (역U 가설)"""
    ds = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
    rows = []
    for d in ds:
        P = dict(DEFAULTS, d=d)
        for r in range(R):
            rng = seeds_rng(f"B{d}", r)
            inst = make_instance(rng, P)
            cyc = graph_cycles(inst, 4)
            row = dict(d=d, seed=r)
            for L in (2, 3, 4):
                row[f"L{L}"] = clear_cycles(inst, cyc, L, "ntu", P)["lots_share"]
            row["market"] = market_clear(inst, P, rng)["lots_share"]
            row["gain_3v2"] = row["L3"] - row["L2"]
            row["gain_4v3"] = row["L4"] - row["L3"]
            rows.append(row)
    df = pd.DataFrame(rows)
    tab = summarize(df, "d", ["L2", "L3", "L4", "market", "gain_3v2", "gain_4v3"])
    tab.to_csv(os.path.join(outdir, "B_density.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, lab in [("L2", "Barter 2-way"), ("L3", "Cycles <=3"), ("L4", "Cycles <=4"), ("market", "Market (ask=list)")]:
        ax[0].errorbar(tab["d"], tab[col] * 100, yerr=tab[col + "_ci"] * 100, marker="o", label=lab, capsize=2)
    ax[0].set_xlabel("Mean number of acceptable lots per firm (d)"); ax[0].set_ylabel("Lots moved (%)"); ax[0].legend()
    for col, lab in [("gain_3v2", "3-cycles vs 2-way"), ("gain_4v3", "4-cycles vs <=3")]:
        ax[1].errorbar(tab["d"], tab[col] * 100, yerr=tab[col + "_ci"] * 100, marker="s", label=lab, capsize=2)
    ax[1].axhline(0, color="gray", lw=0.8); ax[1].set_xlabel("d"); ax[1].set_ylabel("Incremental lots moved (pp)"); ax[1].legend()
    fig.suptitle(f"B. Density sweep (N={DEFAULTS['N']}, {R} runs, 95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_B_density.png"), dpi=140); plt.close(fig)
    return tab


def exp_C(R, outdir):
    """C. 순환 길이 상한 (RQ3): 이탈 확률 x 물류비 배율 에서 기대 순이익을 최대화하는 L"""
    pfails = [0.0, 0.02, 0.05, 0.10, 0.20]
    scales = [0.5, 1, 2, 4]
    rows = []
    for r in range(R):
        rng = seeds_rng("C", r)
        base = make_instance(rng, DEFAULTS)
        cyc = graph_cycles(base, 4)
        for sc in scales:
            inst = with_cost_scale(base, sc)
            for pf in pfails:
                P = dict(DEFAULTS, p_fail=pf, cost_scale=sc)
                for L in (2, 3, 4):
                    res = clear_cycles(inst, cyc, L, "ntu", P)
                    rows.append(dict(seed=r, scale=sc, p_fail=pf, L=L, surplus=res["surplus_per_firm"], lots=res["lots_share"]))
    df = pd.DataFrame(rows)
    tab = summarize(df, ["scale", "p_fail", "L"], ["surplus", "lots"])
    tab.to_csv(os.path.join(outdir, "C_length.csv"), index=False)
    best = (tab.loc[tab.groupby(["scale", "p_fail"])["surplus"].idxmax()][["scale", "p_fail", "L", "surplus"]]
            .rename(columns={"L": "best_L"}))
    best.to_csv(os.path.join(outdir, "C_best_length.csv"), index=False)
    fig, ax = plt.subplots(1, len(scales), figsize=(14, 3.6), sharey=False)
    for a, sc in zip(ax, scales):
        for L in (2, 3, 4):
            t = tab[(tab.scale == sc) & (tab.L == L)]
            a.errorbar(t["p_fail"] * 100, t["surplus"], yerr=t["surplus_ci"], marker="o", label=f"L<={L}", capsize=2)
        a.set_title(f"cost x{sc}"); a.set_xlabel("per-firm failure prob. (%)")
    ax[0].set_ylabel("Expected surplus per firm"); ax[0].legend()
    fig.suptitle(f"C. Cycle-length cap vs failure risk and logistics cost ({R} runs)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_C_length.png"), dpi=140); plt.close(fig)
    return best


def exp_D(R, outdir):
    """D. 참여 유인 (RQ4): 물류비 배율이 커지면 성사 거래가 얼마나 줄어드는가"""
    scales = [0, 0.5, 1, 2, 4, 8, 16]
    rows = []
    for r in range(R):
        rng = seeds_rng("D", r)
        base = make_instance(rng, DEFAULTS)
        cyc = graph_cycles(base, 3)
        for sc in scales:
            inst = with_cost_scale(base, sc)
            P = dict(DEFAULTS, cost_scale=sc)
            res = clear_cycles(inst, cyc, 3, "ntu", P)
            resc = clear_cycles(inst, cyc, 3, "credit_surplus", P)
            tot = feas = 0
            for k in (2, 3):
                if len(cyc[k]):
                    net, _, _ = cycle_stats(inst, cyc[k])
                    tot += len(net); feas += int((net >= 0).all(1).sum())
            mc = np.median(inst["C"][inst["A"].T]) if sc > 0 else 0.0
            rows.append(dict(seed=r, scale=sc, lots=res["lots_share"], lots_credit=resc["lots_share"],
                             feasible_rate=feas / tot if tot else np.nan,
                             med_cost_pct=100 * mc / np.median(inst["v"])))
    df = pd.DataFrame(rows)
    tab = summarize(df, "scale", ["lots", "lots_credit", "feasible_rate", "med_cost_pct"])
    tab.to_csv(os.path.join(outdir, "D_incentive.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(tab["scale"], tab["lots"] * 100, yerr=tab["lots_ci"] * 100, marker="o", label="No credit (all members gain)", capsize=2)
    ax[0].errorbar(tab["scale"], tab["lots_credit"] * 100, yerr=tab["lots_credit_ci"] * 100, marker="s", label="Credit (surplus-split, ideal)", capsize=2)
    ax[0].set_xlabel("Logistics cost multiplier"); ax[0].set_ylabel("Lots moved (%)"); ax[0].legend()
    ax[1].errorbar(tab["scale"], tab["feasible_rate"] * 100, yerr=tab["feasible_rate_ci"] * 100, marker="o", color="#e15759", capsize=2)
    ax[1].set_xlabel("Logistics cost multiplier"); ax[1].set_ylabel("Cycles passing all-member participation test (%)")
    fig.suptitle(f"D. Participation incentive vs logistics cost ({R} runs)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_D_incentive.png"), dpi=140); plt.close(fig)
    return tab


def exp_E(R, outdir):
    """E. 임계 규모 (RQ6): 참여 기업 수 N 에 따른 성사 비율 (쌍별 수용 확률 q 고정)"""
    q = DEFAULTS["d"] / (DEFAULTS["N"] - 1)
    Ns = [20, 30, 45, 60, 90, 120, 200]
    rows = []
    for n in Ns:
        P = dict(DEFAULTS, N=n, d=q * (n - 1))
        for r in range(R):
            rng = seeds_rng(f"E{n}", r)
            inst = make_instance(rng, P)
            cyc = graph_cycles(inst, 3)
            rows.append(dict(N=n, seed=r,
                             L2=clear_cycles(inst, cyc, 2, "ntu", P)["lots_share"],
                             L3=clear_cycles(inst, cyc, 3, "ntu", P)["lots_share"],
                             market=market_clear(inst, P, rng)["lots_share"]))
    df = pd.DataFrame(rows)
    tab = summarize(df, "N", ["L2", "L3", "market"])
    tab.to_csv(os.path.join(outdir, "E_scale.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, lab in [("L2", "Barter 2-way"), ("L3", "Cycles <=3"), ("market", "Market (ask=list)")]:
        ax.errorbar(tab["N"], tab[col] * 100, yerr=tab[col + "_ci"] * 100, marker="o", label=lab, capsize=2)
    ax.set_xlabel("Number of participating firms (N), fixed pairwise acceptance prob."); ax.set_ylabel("Lots moved (%)"); ax.legend()
    ax.set_title(f"E. Critical mass ({R} runs)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_E_scale.png"), dpi=140); plt.close(fig)
    return tab


def exp_F(R, outdir):
    """F. 가격 장터와의 비교: 판매자 호가 수준(정가 대비)에 따라 장터 vs 순환 교환의 우열"""
    alphas = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3]
    rows = []
    for a_ in alphas:
        P = dict(DEFAULTS, ask_ratio=a_)
        for r in range(R):
            rng = seeds_rng("F", r)             # 같은 인스턴스를 호가 수준만 바꿔 비교
            inst = make_instance(rng, DEFAULTS)
            cyc = graph_cycles(inst, 3)
            rows.append(dict(alpha=a_, seed=r,
                             market=market_clear(inst, P, np.random.default_rng([7, r]))["lots_share"],
                             market_s=market_clear(inst, P, np.random.default_rng([7, r]))["surplus_per_firm"],
                             barter=clear_cycles(inst, cyc, 3, "ntu", P)["lots_share"],
                             barter_s=clear_cycles(inst, cyc, 3, "ntu", P)["surplus_per_firm"]))
    df = pd.DataFrame(rows)
    tab = summarize(df, "alpha", ["market", "barter", "market_s", "barter_s"])
    tab.to_csv(os.path.join(outdir, "F_anchor.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(tab["alpha"], tab["market"] * 100, yerr=tab["market_ci"] * 100, marker="o", label="Price market", capsize=2)
    ax[0].errorbar(tab["alpha"], tab["barter"] * 100, yerr=tab["barter_ci"] * 100, marker="s", label="Barter cycles <=3", capsize=2)
    ax[0].set_xlabel("Seller ask / list price"); ax[0].set_ylabel("Lots moved (%)"); ax[0].legend()
    ax[1].errorbar(tab["alpha"], tab["market_s"], yerr=tab["market_s_ci"], marker="o", label="Price market", capsize=2)
    ax[1].errorbar(tab["alpha"], tab["barter_s"], yerr=tab["barter_s_ci"], marker="s", label="Barter cycles <=3", capsize=2)
    ax[1].set_xlabel("Seller ask / list price"); ax[1].set_ylabel("Expected surplus per firm")
    fig.suptitle(f"F. When does a price market beat barter? ({R} runs)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_F_anchor.png"), dpi=140); plt.close(fig)
    return tab



def exp_G(R, outdir):
    """G. 장터의 탐색 마찰: 구매자가 판매 로트를 발견할 확률(visibility)에 따른 장터 vs 순환 교환"""
    vis = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    rows = []
    for v_ in vis:
        P = dict(DEFAULTS, visibility=v_)
        for r in range(R):
            rng = seeds_rng("G", r)
            inst = make_instance(rng, DEFAULTS)
            cyc = graph_cycles(inst, 3)
            rows.append(dict(visibility=v_, seed=r,
                             market=market_clear(inst, P, np.random.default_rng([9, r]))["lots_share"],
                             barter_ntu=clear_cycles(inst, cyc, 3, "ntu", P)["lots_share"],
                             barter_credit=clear_cycles(inst, cyc, 3, "credit_list", P)["lots_share"]))
    df = pd.DataFrame(rows)
    tab = summarize(df, "visibility", ["market", "barter_ntu", "barter_credit"])
    tab.to_csv(os.path.join(outdir, "G_visibility.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, lab in [("market", "Price market"), ("barter_ntu", "Barter cycles <=3 (no credit)"), ("barter_credit", "Barter cycles <=3 (credit)")]:
        ax.errorbar(tab["visibility"], tab[col] * 100, yerr=tab[col + "_ci"] * 100, marker="o", label=lab, capsize=2)
    ax.set_xlabel("Prob. a buyer discovers a seller's lot (market search friction)"); ax.set_ylabel("Lots moved (%)"); ax.legend()
    ax.set_title(f"G. Market search friction ({R} runs)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_G_visibility.png"), dpi=140); plt.close(fig)
    return tab


def self_test():
    """코드가 맞게 동작하는지 작은 문제에서 무차별 대입 결과와 비교."""
    import itertools
    ok = True
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        P = dict(DEFAULTS, N=8, d=3.5, p_fail=0.05)
        inst = make_instance(rng, P)
        # (1) 순환 열거 검증
        cyc = graph_cycles(inst, 4)
        found = {frozenset([tuple(c)]) for k in cyc for c in cyc[k]}
        cnt = sum(len(cyc[k]) for k in cyc)
        brute = 0
        for k in (2, 3, 4):
            for nodes in itertools.permutations(range(8), k):
                if nodes[0] != min(nodes):
                    continue
                if all(inst["A"][nodes[(i + 1) % k], nodes[i]] for i in range(k)):
                    brute += 1
        if cnt != brute:
            ok = False; print("열거 불일치", seed, cnt, brute)
        # (2) 정수계획 vs 무차별 대입
        for regime in ("ntu", "credit_list"):
            for L in (2, 3, 4):
                res = clear_cycles(inst, cyc, L, regime, P)
                cand, w, ks = candidates(inst, cyc, L, regime, P)
                nodes_list = [tuple(c) for arr in cand for c in arr]
                best = 0.0
                for mask in range(1 << len(nodes_list)) if len(nodes_list) <= 14 else []:
                    used, tot, good = set(), 0.0, True
                    for i in range(len(nodes_list)):
                        if mask >> i & 1:
                            if used & set(nodes_list[i]):
                                good = False; break
                            used |= set(nodes_list[i]); tot += w[i]
                    if good and tot > best:
                        best = tot
                if len(nodes_list) <= 14 and abs(best - res["surplus_per_firm"] * 8) > 1e-6:
                    ok = False; print("IP 불일치", seed, regime, L, best, res["surplus_per_firm"] * 8)
    print("SELF-TEST", "통과" if ok else "실패")
    return ok

EXPERIMENTS = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "F": exp_F, "G": exp_G}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="빠른 점검(반복 5회)")
    ap.add_argument("--seeds", type=int, default=None, help="실험당 반복 횟수(기본 30)")
    ap.add_argument("--only", default="", help="예: A,B")
    ap.add_argument("--out", default="results")
    ap.add_argument("--test", action="store_true", help="자체 검증 테스트만 실행")
    args = ap.parse_args()
    if args.test:
        raise SystemExit(0 if self_test() else 1)
    R = args.seeds or (5 if args.quick else 30)
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    for key in (args.only.split(",") if args.only else EXPERIMENTS):
        t0 = time.time()
        tab = EXPERIMENTS[key.strip()](R, args.out)
        print(f"\n=== 실험 {key} 완료 ({time.time() - t0:.0f}s) ===")
        print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
