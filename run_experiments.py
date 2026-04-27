"""
run_experiments.py
------------------
批量实验运行器：规则策略 vs RL 智能体对比分析。

用法：
    python run_experiments.py                   # 使用下方 USER CONFIGURATION 里的设置
    python run_experiments.py --mode 1v1        # 命令行覆盖模式（其余设置仍读配置区）
    python run_experiments.py --seed 7 --rl-episodes 80
    python run_experiments.py --no-events --rl-episodes 0

═══════════════════════════════════════════════════════════════════
  直接修改下面的 USER CONFIGURATION 区域来自定义实验，
  不需要改动下方的任何函数代码。
═══════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# USER CONFIGURATION — 在这里设置你的实验参数
# ═══════════════════════════════════════════════════════════════

# ── 基础设置 ──────────────────────────────────────────────────
MODE        = "5-Nation"  # 竞争模式: "1v1" | "3-Nation" | "4-Nation" | "5-Nation" | "1v3"
SEED        = 42          # 随机种子（相同种子 → 相同随机事件序列）
EVENTS      = True        # True = 开启历史随机事件
RL_EPISODES = 60          # RL 训练轮数；设为 0 则跳过 RL，只跑规则策略

# ── 每个国家的初始参数覆盖 ────────────────────────────────────
# 只填想修改的字段，其余保持原型默认值。
# 国家名就是原型字母（"A"/"B"/"C"/"D"/"E"）或带后缀如 "A·A"（1v3 模式里的副本）
#
# 地理参数（geo）取值范围 0.0 ~ 1.0：
#   coast_access       海岸条件（影响贸易、航海）
#   terrain_quality    耕地质量（影响农业 GDP）
#   climate_score      气候适宜度（影响综合生产率）
#   river_density      内河密度（降低内陆运输成本）
#   strategic_location 战略位置（影响军事和贸易地位）
#
# 资源参数（res）取值范围 0.0 ~ 4.0：
#   food   粮食    metal  金属    wood   木材
#   luxury 奢侈品  coal   煤炭
#
# 规则策略（strategy）可选值：
#   "AgrarianConservative" | "MaritimeExpansionist" | "IndustrialPioneer"
#   "TradeHub"             | "Mercantilist"
CIV_OVERRIDES = {
    # 示例：把 A 国改成海岸优势更强、煤炭更少
    # "A": {
    #     "geo": {"coast_access": 0.6, "terrain_quality": 0.7},
    #     "res": {"coal": 0.2, "food": 3.5},
    #     "strategy": "AgrarianConservative",
    # },
    # 示例：把 C 国改成工业先驱但限制粮食
    # "C": {
    #     "res": {"coal": 3.0, "food": 0.8},
    #     "strategy": "IndustrialPioneer",
    # },
}

# ── RL 智能体分配 ──────────────────────────────────────────────
# 指定哪些国家使用 RL 策略及其优化目标。
# 留空 {} 则自动将前 3 个国家分配为 gdp / power / trade。
#
# RL 策略可选值：
#   "RL_gdp"   → Q-learning，奖励 = GDP 增长率
#   "RL_power" → Q-learning，奖励 = GDP × 军事 × √领土（综合国力）
#   "RL_trade" → Q-learning，奖励 = 贸易收益
RL_AGENT_MAP = {
    # "A": "RL_gdp",
    # "B": "RL_power",
    # "C": "RL_trade",
}

# ── 输出指标选择 ──────────────────────────────────────────────
# 控制台表格和 CSV 里包含哪些列（顺序即输出顺序）。
# 完整可选列表：
#   经济: "gdp" "gdp_per_capita" "population" "trade_income" "colonial_income"
#   国力: "territories" "military_str" "trade_openness"
#   技术: "tech_composite" "tech_agri" "tech_nav" "tech_mil" "tech_ind" "tech_com"
#   决策: "decision_tech" "decision_trade" "decision_expand"
OUTPUT_METRICS = [
    "gdp",
    "tech_composite",
    "trade_income",
    "territories",
    "military_str",
]

# ── RL 输出控制 ────────────────────────────────────────────────
# 每项设为 True/False 独立控制是否输出对应的 RL 分析内容。
RL_OUTPUT = {
    "training_curve":   True,   # 训练轮次收益变化摘要（start/end/Δ%）
    "gdp_comparison":   True,   # 规则策略 vs RL 的终局 GDP 对比表
    "decision_summary": True,   # 每个 RL Agent 的决策模式概览
    "action_frequency": True,   # 动作频率分布（Tech/Trade/Expand 各选多少次）
    "era_breakdown":    True,   # 各历史时期的技术偏好分析
    "per_turn_trace":   True,   # 每回合详细决策记录
    "save_chart":       True,   # 保存 rule vs RL 对比图表（PNG）
    "save_csv":         True,   # 保存 RL 历史数据（CSV）
}

# ═══════════════════════════════════════════════════════════════
# 以下为功能代码 — 一般无需修改
# ═══════════════════════════════════════════════════════════════

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import SimulationEngine
from src.archetypes import build_competition_civs, COMPETITION_MODES
from src.strategies import make_strategy, QLearningStrategy, RULE_BASED_STRATEGIES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 显示常量 ──────────────────────────────────────────────────
TECH_SHORT   = {"agriculture": "Agri", "navigation": "Nav", "military": "Mil",
                "industry": "Ind",  "commerce":  "Com"}
TRADE_SHORT  = {"open": "Open", "balanced": "Bal", "closed": "Cls"}
EXPAND_SHORT = {0: "Stay", 1: "Mod", 2: "Agg"}
TECH_ORDER   = ["agriculture", "navigation", "military", "industry", "commerce"]
TECH_IDX     = {t: i for i, t in enumerate(TECH_ORDER)}
TRADE_ORDER  = {"open": 0, "balanced": 1, "closed": 2}
ERA_BANDS    = [(1000, 1400, "#8ecae6"), (1400, 1600, "#ffb703"),
                (1600, 1750, "#fb8500"), (1750, 1860, "#8338ec")]
PALETTE      = ["#e63946", "#457b9d", "#f4a261", "#2a9d8f", "#9c6644"]

# 所有可输出列的显示宽度（用于控制台对齐）
_COL_WIDTH = {
    "gdp": 9, "gdp_per_capita": 11, "population": 10,
    "trade_income": 10, "colonial_income": 13, "territories": 8,
    "military_str": 9, "trade_openness": 10, "tech_composite": 10,
    "tech_agri": 8, "tech_nav": 8, "tech_mil": 8, "tech_ind": 8, "tech_com": 8,
    "decision_tech": 8, "decision_trade": 8, "decision_expand": 8,
}
_COL_FORMAT = {
    "decision_tech": "s", "decision_trade": "s", "decision_expand": "s",
}


def _sep(n=72):
    return "─" * n


# ─────────────────────────────────────────────────────────────────────────
# 模拟运行器
# ─────────────────────────────────────────────────────────────────────────

def run_rule_based(mode: str, seed: int, events: bool,
                   overrides: dict = None) -> pd.DataFrame:
    """
    运行规则策略模拟。

    参数：
      mode      : 竞争模式名（见 COMPETITION_MODES）
      seed      : 随机种子
      events    : 是否开启随机事件
      overrides : CIV_OVERRIDES 格式的参数覆盖字典
    返回：
      历史数据 DataFrame（每行 = 一个国家某回合的快照）
    """
    civs   = build_competition_civs(mode, overrides or {})
    engine = SimulationEngine(civs=civs, events_enabled=events, seed=seed)
    engine.run()
    return engine.get_history_df()


def run_rl(mode: str, seed: int, episodes: int,
           overrides: dict = None, rl_agent_map: dict = None):
    """
    训练 RL 智能体，然后跑一次最终模拟。

    参数：
      mode         : 竞争模式名
      seed         : 随机种子
      episodes     : RL 训练轮数
      overrides    : CIV_OVERRIDES 格式的初始参数覆盖
      rl_agent_map : {国家名: RL策略类型} 的字典；
                     若为空则自动将前 3 个国家分配 gdp/power/trade
    返回：
      (rl_df, curves, engine)
      rl_df  : 最终模拟历史 DataFrame
      curves : {国家名: [每轮终局GDP]} 训练曲线
      engine : SimulationEngine 实例（含 strategy_map）
    """
    civs           = build_competition_civs(mode, overrides or {})
    rl_agent_map   = rl_agent_map or {}
    default_rl     = ["RL_gdp", "RL_power", "RL_trade"]
    strategy_map   = {}

    for i, civ in enumerate(civs):
        if civ.name in rl_agent_map:
            # 用户显式指定了该国的 RL 策略
            strategy_map[civ.name] = make_strategy(rl_agent_map[civ.name])
        elif not rl_agent_map and i < len(default_rl):
            # 未指定时自动分配前 3 国
            strategy_map[civ.name] = make_strategy(default_rl[i])
        else:
            strategy_map[civ.name] = make_strategy(civ.strategy_name)

    engine = SimulationEngine(
        civs=civs, strategy_map=strategy_map,
        events_enabled=True, seed=seed, training_mode=True,
    )
    curves = engine.train_rl_agents(n_episodes=episodes)
    engine.run()
    return engine.get_history_df(), curves, engine


# ─────────────────────────────────────────────────────────────────────────
# 文本输出
# ─────────────────────────────────────────────────────────────────────────

def print_turn_table(df: pd.DataFrame, mode: str,
                     metrics: list = None) -> None:
    """
    打印每回合各国的指标快照表。

    参数：
      df      : 历史 DataFrame
      mode    : 竞争模式名（仅用于标题）
      metrics : 要展示的列名列表；None 则默认展示 gdp 和 tech_composite
    """
    if metrics is None:
        metrics = ["gdp", "tech_composite"]

    civs  = sorted(df["civilization"].unique())
    years = sorted(df["year"].unique())
    col_w = 10

    header = f"{'Year':<6}" + "".join(f"  {c:>{col_w}}" for c in civs)

    print(f"\n{'═'*72}")
    print(f"Mode: {mode}  —  per-turn snapshots")
    print('═'*72)

    for metric in metrics:
        if metric not in df.columns:
            print(f"  [skip] 列 '{metric}' 不存在")
            continue
        print(f"\n  {metric}:")
        print(f"  {header}")
        print(f"  {_sep(6 + (col_w + 2) * len(civs))}")
        for y in years:
            snap = df[df["year"] == y]
            row  = f"  {y:<6}"
            for civ in civs:
                r = snap[snap["civilization"] == civ]
                v = r[metric].values[0] if not r.empty else float("nan")
                fmt_char = _COL_FORMAT.get(metric, "f")
                if fmt_char == "s":
                    row += f"  {str(v):>{col_w}}"
                else:
                    row += f"  {v:>{col_w}.3f}"
            print(row)


def print_decisions_table(df: pd.DataFrame, civ: str,
                          metrics: list = None, label: str = "") -> None:
    """
    打印单个国家每回合的决策 + 指定指标。

    参数：
      df      : 历史 DataFrame
      civ     : 国家名
      metrics : 要展示的指标列表；None 则使用 OUTPUT_METRICS
      label   : 附加在国家名后的标签（如 "rule" 或 "RL/gdp"）
    """
    if metrics is None:
        metrics = OUTPUT_METRICS

    d   = df[df["civilization"] == civ].sort_values("year")
    tag = f"  [{label}]" if label else ""

    # 决策列始终展示
    fixed_cols  = ["Year", "Tech", "Trade", "Exp"]
    fixed_width = [6, 6, 6, 5]
    metric_width = [_COL_WIDTH.get(m, 9) for m in metrics]

    header = (f"  {fixed_cols[0]:<{fixed_width[0]}} "
              f"{fixed_cols[1]:>{fixed_width[1]}} "
              f"{fixed_cols[2]:>{fixed_width[2]}} "
              f"{fixed_cols[3]:>{fixed_width[3]}}")
    for m, w in zip(metrics, metric_width):
        header += f"  {m:>{w}}"

    print(f"\n  {civ}{tag}")
    print(header)
    print(f"  {_sep(sum(fixed_width) + sum(metric_width) + 2 * len(metrics) + 3)}")

    for _, r in d.iterrows():
        tech  = TECH_SHORT.get(r.get("decision_tech",   ""), str(r.get("decision_tech",  "")))
        trade = TRADE_SHORT.get(r.get("decision_trade", ""), str(r.get("decision_trade", "")))
        exp   = EXPAND_SHORT.get(int(r.get("decision_expand", 0)),
                                 str(int(r.get("decision_expand", 0))))
        line  = (f"  {int(r['year']):<{fixed_width[0]}} "
                 f"{tech:>{fixed_width[1]}} "
                 f"{trade:>{fixed_width[2]}} "
                 f"{exp:>{fixed_width[3]}}")
        for m, w in zip(metrics, metric_width):
            v = r.get(m, float("nan"))
            fmt_char = _COL_FORMAT.get(m, "f")
            if fmt_char == "s":
                line += f"  {str(v):>{w}}"
            else:
                try:
                    line += f"  {float(v):>{w}.3f}"
                except (ValueError, TypeError):
                    line += f"  {'N/A':>{w}}"
        print(line)


def print_rl_decision_summary(engine: SimulationEngine, rl_df: pd.DataFrame,
                               metrics: list = None, rl_output: dict = None) -> None:
    """
    打印 RL 智能体的决策分析报告。

    参数：
      engine    : 训练后的 SimulationEngine（含 strategy_map）
      rl_df     : RL 最终模拟的历史 DataFrame
      metrics   : 每回合明细里展示的指标（传给 print_decisions_table）
      rl_output : RL_OUTPUT 格式的开关字典
    """
    if metrics is None:
        metrics = OUTPUT_METRICS
    if rl_output is None:
        rl_output = {k: True for k in RL_OUTPUT}

    print(f"\n{'═'*72}")
    print("RL Agent Decision Analysis")
    print('═'*72)

    for civ_name, strat in engine.strategy_map.items():
        if not isinstance(strat, QLearningStrategy):
            continue

        d = rl_df[rl_df["civilization"] == civ_name]
        print(f"\n  Agent : {civ_name}")
        print(f"  Reward: {strat.reward_type}")
        print(f"  Q-states explored: {len(strat.q_table)}")

        # 动作频率
        if rl_output.get("action_frequency", True):
            tech_v  = d["decision_tech"].value_counts()
            trade_v = d["decision_trade"].value_counts()
            exp_v   = d["decision_expand"].value_counts()
            n       = max(len(d), 1)
            print(f"\n  Action frequencies ({n} turns total):")
            print(f"    Tech:   " +
                  "  ".join(f"{TECH_SHORT.get(k,k)}={v}({100*v//n}%)"
                            for k, v in tech_v.items()))
            print(f"    Trade:  " +
                  "  ".join(f"{TRADE_SHORT.get(k,k)}={v}({100*v//n}%)"
                            for k, v in trade_v.items()))
            print(f"    Expand: " +
                  "  ".join(f"{EXPAND_SHORT.get(int(k),k)}={v}({100*v//n}%)"
                            for k, v in exp_v.items()))

        # 各时期技术偏好
        if rl_output.get("era_breakdown", True):
            era_map = {"MEDIEVAL": "E1-Med", "DISCOVERY": "E2-Disc",
                       "MERCANTILE": "E3-Merc", "INDUSTRIAL": "E4-Ind"}
            print(f"\n  Tech focus by era:")
            for era_key, era_label in era_map.items():
                de = d[d["era"] == era_key]
                if de.empty:
                    continue
                top = de["decision_tech"].value_counts().idxmax()
                pct = int(100 * de["decision_tech"].value_counts().max() / len(de))
                print(f"    {era_label}: preferred {TECH_SHORT.get(top, top):>4}  ({pct}% of turns)")

        # 每回合决策明细
        if rl_output.get("per_turn_trace", True):
            print_decisions_table(rl_df, civ_name,
                                  metrics=metrics,
                                  label=f"RL/{strat.reward_type}")


# ─────────────────────────────────────────────────────────────────────────
# 图表输出
# ─────────────────────────────────────────────────────────────────────────

def save_comparison_chart(rule_df: pd.DataFrame, rl_df: pd.DataFrame,
                          mode: str, rl_civs: list,
                          metric: str = "gdp") -> str:
    """
    生成规则策略 vs RL 的对比图（4 面板）。

    面板布局：
      上方：指定 metric 的 rule（实线）vs RL（虚线）时序对比
      中间：RL 技术选择热力图（国家 × 年份）
      下方：RL 贸易政策热力图 | RL 扩张等级热力图

    参数：
      metric : 上方对比图使用的指标列名（默认 "gdp"）
    返回：
      保存路径
    """
    fig = plt.figure(figsize=(17, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.28)

    # ── 面板 1：指标对比 ─────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    civs   = sorted(rule_df["civilization"].unique())

    for i, civ in enumerate(civs):
        color  = PALETTE[i % len(PALETTE)]
        d_rule = rule_df[rule_df["civilization"] == civ]
        d_rl   = rl_df[rl_df["civilization"] == civ]
        if metric in rule_df.columns:
            ax_top.plot(d_rule["year"], d_rule[metric],
                        color=color, linewidth=2.2, label=f"{civ} rule")
        if metric in rl_df.columns:
            ls = "--" if civ in rl_civs else ":"
            ax_top.plot(d_rl["year"], d_rl[metric],
                        color=color, linewidth=1.6, linestyle=ls, alpha=0.75,
                        label=f"{civ} RL" if civ in rl_civs else "_")

    for x1, x2, c in ERA_BANDS:
        ax_top.axvspan(x1, x2, alpha=0.05, color=c)

    ax_top.set_xlabel("Year")
    ax_top.set_ylabel(metric)
    ax_top.set_title(f"{metric} — rule-based (solid) vs RL agents (dashed)  [{mode}]")
    ax_top.legend(fontsize=7.5, ncol=min(len(civs) * 2, 6), loc="upper left")
    ax_top.grid(alpha=0.25)

    if not rl_civs:
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"comparison_{mode.replace('-','_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    # ── 面板 2-4：RL 决策热力图 ──────────────────────────────
    rl_sub = rl_df[rl_df["civilization"].isin(rl_civs)].copy()
    years  = sorted(rl_sub["year"].unique())
    nr     = len(rl_civs)

    def _make_matrix(col, mapping):
        m = np.zeros((nr, len(years)))
        for ri, civ in enumerate(rl_civs):
            for ci, y in enumerate(years):
                r = rl_sub[(rl_sub["civilization"] == civ) & (rl_sub["year"] == y)]
                if not r.empty:
                    raw = r[col].values[0]
                    m[ri, ci] = mapping.get(raw, mapping.get(int(raw) if str(raw).isdigit() else raw, 0))
        return m

    def _label_axes(ax, title, xtick_step=2):
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rl_civs, fontsize=8)
        xt = list(range(0, len(years), xtick_step))
        ax.set_xticks(xt)
        ax.set_xticklabels([years[i] for i in xt], rotation=45, fontsize=7)
        ax.set_title(title, fontsize=9)

    ax_tech = fig.add_subplot(gs[1, :])
    mat_tech = _make_matrix("decision_tech", TECH_IDX)
    im1 = ax_tech.imshow(mat_tech, aspect="auto", cmap="tab10",
                          vmin=0, vmax=4, interpolation="nearest")
    _label_axes(ax_tech, "Tech Focus  (0=Agri  1=Nav  2=Mil  3=Ind  4=Com)")
    cb1 = fig.colorbar(im1, ax=ax_tech, orientation="vertical", pad=0.01, fraction=0.015)
    cb1.set_ticks([0, 1, 2, 3, 4])
    cb1.set_ticklabels(["Agri", "Nav", "Mil", "Ind", "Com"], fontsize=7)

    ax_trade = fig.add_subplot(gs[2, 0])
    mat_trade = _make_matrix("decision_trade", TRADE_ORDER)
    im2 = ax_trade.imshow(mat_trade, aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=2, interpolation="nearest")
    _label_axes(ax_trade, "Trade Policy  (0=Open  1=Balanced  2=Closed)")
    cb2 = fig.colorbar(im2, ax=ax_trade, pad=0.02, fraction=0.08)
    cb2.set_ticks([0, 1, 2])
    cb2.set_ticklabels(["Open", "Bal", "Cls"], fontsize=7)

    ax_exp = fig.add_subplot(gs[2, 1])
    mat_exp = _make_matrix("decision_expand", {0: 0, 1: 1, 2: 2})
    im3 = ax_exp.imshow(mat_exp, aspect="auto", cmap="YlOrRd",
                         vmin=0, vmax=2, interpolation="nearest")
    _label_axes(ax_exp, "Expansion  (0=Stay  1=Moderate  2=Aggressive)")
    cb3 = fig.colorbar(im3, ax=ax_exp, pad=0.02, fraction=0.08)
    cb3.set_ticks([0, 1, 2])
    cb3.set_ticklabels(["Stay", "Mod", "Agg"], fontsize=7)

    path = os.path.join(OUTPUT_DIR, f"rl_comparison_{mode.replace('-','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_turn_charts(rule_df: pd.DataFrame, mode: str,
                     metrics: list = None) -> str:
    """
    生成各国 GDP + 指定指标的小多图（small-multiples）。

    参数：
      metrics : 在副 Y 轴上叠加显示的指标（默认 ["tech_composite"]）
    返回：
      保存路径
    """
    if metrics is None:
        overlay = ["tech_composite"]
    else:
        overlay = [m for m in metrics if m != "gdp" and m in rule_df.columns][:1]

    civs = sorted(rule_df["civilization"].unique())
    ncol = min(3, len(civs))
    nrow = -(-len(civs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    fig.suptitle(f"Per-turn data — {mode}", fontsize=13)

    for idx, civ in enumerate(civs):
        ax  = axes[idx // ncol][idx % ncol]
        d   = rule_df[rule_df["civilization"] == civ]
        col = PALETTE[idx % len(PALETTE)]

        ax.plot(d["year"], d["gdp"], color=col, linewidth=2, label="GDP")

        if overlay and overlay[0] in d.columns:
            ax2 = ax.twinx()
            ax2.plot(d["year"], d[overlay[0]], color=col, linewidth=1.2,
                     linestyle="--", alpha=0.6, label=overlay[0])
            ax2.set_ylabel(overlay[0], fontsize=8)

        for x1, x2, c in ERA_BANDS:
            ax.axvspan(x1, x2, alpha=0.06, color=c)

        ax.set_title(civ, fontsize=10, fontweight="bold", color=col)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("GDP",  fontsize=8)
        ax.grid(alpha=0.2)

    for idx in range(len(civs), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"turns_{mode.replace('-','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 命令行参数可以覆盖配置区的值（不改配置区也能用）
    parser = argparse.ArgumentParser(description="World Economy Sim — experiment runner")
    parser.add_argument("--mode",        default=None,
                        help=f"覆盖 MODE。可选: {list(COMPETITION_MODES.keys())}")
    parser.add_argument("--seed",        type=int, default=None, help="覆盖 SEED")
    parser.add_argument("--no-events",   action="store_true",    help="关闭随机事件（覆盖 EVENTS）")
    parser.add_argument("--rl-episodes", type=int, default=None, help="覆盖 RL_EPISODES")
    args = parser.parse_args()

    # 命令行优先，没传则读配置区
    mode        = args.mode        if args.mode        is not None else MODE
    seed        = args.seed        if args.seed        is not None else SEED
    rl_episodes = args.rl_episodes if args.rl_episodes is not None else RL_EPISODES
    events      = (not args.no_events) if args.no_events else EVENTS

    if mode not in COMPETITION_MODES:
        print(f"Unknown mode '{mode}'. Available: {list(COMPETITION_MODES.keys())}")
        sys.exit(1)

    plt.rcParams["font.family"] = ["DejaVu Sans"]

    print(f"\nOutput dir : {OUTPUT_DIR}")
    print(f"Mode       : {mode}  ({COMPETITION_MODES[mode]['desc']})")
    print(f"Seed       : {seed}   Events: {events}   RL episodes: {rl_episodes}")
    print(f"Metrics    : {OUTPUT_METRICS}")
    if CIV_OVERRIDES:
        print(f"Overrides  : {list(CIV_OVERRIDES.keys())}")
    if RL_AGENT_MAP:
        print(f"RL agents  : {RL_AGENT_MAP}")

    # ── 1. 规则策略模拟 ──────────────────────────────────────
    print("\nRunning rule-based simulation...")
    rule_df = run_rule_based(mode, seed, events, CIV_OVERRIDES)

    print_turn_table(rule_df, mode, metrics=OUTPUT_METRICS)

    print(f"\n{'═'*72}")
    print("Rule-based — decisions per nation:")
    for civ in sorted(rule_df["civilization"].unique()):
        print_decisions_table(rule_df, civ, metrics=OUTPUT_METRICS, label="rule")

    csv_path = os.path.join(OUTPUT_DIR, f"turns_{mode.replace('-','_')}.csv")
    rule_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved  : {csv_path}")

    chart_path = save_turn_charts(rule_df, mode, metrics=OUTPUT_METRICS)
    print(f"Chart saved: {chart_path}")

    # ── 2. RL 训练与对比 ─────────────────────────────────────
    if rl_episodes > 0:
        print(f"\nTraining RL agents ({rl_episodes} episodes)...")
        rl_df, curves, rl_engine = run_rl(
            mode, seed, rl_episodes,
            overrides=CIV_OVERRIDES,
            rl_agent_map=RL_AGENT_MAP,
        )

        rl_civs = [n for n, s in rl_engine.strategy_map.items()
                   if isinstance(s, QLearningStrategy)]

        # 训练曲线摘要
        if RL_OUTPUT.get("training_curve", True):
            print(f"\n  Training improvement (ep-1 → ep-{rl_episodes}):")
            for civ, vals in curves.items():
                if vals:
                    delta = (vals[-1] - vals[0]) / max(abs(vals[0]), 1e-6) * 100
                    print(f"    {civ:<15}  start={vals[0]:.2f}  "
                          f"end={vals[-1]:.2f}  Δ={delta:+.1f}%")

        # 终局 GDP 对比
        if RL_OUTPUT.get("gdp_comparison", True):
            print(f"\n{'═'*72}")
            print("Final GDP — rule-based vs RL:")
            print(f"  {'Nation':<15} {'Rule':>9} {'RL':>9} {'Diff':>9}")
            print(f"  {_sep(46)}")
            for civ in sorted(rule_df["civilization"].unique()):
                r_max = rule_df[rule_df["civilization"] == civ]["year"].max()
                l_max = rl_df[rl_df["civilization"] == civ]["year"].max()
                r_gdp = rule_df[(rule_df["civilization"]==civ)&(rule_df["year"]==r_max)]["gdp"].values[0]
                l_gdp = rl_df[(rl_df["civilization"]==civ)&(rl_df["year"]==l_max)]["gdp"].values[0]
                diff  = l_gdp - r_gdp
                tag   = " ↑" if diff > 0.05 else (" ↓" if diff < -0.05 else "  ")
                print(f"  {civ:<15} {r_gdp:>9.2f} {l_gdp:>9.2f} {diff:>+9.2f}{tag}")

        # RL 决策分析
        if RL_OUTPUT.get("decision_summary", True):
            print_rl_decision_summary(
                rl_engine, rl_df,
                metrics=OUTPUT_METRICS,
                rl_output=RL_OUTPUT,
            )

        # 保存 RL CSV
        if RL_OUTPUT.get("save_csv", True):
            rl_csv = os.path.join(OUTPUT_DIR, f"rl_turns_{mode.replace('-','_')}.csv")
            rl_df.to_csv(rl_csv, index=False)
            print(f"\nRL CSV saved : {rl_csv}")

        # 保存对比图表
        if RL_OUTPUT.get("save_chart", True):
            comp_path = save_comparison_chart(rule_df, rl_df, mode, rl_civs)
            print(f"Comparison chart: {comp_path}")

    print(f"\n{'═'*72}")
    print("Done.  To explore interactively:  streamlit run app.py")
    print('═'*72)