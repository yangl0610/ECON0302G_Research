"""
run_discovery.py
----------------
新实验组：为 LaTeX 报告生成图表和数据。

实验 4: 策略替换反事实 — 相同禀赋，农业保守 vs 工业先驱策略
实验 5: 五强全规则策略 — 历史原型的长期自然均衡
实验 6: 三国 RL 博弈 — 农业/航海/工业智能体的策略涌现
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import SimulationEngine
from src.archetypes import build_competition_civs, _CONFIGS
from src.civilization import Geography, Resources, Civilization
from src.strategies import make_strategy, QLearningStrategy

DISC_DIR = os.path.join(os.path.dirname(__file__), "discovery")
os.makedirs(DISC_DIR, exist_ok=True)

ERA_BANDS = [(1000, 1400, "#8ecae6"), (1400, 1600, "#ffb703"),
             (1600, 1750, "#fb8500"), (1750, 1860, "#8338ec")]

# ──────────────────────────────────────────────────────────
# 辅助：运行规则策略模拟
# ──────────────────────────────────────────────────────────
def run_rule(civs_list):
    eng = SimulationEngine(civs=civs_list, events_enabled=False, seed=42)
    eng.run()
    return eng.get_history_df()

def run_rl_exp(civs_list, rl_map, competitive=False, episodes=120, seed=42):
    from copy import deepcopy
    strategy_map = {}
    for civ in civs_list:
        if civ.name in rl_map:
            strat = make_strategy(rl_map[civ.name])
            if isinstance(strat, QLearningStrategy):
                strat.competitive = competitive
                strat.competitive_weight = 0.25
        else:
            strat = make_strategy(civ.strategy_name)
        strategy_map[civ.name] = strat
    eng = SimulationEngine(civs=civs_list, strategy_map=strategy_map,
                           events_enabled=False, seed=seed, training_mode=True)
    eng.train_rl_agents(n_episodes=episodes)
    eng.run()
    return eng.get_history_df(), eng

def civ_from_cfg(name, arch, strategy_override=None, color=None):
    cfg = _CONFIGS[arch]
    return Civilization(
        name=name,
        geography=Geography(**cfg["geo"]),
        resources=Resources(**cfg["res"]),
        strategy_name=strategy_override or cfg["strategy"],
        color=color or cfg["color"],
    )

def add_era_bands(ax):
    for x1, x2, c in ERA_BANDS:
        ax.axvspan(x1, x2, alpha=0.07, color=c)

def add_era_labels(ax, y_pos_ratio=0.97):
    labels = [("Medieval\n1000-1400", 1200), ("Discovery\n1400-1600", 1500),
              ("Mercantilist\n1600-1750", 1675), ("Industrial\n1750-1850", 1800)]
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * y_pos_ratio
    for text, x in labels:
        ax.text(x, y, text, fontsize=6, ha='center', va='top', alpha=0.55,
                style='italic', color='#444')

# ══════════════════════════════════════════════════════════════
# 实验 4: 策略替换反事实
#   场景一（Baseline）: A(AgrarianConservative) + C(IndustrialPioneer)  1v1
#   场景二（Counterfactual）: A改为IndustrialPioneer + C(IndustrialPioneer)
#   问题: 农业文明切换策略后能否追上工业文明?
# ══════════════════════════════════════════════════════════════
print("\n[Exp 4] 策略替换反事实...")

civs_base = [civ_from_cfg("A_Agrarian", "A"), civ_from_cfg("C_Industrial", "C")]
civs_cf   = [civ_from_cfg("A_Industrial", "A", strategy_override="IndustrialPioneer",
                           color="#1a7a6e"),
             civ_from_cfg("C_Industrial", "C")]

df_base = run_rule(civs_base)
df_cf   = run_rule(civs_cf)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Exp 4 — Strategy Substitution Counterfactual\n"
             "Agrarian geography: AgrarianConservative (solid) vs IndustrialPioneer (dashed)",
             fontsize=11)

COLORS = {"A_Agrarian": "#2a9d8f", "A_Industrial": "#1a7a6e",
          "C_Industrial": "#e63946"}

for ax, col, ylabel in zip(axes,
        ["gdp", "gdp_per_capita", "tech_composite"],
        ["GDP (relative)", "GDP per Capita", "Tech Composite Index"]):
    for civ in ["A_Agrarian", "C_Industrial"]:
        d = df_base[df_base["civilization"] == civ]
        ax.plot(d["year"], d[col], color=COLORS[civ], lw=2, label=civ)
    d = df_cf[df_cf["civilization"] == "A_Industrial"]
    ax.plot(d["year"], d[col], color=COLORS["A_Industrial"], lw=2,
            linestyle="--", label="A (IndustrialPioneer)")
    add_era_bands(ax)
    ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
    ax.set_title(ylabel); ax.legend(fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
p4 = os.path.join(DISC_DIR, "exp4_strategy_substitution.png")
plt.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()
print(f"  -> {p4}")

# ── 打印终局数字 ──────────────────────────────────────────
def final_stats(df, civ_name):
    last = df[df["civilization"] == civ_name]["year"].max()
    r = df[(df["civilization"] == civ_name) & (df["year"] == last)].iloc[0]
    return r["gdp"], r["gdp_per_capita"], r["tech_composite"]

g_a_rule, pc_a_rule, tc_a_rule = final_stats(df_base, "A_Agrarian")
g_c, pc_c, tc_c                = final_stats(df_base, "C_Industrial")
g_a_cf, pc_a_cf, tc_a_cf       = final_stats(df_cf,   "A_Industrial")

print(f"  终局 GDP:  A_Agrarian={g_a_rule:.2f}  A_Industrial={g_a_cf:.2f}  C={g_c:.2f}")
print(f"  终局 GDPpc: A_Agrarian={pc_a_rule:.3f}  A_Industrial={pc_a_cf:.3f}  C={pc_c:.3f}")
print(f"  终局 Tech: A_Agrarian={tc_a_rule:.2f}  A_Industrial={tc_a_cf:.2f}  C={tc_c:.2f}")

# ══════════════════════════════════════════════════════════════
# 实验 5: 五强全规则策略 — 历史原型自然均衡
#   默认 5-Nation (A/B/C/D/E) 全用规则策略
#   问题: 不同历史原型在800年内的长期竞争均衡是什么?
# ══════════════════════════════════════════════════════════════
print("\n[Exp 5] 五强全规则策略...")

civs_5 = build_competition_civs("5-Nation")
df_5   = run_rule(civs_5)

COLORS_5 = {"A": "#2a9d8f", "B": "#457b9d", "C": "#e63946",
            "D": "#f4a261", "E": "#9c6644"}
STRAT_LABEL = {"A": "A: Agrarian", "B": "B: Maritime", "C": "C: Industrial",
               "D": "D: TradeHub", "E": "E: Mercantilist"}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Exp 5 — Five-Civilisation Rule-Based Competition\n"
             "Natural equilibrium of five historical archetypes (1000–1850)", fontsize=11)

metrics = [("gdp", "GDP"), ("gdp_per_capita", "GDP per Capita"),
           ("tech_composite", "Tech Composite"), ("trade_income", "Trade Income")]

for ax, (col, title) in zip(axes.flat, metrics):
    for civ in df_5["civilization"].unique():
        arch = civ.split("·")[0]
        d = df_5[df_5["civilization"] == civ]
        ax.plot(d["year"], d[col], color=COLORS_5.get(arch, "#888"),
                lw=2, label=STRAT_LABEL.get(arch, arch))
    add_era_bands(ax)
    ax.set_xlabel("Year"); ax.set_ylabel(title)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
p5 = os.path.join(DISC_DIR, "exp5_five_nation_rule.png")
plt.savefig(p5, dpi=150, bbox_inches="tight"); plt.close()
print(f"  -> {p5}")

print("  终局排名 (GDP):")
last_year = df_5["year"].max()
final_5 = df_5[df_5["year"] == last_year][["civilization","gdp","gdp_per_capita",
                                            "tech_composite","trade_income"]].sort_values("gdp", ascending=False)
print(final_5.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 实验 6: 三国 RL 博弈 (A/B/C) — 策略涌现对比
#   A (Agrarian) + B (Maritime) + C (Industrial) 各用 power-maximizing RL
#   对比规则策略: RL优化能否让弱势文明逆袭?
#   同时展示 RL 学到的技术偏好热力图
# ══════════════════════════════════════════════════════════════
print("\n[Exp 6] 三国 RL 博弈...")

civs_3_rule = build_competition_civs("3-Nation")
df_3_rule   = run_rule(civs_3_rule)

civs_3_rl   = build_competition_civs("3-Nation")
rl_map_3    = {"A": "RL_power", "B": "RL_power", "C": "RL_power"}
df_3_rl, eng_3 = run_rl_exp(civs_3_rl, rl_map_3, competitive=True, episodes=120)

fig = plt.figure(figsize=(15, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32)
fig.suptitle("Exp 6 — Three-Nation RL Competition (A/B/C)\n"
             "Rule-based (solid) vs RL power-maximising with competitive incentive (dashed)",
             fontsize=11)

COLORS_3 = {"A": "#2a9d8f", "B": "#457b9d", "C": "#e63946"}
civs_order = ["A", "B", "C"]

def plot_ts(ax, col, title, ylabel):
    for civ in civs_order:
        c = COLORS_3[civ]
        dr = df_3_rule[df_3_rule["civilization"] == civ]
        dl = df_3_rl[df_3_rl["civilization"] == civ]
        ax.plot(dr["year"], dr[col], color=c, lw=2.0, label=f"{civ} rule")
        ax.plot(dl["year"], dl[col], color=c, lw=1.4, linestyle="--",
                alpha=0.8, label=f"{civ} RL")
    add_era_bands(ax)
    ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.2)

plot_ts(fig.add_subplot(gs[0, 0]), "gdp", "GDP", "GDP")
plot_ts(fig.add_subplot(gs[0, 1]), "gdp_per_capita", "GDP per Capita", "GDP/cap")
plot_ts(fig.add_subplot(gs[1, 0]), "tech_composite", "Tech Composite", "Tech")
plot_ts(fig.add_subplot(gs[1, 1]), "trade_income", "Trade Income", "Trade Inc.")

# ── RL 技术选择热力图 ─────────────────────────────────────
rl_civs_3 = [n for n, s in eng_3.strategy_map.items() if isinstance(s, QLearningStrategy)]
TECH_IDX   = {"agriculture": 0, "navigation": 1, "military": 2,
              "industry": 3, "commerce": 4}
TRADE_IDX  = {"open": 0, "balanced": 1, "closed": 2}

years_3 = sorted(df_3_rl["year"].unique())
nr      = len(rl_civs_3)

def make_mat(col, mapping):
    m = np.zeros((nr, len(years_3)))
    for ri, civ in enumerate(rl_civs_3):
        for ci, y in enumerate(years_3):
            r = df_3_rl[(df_3_rl["civilization"] == civ) & (df_3_rl["year"] == y)]
            if not r.empty:
                raw = r[col].values[0]
                m[ri, ci] = mapping.get(raw, mapping.get(int(raw) if str(raw).isdigit() else raw, 0))
    return m

ax_tech = fig.add_subplot(gs[2, :])
mat_t   = make_mat("decision_tech", TECH_IDX)
im = ax_tech.imshow(mat_t, aspect="auto", cmap="tab10", vmin=0, vmax=4,
                    interpolation="nearest")
ax_tech.set_yticks(range(nr)); ax_tech.set_yticklabels(rl_civs_3, fontsize=9)
xt = list(range(0, len(years_3), 2))
ax_tech.set_xticks(xt)
ax_tech.set_xticklabels([years_3[i] for i in xt], rotation=45, fontsize=7)
ax_tech.set_title("RL Tech Focus Heatmap  (0=Agri  1=Nav  2=Mil  3=Ind  4=Com)", fontsize=9)
cb = fig.colorbar(im, ax=ax_tech, orientation="vertical", pad=0.01, fraction=0.012)
cb.set_ticks([0,1,2,3,4]); cb.set_ticklabels(["Agri","Nav","Mil","Ind","Com"], fontsize=8)

plt.tight_layout()
p6 = os.path.join(DISC_DIR, "exp6_three_nation_rl.png")
plt.savefig(p6, dpi=150, bbox_inches="tight"); plt.close()
print(f"  -> {p6}")

print("  终局 GDP (rule vs RL):")
last_r = df_3_rule["year"].max(); last_l = df_3_rl["year"].max()
for civ in civs_order:
    g_r = df_3_rule[(df_3_rule["civilization"]==civ)&(df_3_rule["year"]==last_r)]["gdp"].values[0]
    g_l = df_3_rl[(df_3_rl["civilization"]==civ)&(df_3_rl["year"]==last_l)]["gdp"].values[0]
    print(f"  {civ}: rule={g_r:.2f}  RL={g_l:.2f}  Δ={g_l-g_r:+.2f}")

print("\n全部实验完成，图表保存在:", DISC_DIR)
