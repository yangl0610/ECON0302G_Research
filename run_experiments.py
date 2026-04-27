"""
run_experiments.py
------------------
命令行实验运行器。无需打开 Streamlit，直接在终端运行，
输出文字报告和保存图表 PNG。

用法：
    python run_experiments.py                    # 跑所有实验
    python run_experiments.py --seed 42          # 指定随机种子
    python run_experiments.py --no-events        # 关闭随机事件
    python run_experiments.py --train-rl 80      # 训练 RL 80 轮
    python run_experiments.py --monte-carlo 20   # 蒙特卡洛 20 次模拟
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，可以保存图片
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import SimulationEngine, build_default_civs
from src.strategies import make_strategy


# ─────────────────────────────────────────────
# 输出目录
# ─────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 实验 1：历史基准线
# ─────────────────────────────────────────────
def experiment_baseline(seed: int, events: bool) -> pd.DataFrame:
    """
    跑一次历史基准模拟（使用默认 8 个文明和各自的规则式策略）。
    输出：各文明 GDP、人均 GDP、技术水平的摘要表。
    """
    print("\n" + "="*60)
    print("实验 1：历史基准线模拟")
    print(f"  随机种子={seed}，随机事件={'开启' if events else '关闭'}")
    print("="*60)

    engine = SimulationEngine(events_enabled=events, seed=seed)
    engine.run()
    df = engine.get_history_df()

    # 打印 1000 AD 和 1850 AD 的快照
    for year in [1000, 1400, 1600, 1750, 1850]:
        snap = df[df["year"] == year][["civilization", "gdp", "gdp_per_capita", "tech_composite", "territories"]]
        snap = snap.sort_values("gdp", ascending=False)
        print(f"\n  ─── {year} AD ───")
        print(snap.to_string(index=False))

    # 保存静态折线图
    _save_gdp_chart(df, "baseline_gdp.png", title="历史基准线：GDP 演变（1000-1850）")

    # 打印 1850 年 GDP 增长倍数
    print("\n  ─── 1850年 GDP 相对 1000年的增长倍数 ───")
    gdp_1000 = df[df["year"] == df["year"].min()].set_index("civilization")["gdp"]
    gdp_1850 = df[df["year"] == df["year"].max()].set_index("civilization")["gdp"]
    for civ in gdp_1850.index:
        if civ in gdp_1000.index and gdp_1000[civ] > 0:
            ratio = gdp_1850[civ] / gdp_1000[civ]
            print(f"    {civ:<20} {ratio:.1f}x")

    return df


# ─────────────────────────────────────────────
# 实验 2：策略互换（反事实）
# ─────────────────────────────────────────────
def experiment_strategy_swap(seed: int) -> None:
    """
    给中华帝国换上不同的策略，其他文明不变，
    观察其 GDP 轨迹的变化，评估策略选择对历史路径的影响力。
    """
    print("\n" + "="*60)
    print("实验 2：策略反事实——改变中华帝国的策略")
    print("="*60)

    strategies_to_test = [
        ("AgrarianConservative（历史实际）",  "AgrarianConservative"),
        ("MaritimeExpansionist（航海扩张）",  "MaritimeExpansionist"),
        ("IndustrialPioneer（工业先行）",      "IndustrialPioneer"),
        ("TradeHub（贸易枢纽）",               "TradeHub"),
    ]

    results = {}
    for label, strat_name in strategies_to_test:
        civs = build_default_civs()
        strat_map = {}
        for c in civs:
            if c.name == "中华帝国":
                strat_map[c.name] = make_strategy(strat_name)
            else:
                strat_map[c.name] = make_strategy(c.strategy_name)

        engine = SimulationEngine(civs=civs, strategy_map=strat_map, events_enabled=False, seed=seed)
        engine.run()
        df = engine.get_history_df()
        results[label] = df[df["civilization"] == "中华帝国"]

    # 打印 1850 年终值
    print("\n  中华帝国 1850年 GDP 对比：")
    for label, d in results.items():
        final = d[d["year"] == d["year"].max()]
        if not final.empty:
            print(f"    {label:<40} GDP={final['gdp'].values[0]:.2f}  "
                  f"人均GDP={final['gdp_per_capita'].values[0]:.4f}  "
                  f"工业技术={final['tech_ind'].values[0]:.2f}")

    # 保存对比图
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261"]
    for (label, d), color in zip(results.items(), colors):
        ax.plot(d["year"], d["gdp"], label=label, color=color, linewidth=2)

    # 添加时代背景色
    ax.axvspan(1000, 1400, alpha=0.05, color="blue",   label="_中世纪")
    ax.axvspan(1400, 1600, alpha=0.05, color="orange", label="_大航海")
    ax.axvspan(1600, 1750, alpha=0.05, color="red",    label="_重商主义")
    ax.axvspan(1750, 1850, alpha=0.05, color="purple", label="_工业革命")

    ax.set_xlabel("年份")
    ax.set_ylabel("GDP（相对单位）")
    ax.set_title("策略反事实：不同策略下中华帝国的 GDP 轨迹")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "counterfactual_china_strategy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存：{save_path}")


# ─────────────────────────────────────────────
# 实验 3：地理反事实
# ─────────────────────────────────────────────
def experiment_geography_swap(seed: int) -> None:
    """
    改变西北欧（荷英）的煤炭资源，研究工业革命是否必然发生在此。
    核心问题：如果英国没有煤炭，工业革命会在哪里发生？
    """
    print("\n" + "="*60)
    print("实验 3：地理反事实——煤炭对工业革命的影响")
    print("="*60)

    scenarios = {
        "历史实际（高煤炭：1.8）": 1.8,
        "中等煤炭（1.0）":          1.0,
        "低煤炭（0.3）":             0.3,
        "无煤炭（0.0）":             0.0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2a9d8f", "#457b9d", "#e9c46a", "#e63946"]

    for (label, coal_val), color in zip(scenarios.items(), colors):
        civs = build_default_civs()
        for c in civs:
            if c.name == "西北欧（荷英）":
                c.resources.coal = coal_val
        engine = SimulationEngine(civs=civs, events_enabled=False, seed=seed)
        engine.run()
        df = engine.get_history_df()

        d = df[df["civilization"] == "西北欧（荷英）"]
        axes[0].plot(d["year"], d["gdp"], label=label, color=color, linewidth=2)
        axes[1].plot(d["year"], d["tech_ind"], label=label, color=color, linewidth=2)

        final = d[d["year"] == d["year"].max()]
        print(f"  {label:<35} "
              f"GDP={final['gdp'].values[0]:.2f}  "
              f"工业技术={final['tech_ind'].values[0]:.2f}")

    for ax, ylabel, title in zip(
        axes,
        ["GDP（相对单位）", "工业技术水平（0-10）"],
        ["GDP 轨迹", "工业技术发展"]
    ):
        ax.set_xlabel("年份")
        ax.set_ylabel(ylabel)
        ax.set_title(f"西北欧（荷英）— {title}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axvspan(1750, 1850, alpha=0.08, color="purple")

    plt.suptitle("地理反事实：煤炭资源对工业革命的影响", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "counterfactual_coal.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存：{save_path}")


# ─────────────────────────────────────────────
# 实验 4：蒙特卡洛——评估历史偶然性
# ─────────────────────────────────────────────
def experiment_monte_carlo(n_runs: int = 20) -> None:
    """
    运行 N 次完全相同参数的模拟（只改变随机种子），
    观察 1850 年 GDP 排名的方差，
    从而定量评估"历史偶然性"对结局的影响程度。

    如果每次排名完全相同 → 结果高度决定论（地理/策略决定一切）
    如果排名高度随机     → 历史偶然性占主导
    """
    print("\n" + "="*60)
    print(f"实验 4：蒙特卡洛模拟（N={n_runs}）——评估历史偶然性")
    print("="*60)

    all_results = []
    for i in range(n_runs):
        engine = SimulationEngine(events_enabled=True, seed=i, noise_std=0.03)
        engine.run()
        df = engine.get_history_df()
        final = df[df["year"] == df["year"].max()][["civilization", "gdp", "gdp_per_capita"]]
        final["run"] = i
        all_results.append(final)

    all_df = pd.concat(all_results, ignore_index=True)

    # 计算各文明 GDP 的均值和标准差
    stats = all_df.groupby("civilization")["gdp"].agg(["mean", "std", "min", "max"])
    stats["cv"] = (stats["std"] / stats["mean"] * 100).round(1)  # 变异系数（%）
    stats = stats.sort_values("mean", ascending=False)

    print("\n  1850年 GDP 统计（N 次模拟）：")
    print(f"  {'文明':<20} {'均值':>8} {'标准差':>8} {'最小':>8} {'最大':>8} {'变异系数%':>10}")
    for civ, row in stats.iterrows():
        print(f"  {civ:<20} {row['mean']:>8.2f} {row['std']:>8.2f} "
              f"{row['min']:>8.2f} {row['max']:>8.2f} {row['cv']:>9.1f}%")

    # 计算每次模拟中 GDP 排名第一的文明
    rank1_counts = all_df.loc[all_df.groupby("run")["gdp"].idxmax(), "civilization"].value_counts()
    print("\n  N 次模拟中 GDP 第一的文明频次：")
    for civ, count in rank1_counts.items():
        print(f"    {civ:<25} {count}/{n_runs} 次 ({100*count/n_runs:.0f}%)")

    # 绘制箱线图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    civs_ordered = stats.index.tolist()
    data_for_box = [all_df[all_df["civilization"] == c]["gdp"].values for c in civs_ordered]

    ax1.boxplot(data_for_box, labels=[c.replace("（", "\n（") for c in civs_ordered], vert=True)
    ax1.set_ylabel("1850年 GDP")
    ax1.set_title(f"GDP 分布（N={n_runs} 次模拟）")
    ax1.tick_params(axis='x', labelsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 变异系数条形图（越高 = 历史偶然性越大）
    ax2.barh(civs_ordered, stats["cv"], color="#457b9d", alpha=0.8)
    ax2.set_xlabel("变异系数 CV（%）")
    ax2.set_title("历史偶然性敏感度\n（变异系数越高 = 结果越不确定）")
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(stats["cv"].mean(), color="red", linestyle="--", label=f"均值 {stats['cv'].mean():.1f}%")
    ax2.legend()

    plt.suptitle(f"蒙特卡洛分析：历史偶然性对 1850 年格局的影响（N={n_runs}）", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "monte_carlo.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存：{save_path}")

    # 总结
    avg_cv = stats["cv"].mean()
    print(f"\n  结论：平均变异系数 = {avg_cv:.1f}%")
    if avg_cv < 10:
        print("  → 历史结果高度稳定，地理/策略是主要决定因素（强决定论）")
    elif avg_cv < 25:
        print("  → 历史结果中等稳定，偶然性有影响但非决定性")
    else:
        print("  → 历史结果高度不确定，偶然性起重要作用")


# ─────────────────────────────────────────────
# 辅助：保存 GDP 折线图
# ─────────────────────────────────────────────
def _save_gdp_chart(df: pd.DataFrame, filename: str, title: str) -> None:
    colors = ["#e63946", "#457b9d", "#f4a261", "#2a9d8f",
              "#e9c46a", "#a8dadc", "#6d4c41", "#9c6644"]
    civs = df["civilization"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, civ in enumerate(civs):
        d = df[df["civilization"] == civ]
        ax.plot(d["year"], d["gdp"], label=civ, color=colors[i % len(colors)], linewidth=1.8)

    ax.axvspan(1000, 1400, alpha=0.05, color="blue")
    ax.axvspan(1400, 1600, alpha=0.05, color="orange")
    ax.axvspan(1600, 1750, alpha=0.05, color="red")
    ax.axvspan(1750, 1850, alpha=0.07, color="purple")

    for x, label in [(1000, "中世纪"), (1400, "大航海"), (1600, "重商主义"), (1750, "工业革命")]:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(x + 5, ax.get_ylim()[1] * 0.95, label, fontsize=8, color="gray", alpha=0.7)

    ax.set_xlabel("年份")
    ax.set_ylabel("GDP（相对单位）")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图表已保存：{save_path}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="世界经济形成模拟——实验运行器")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-events",    action="store_true", help="关闭随机事件")
    parser.add_argument("--train-rl",     type=int,   default=0, help="RL 训练轮数（0=跳过）")
    parser.add_argument("--monte-carlo",  type=int,   default=0, help="蒙特卡洛次数（0=跳过）")
    args = parser.parse_args()

    events = not args.no_events

    # 必须配置中文字体（不配置则中文会乱码）
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"输出目录：{OUTPUT_DIR}")

    # 实验 1
    experiment_baseline(args.seed, events)

    # 实验 2
    experiment_strategy_swap(args.seed)

    # 实验 3
    experiment_geography_swap(args.seed)

    # 实验 4（蒙特卡洛，可选）
    if args.monte_carlo > 0:
        experiment_monte_carlo(args.monte_carlo)

    print("\n" + "="*60)
    print("所有实验完成！图表已保存到 output/ 目录。")
    print("启动可视化仪表板：  streamlit run app.py")
    print("="*60)
