"""
run_experiments.py
------------------
Per-turn analysis and RL comparison for a given competition mode.

Usage:
    python run_experiments.py                            # 5-Nation, seed=42, 60 RL episodes
    python run_experiments.py --mode 1v1
    python run_experiments.py --mode 3-Nation --seed 7 --rl-episodes 80
    python run_experiments.py --no-events --rl-episodes 0   # rule-based only
"""

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
from src.strategies import make_strategy, QLearningStrategy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Display shorthands ────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────

def run_rule_based(mode: str, seed: int, events: bool) -> pd.DataFrame:
    civs = build_competition_civs(mode)
    engine = SimulationEngine(civs=civs, events_enabled=events, seed=seed)
    engine.run()
    return engine.get_history_df()


def run_rl(mode: str, seed: int, episodes: int):
    """Train RL agents, then run one final simulation. Returns (df, curves, engine)."""
    civs = build_competition_civs(mode)
    rl_reward_types = ["RL_gdp", "RL_power", "RL_trade"]
    strategy_map = {}
    for i, civ in enumerate(civs):
        strategy_map[civ.name] = (
            make_strategy(rl_reward_types[i]) if i < len(rl_reward_types)
            else make_strategy(civ.strategy_name)
        )
    engine = SimulationEngine(
        civs=civs, strategy_map=strategy_map,
        events_enabled=True, seed=seed, training_mode=True,
    )
    curves = engine.train_rl_agents(n_episodes=episodes)
    engine.run()
    return engine.get_history_df(), curves, engine


# ─────────────────────────────────────────────────────────────────────────
# Text output helpers
# ─────────────────────────────────────────────────────────────────────────

def _sep(n=68): return "─" * n


def print_turn_table(df: pd.DataFrame, mode: str) -> None:
    """Print per-turn GDP and tech_composite for all nations."""
    civs  = sorted(df["civilization"].unique())
    years = sorted(df["year"].unique())
    col_w = 10

    header = f"{'Year':<6}" + "".join(f"  {c:>{col_w}}" for c in civs)

    print(f"\n{'═'*68}")
    print(f"Mode: {mode}  —  per-turn snapshots")
    print('═'*68)

    for metric, label in [("gdp", "GDP"), ("tech_composite", "Tech")]:
        print(f"\n  {label}:")
        print(f"  {header}")
        print(f"  {_sep(6 + (col_w + 2) * len(civs))}")
        for y in years:
            snap = df[df["year"] == y]
            row  = f"  {y:<6}"
            for civ in civs:
                r = snap[snap["civilization"] == civ]
                v = r[metric].values[0] if not r.empty else float("nan")
                row += f"  {v:>{col_w}.3f}"
            print(row)


def print_decisions_table(df: pd.DataFrame, civ: str, label: str = "") -> None:
    """Print per-turn tech/trade/expand decisions and key metrics for one nation."""
    d   = df[df["civilization"] == civ].sort_values("year")
    tag = f"  [{label}]" if label else ""
    print(f"\n  {civ}{tag}")
    print(f"  {'Year':<6} {'Tech':>6} {'Trade':>6} {'Exp':>5}  "
          f"{'GDP':>8}  {'TradeInc':>9}  {'Terr':>6}  {'TechC':>7}")
    print(f"  {_sep(64)}")
    for _, r in d.iterrows():
        tech  = TECH_SHORT.get(r["decision_tech"],   str(r["decision_tech"]))
        trade = TRADE_SHORT.get(r["decision_trade"],  str(r["decision_trade"]))
        exp   = EXPAND_SHORT.get(int(r["decision_expand"]), str(int(r["decision_expand"])))
        print(f"  {int(r['year']):<6} {tech:>6} {trade:>6} {exp:>5}  "
              f"{r['gdp']:>8.2f}  {r['trade_income']:>9.4f}  "
              f"{r['territories']:>6.2f}  {r['tech_composite']:>7.3f}")


def print_rl_decision_summary(engine: SimulationEngine, rl_df: pd.DataFrame) -> None:
    """Detailed breakdown of how each RL agent made decisions."""
    print(f"\n{'═'*68}")
    print("RL Agent Decision Analysis")
    print('═'*68)

    for civ_name, strat in engine.strategy_map.items():
        if not isinstance(strat, QLearningStrategy):
            continue

        d = rl_df[rl_df["civilization"] == civ_name]

        print(f"\n  Agent : {civ_name}")
        print(f"  Reward: {strat.reward_type}")
        print(f"  Q-states explored: {len(strat.q_table)}")

        # Action frequency table
        tech_v  = d["decision_tech"].value_counts()
        trade_v = d["decision_trade"].value_counts()
        exp_v   = d["decision_expand"].value_counts()
        n       = len(d)

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

        # Era-level breakdown: which tech did RL prefer per era?
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

        # Per-turn decision trace
        print_decisions_table(rl_df, civ_name, label=f"RL/{strat.reward_type}")


# ─────────────────────────────────────────────────────────────────────────
# Chart output
# ─────────────────────────────────────────────────────────────────────────

def save_comparison_chart(rule_df: pd.DataFrame, rl_df: pd.DataFrame,
                          mode: str, rl_civs: list) -> str:
    """
    4-panel figure:
      - Top:    GDP rule-based (solid) vs RL (dashed) for all nations
      - Mid:    RL tech-focus heatmap (nation × year)
      - Bottom: RL trade-policy heatmap | RL expansion heatmap
    """
    fig = plt.figure(figsize=(17, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.28)

    # ── Panel 1: GDP comparison ──────────────────────────────────────────
    ax_gdp = fig.add_subplot(gs[0, :])
    civs   = sorted(rule_df["civilization"].unique())

    for i, civ in enumerate(civs):
        color  = PALETTE[i % len(PALETTE)]
        d_rule = rule_df[rule_df["civilization"] == civ]
        d_rl   = rl_df[rl_df["civilization"] == civ]
        ax_gdp.plot(d_rule["year"], d_rule["gdp"],
                    color=color, linewidth=2.2, label=f"{civ} rule")
        ls = "--" if civ in rl_civs else ":"
        ax_gdp.plot(d_rl["year"], d_rl["gdp"],
                    color=color, linewidth=1.6, linestyle=ls, alpha=0.75,
                    label=f"{civ} RL" if civ in rl_civs else "_")

    for x1, x2, c in ERA_BANDS:
        ax_gdp.axvspan(x1, x2, alpha=0.05, color=c)

    ax_gdp.set_xlabel("Year")
    ax_gdp.set_ylabel("GDP")
    ax_gdp.set_title(f"GDP — rule-based (solid) vs RL agents (dashed)  [{mode}]")
    ax_gdp.legend(fontsize=7.5, ncol=min(len(civs) * 2, 6), loc="upper left")
    ax_gdp.grid(alpha=0.25)

    if not rl_civs:
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"comparison_{mode.replace('-','_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    # ── Panels 2-4: RL decision heatmaps ────────────────────────────────
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
                    if raw in mapping:
                        m[ri, ci] = mapping[raw]
                    else:
                        try:
                            m[ri, ci] = mapping.get(int(raw), 0)
                        except (ValueError, TypeError):
                            m[ri, ci] = 0
        return m

    def _label_axes(ax, title, xtick_step=2):
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rl_civs, fontsize=8)
        xt = list(range(0, len(years), xtick_step))
        ax.set_xticks(xt)
        ax.set_xticklabels([years[i] for i in xt], rotation=45, fontsize=7)
        ax.set_title(title, fontsize=9)

    # Tech focus
    ax_tech = fig.add_subplot(gs[1, :])
    mat_tech = _make_matrix("decision_tech", TECH_IDX)
    im1 = ax_tech.imshow(mat_tech, aspect="auto", cmap="tab10",
                          vmin=0, vmax=4, interpolation="nearest")
    _label_axes(ax_tech, "Tech Focus  (0=Agri  1=Nav  2=Mil  3=Ind  4=Com)")
    cb1 = fig.colorbar(im1, ax=ax_tech, orientation="vertical",
                        pad=0.01, fraction=0.015)
    cb1.set_ticks([0, 1, 2, 3, 4])
    cb1.set_ticklabels(["Agri", "Nav", "Mil", "Ind", "Com"], fontsize=7)

    # Trade policy
    ax_trade = fig.add_subplot(gs[2, 0])
    mat_trade = _make_matrix("decision_trade", TRADE_ORDER)
    im2 = ax_trade.imshow(mat_trade, aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=2, interpolation="nearest")
    _label_axes(ax_trade, "Trade Policy  (0=Open  1=Balanced  2=Closed)")
    cb2 = fig.colorbar(im2, ax=ax_trade, pad=0.02, fraction=0.08)
    cb2.set_ticks([0, 1, 2])
    cb2.set_ticklabels(["Open", "Bal", "Cls"], fontsize=7)

    # Expansion
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


def save_turn_charts(rule_df: pd.DataFrame, mode: str) -> str:
    """Per-nation GDP + tech breakdown as small-multiples."""
    civs = sorted(rule_df["civilization"].unique())
    ncol = min(3, len(civs))
    nrow = -(-len(civs) // ncol)  # ceil division
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    fig.suptitle(f"Per-turn data — {mode}", fontsize=13)

    for idx, civ in enumerate(civs):
        ax  = axes[idx // ncol][idx % ncol]
        d   = rule_df[rule_df["civilization"] == civ]
        col = PALETTE[idx % len(PALETTE)]

        ax.plot(d["year"], d["gdp"], color=col, linewidth=2, label="GDP")
        ax2 = ax.twinx()
        ax2.plot(d["year"], d["tech_composite"], color=col, linewidth=1.2,
                 linestyle="--", alpha=0.6, label="Tech")
        ax2.set_ylabel("Tech", fontsize=8)

        for x1, x2, c in ERA_BANDS:
            ax.axvspan(x1, x2, alpha=0.06, color=c)

        ax.set_title(civ, fontsize=10, fontweight="bold", color=col)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("GDP", fontsize=8)
        ax.grid(alpha=0.2)

    # Hide empty panels
    for idx in range(len(civs), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"turns_{mode.replace('-','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Economy Sim — experiment runner")
    parser.add_argument("--mode",        default="5-Nation",
                        help=f"Competition mode. Choices: {list(COMPETITION_MODES.keys())}")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no-events",   action="store_true", help="Disable random events")
    parser.add_argument("--rl-episodes", type=int, default=60,
                        help="RL training episodes (0 = skip RL)")
    args = parser.parse_args()

    events = not args.no_events
    mode   = args.mode

    if mode not in COMPETITION_MODES:
        print(f"Unknown mode '{mode}'. Available: {list(COMPETITION_MODES.keys())}")
        sys.exit(1)

    plt.rcParams["font.family"] = ["DejaVu Sans"]

    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Mode       : {mode}  ({COMPETITION_MODES[mode]['desc']})")
    print(f"Seed       : {args.seed}   Events: {events}   RL episodes: {args.rl_episodes}")

    # ── 1. Rule-based run ────────────────────────────────────────────────
    print("\nRunning rule-based simulation...")
    rule_df = run_rule_based(mode, args.seed, events)

    print_turn_table(rule_df, mode)

    print(f"\n{'═'*68}")
    print("Rule-based — decisions per nation:")
    for civ in sorted(rule_df["civilization"].unique()):
        print_decisions_table(rule_df, civ, label="rule")

    # Save per-turn CSV and chart
    csv_path = os.path.join(OUTPUT_DIR, f"turns_{mode.replace('-','_')}.csv")
    rule_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved : {csv_path}")

    chart_path = save_turn_charts(rule_df, mode)
    print(f"Chart saved: {chart_path}")

    # ── 2. RL run ────────────────────────────────────────────────────────
    if args.rl_episodes > 0:
        print(f"\nTraining RL agents ({args.rl_episodes} episodes)...")
        rl_df, curves, rl_engine = run_rl(mode, args.seed, args.rl_episodes)

        rl_civs = [n for n, s in rl_engine.strategy_map.items()
                   if isinstance(s, QLearningStrategy)]

        # Training curve summary
        print(f"\n  Training improvement (ep-1 → ep-{args.rl_episodes}):")
        for civ, vals in curves.items():
            if vals:
                delta = (vals[-1] - vals[0]) / max(abs(vals[0]), 1e-6) * 100
                print(f"    {civ:<15}  start={vals[0]:.2f}  "
                      f"end={vals[-1]:.2f}  Δ={delta:+.1f}%")

        # Final GDP comparison
        print(f"\n{'═'*68}")
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

        # RL decision analysis
        print_rl_decision_summary(rl_engine, rl_df)

        # Save RL CSV and comparison chart
        rl_csv = os.path.join(OUTPUT_DIR, f"rl_turns_{mode.replace('-','_')}.csv")
        rl_df.to_csv(rl_csv, index=False)
        print(f"\nRL CSV saved : {rl_csv}")

        comp_path = save_comparison_chart(rule_df, rl_df, mode, rl_civs)
        print(f"Comparison chart: {comp_path}")

    print(f"\n{'═'*68}")
    print("Done.  To explore interactively:  streamlit run app.py")
    print('═'*68)
