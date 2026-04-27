"""
app.py — 文明竞争模拟仪表板
运行方式：streamlit run app.py
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import SimulationEngine
from src.strategies import make_strategy
import json
from src.archetypes import (
    build_competition_civs, COMPETITION_MODES,
    archetype_descs, compute_trade_matrix, default_params,
)
from src.strategies import RULE_BASED_STRATEGIES

st.set_page_config(
    page_title="World Econ Sim",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 颜色与标签常量 ────────────────────────────────────────────────────────
TECH_LABELS = {
    "agriculture": "🌾 农业",
    "navigation":  "⛵ 航海",
    "military":    "⚔️ 军事",
    "industry":    "🏭 工业",
    "commerce":    "💰 商业",
}
TECH_COLORS = {
    "agriculture": "#2a9d8f",
    "navigation":  "#457b9d",
    "military":    "#e63946",
    "industry":    "#9c6644",
    "commerce":    "#e9c46a",
}
EXPAND_LABELS = {0: "保守", 1: "温和扩张", 2: "激进殖民"}
TRADE_LABELS  = {"open": "开放", "balanced": "均衡", "closed": "封闭"}

ERA_BAND_DF = pd.DataFrame([
    {"era": "农业时代",  "start": 1000, "end": 1400, "color": "#8ecae6"},
    {"era": "扩张时代",  "start": 1400, "end": 1600, "color": "#ffb703"},
    {"era": "商业时代",  "start": 1600, "end": 1750, "color": "#fb8500"},
    {"era": "工业时代",  "start": 1750, "end": 1860, "color": "#8338ec"},
])


def era_bands() -> alt.Chart:
    return (
        alt.Chart(ERA_BAND_DF)
        .mark_rect(opacity=0.08)
        .encode(
            x=alt.X("start:Q"), x2="end:Q",
            color=alt.Color(
                "era:N",
                scale=alt.Scale(
                    domain=["农业时代", "扩张时代", "商业时代", "工业时代"],
                    range=["#8ecae6", "#ffb703", "#fb8500", "#8338ec"],
                ),
                legend=alt.Legend(title="发展阶段", symbolOpacity=1),
            ),
            tooltip=alt.value(None),
        )
    )


def line_chart(
    df: pd.DataFrame,
    y: str,
    y_title: str,
    color_map: dict,
    height: int = 300,
    title: str = "",
    with_bands: bool = True,
) -> alt.LayerChart:
    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.3, point=alt.OverlayMarkDef(opacity=0, size=200))
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y(f"{y}:Q", title=y_title),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=alt.Legend(title="Nation", symbolSize=160, labelFontSize=12,
                                  symbolStrokeWidth=3, symbolOpacity=1),
            ),
            tooltip=["civilization:N", "year:Q", alt.Tooltip(f"{y}:Q", format=".3f")],
        )
    )
    base = alt.layer(era_bands(), lines) if with_bands else lines
    return base.properties(width="container", height=height, title=title).interactive()


# ── 缓存：运行模拟 ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="正在运行模拟...")
def run_simulation(mode_name: str, seed: int, events: bool, noise: float,
                   overrides_json: str = "{}"):
    overrides = json.loads(overrides_json)
    civs = build_competition_civs(mode_name, overrides)
    engine = SimulationEngine(civs=civs, events_enabled=events, noise_std=noise, seed=seed)
    engine.run()
    return engine.get_history_df(), engine.get_event_df()


@st.cache_data(show_spinner="正在训练 RL 智能体...")
def run_with_rl(mode_name: str, seed: int, n_episodes: int, overrides_json: str = "{}"):
    overrides = json.loads(overrides_json)
    civs = build_competition_civs(mode_name, overrides)
    rl_types = ["RL_gdp", "RL_power", "RL_trade"]
    strategy_map = {}
    for i, civ in enumerate(civs):
        strategy_map[civ.name] = (
            make_strategy(rl_types[i]) if i < len(rl_types)
            else make_strategy(civ.strategy_name)
        )
    engine = SimulationEngine(
        civs=civs, strategy_map=strategy_map,
        events_enabled=True, seed=seed, training_mode=True,
    )
    curves = engine.train_rl_agents(n_episodes=n_episodes)
    engine.run()
    return engine.get_history_df(), curves, engine.get_strategy_summary() if hasattr(engine, "get_strategy_summary") else {}


@st.cache_data(show_spinner="正在对比所有竞争模式...")
def run_all_modes(seed: int, events: bool, noise: float):
    results = {}
    for mode_name in COMPETITION_MODES:
        civs = build_competition_civs(mode_name)
        engine = SimulationEngine(civs=civs, events_enabled=events, noise_std=noise, seed=seed)
        engine.run()
        results[mode_name] = engine.get_history_df()
    return results


# ── 侧边栏 ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Sim Controls")
    st.markdown("---")
    mode_name = st.selectbox("Mode", list(COMPETITION_MODES.keys()))
    st.markdown("---")
    seed      = st.slider("Seed", 1, 200, 42)
    events_on = st.checkbox("Random events", value=True)
    noise_lvl = st.slider("Noise", 0.0, 0.08, 0.025, step=0.005)
    st.markdown("---")
    rl_eps = st.slider("RL episodes", 20, 150, 60, step=10)
    st.markdown("---")

    # ── 文明初始设置 ──────────────────────────────
    STRATEGY_OPTIONS = list(RULE_BASED_STRATEGIES.keys())
    GEO_PARAMS = {
        "coast_access":       ("海岸条件",     0.0, 1.0, 0.05),
        "terrain_quality":    ("耕地质量",     0.0, 1.0, 0.05),
        "strategic_location": ("战略位置",     0.0, 1.0, 0.05),
        "climate_score":      ("气候适宜度",   0.0, 1.0, 0.05),
        "river_density":      ("内河密度",     0.0, 1.0, 0.05),
    }
    RES_PARAMS = {
        "food":    ("粮食", 0.0, 4.0, 0.1),
        "metal":   ("金属", 0.0, 3.0, 0.1),
        "wood":    ("木材", 0.0, 3.0, 0.1),
        "luxury":  ("奢侈品", 0.0, 3.0, 0.1),
        "coal":    ("煤炭", 0.0, 3.0, 0.1),
    }

    defaults = default_params(mode_name)
    overrides = {}

    with st.expander("Setup", expanded=False):
        for civ_name, params in defaults.items():
            color = params["color"]
            st.markdown(
                f'<span style="color:{color};font-weight:bold;font-size:14px">'
                f'▍ {civ_name}</span>',
                unsafe_allow_html=True,
            )
            civ_ov = {"geo": {}, "res": {}}

            # 策略选择
            default_strat = params["strategy"]
            chosen_strat = st.selectbox(
                "策略", STRATEGY_OPTIONS,
                index=STRATEGY_OPTIONS.index(default_strat) if default_strat in STRATEGY_OPTIONS else 0,
                key=f"strat_{civ_name}",
            )
            if chosen_strat != default_strat:
                civ_ov["strategy"] = chosen_strat

            # 地理参数
            with st.popover("🗺️ 地理参数"):
                for key, (label, lo, hi, step) in GEO_PARAMS.items():
                    val = st.slider(
                        label, lo, hi,
                        float(round(params["geo"][key], 2)),
                        step=step, key=f"geo_{civ_name}_{key}",
                    )
                    if abs(val - params["geo"][key]) > 1e-6:
                        civ_ov["geo"][key] = val

            # 资源参数
            with st.popover("⛏️ 资源参数"):
                for key, (label, lo, hi, step) in RES_PARAMS.items():
                    val = st.slider(
                        label, lo, hi,
                        float(round(params["res"][key], 1)),
                        step=step, key=f"res_{civ_name}_{key}",
                    )
                    if abs(val - params["res"][key]) > 1e-6:
                        civ_ov["res"][key] = val

            if civ_ov["geo"] or civ_ov["res"] or "strategy" in civ_ov:
                overrides[civ_name] = civ_ov

        if overrides:
            st.info(f"{len(overrides)} overridden")
        if st.button("↺ Reset"):
            for key in list(st.session_state.keys()):
                if any(key.startswith(p) for p in ["geo_", "res_", "strat_"]):
                    del st.session_state[key]
            st.rerun()

    st.caption("Vega-Altair | Cobb-Douglas")

# ── 加载数据 ─────────────────────────────────────────────────────────────
overrides_json = json.dumps(overrides, sort_keys=True) if overrides else "{}"
df, event_df = run_simulation(mode_name, seed, events_on, noise_lvl, overrides_json)

_color_rows = df.drop_duplicates("civilization")[["civilization", "color"]]
COLOR_MAP    = dict(zip(_color_rows["civilization"], _color_rows["color"]))
CIVS         = list(COLOR_MAP.keys())
final_df     = df[df["year"] == df["year"].max()]

st.title(f"World Economy Sim — {mode_name}  ({COMPETITION_MODES[mode_name]['desc']})")
st.markdown("---")

winner = final_df.nlargest(1, "gdp").iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Top GDP", winner["civilization"])
c2.metric("GDP",     f"{winner['gdp']:.1f}")
c3.metric("Nations", len(CIVS))
c4.metric("Span",    "Era 1 → Era 4")

st.markdown("---")

# ── 标签页 ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "GDP",
    "Trade & Territory",
    "Relations",
    "Tech",
    "Mode Compare",
    "ML",
])


# ══════════════════════════════════════════════
# Tab 1：竞争总览
# ══════════════════════════════════════════════
with tab1:
    descs = archetype_descs()
    civ_info = [{"Nation": n, "Strategy": descs.get(n.split("·")[0], "")} for n in CIVS]
    st.dataframe(pd.DataFrame(civ_info), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Power Index")

    power_df = df.copy()
    power_df["power_raw"] = power_df["gdp"] * power_df["military_str"] * np.sqrt(power_df["territories"])
    power_df["power_idx"] = power_df.groupby("year")["power_raw"].transform(lambda x: x / x.max() * 100)

    power_chart = line_chart(power_df, "power_idx", "Power (max=100)", COLOR_MAP,
                             height=340, title="Power Index")
    st.altair_chart(power_chart, use_container_width=True)

    # 颜色图例
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:3px 8px;">'
        f'<span style="background:{c};width:16px;height:16px;border-radius:3px;'
        f'display:inline-block;margin-right:5px;"></span><span style="font-size:13px">{n}</span></span>'
        for n, c in COLOR_MAP.items()
    )
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;padding:4px 0">{swatches}</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("GDP Share")

    share_df = df.copy()
    share_df["share"] = share_df.groupby("year")["gdp"].transform(lambda x: x / x.sum())
    share_chart = (
        alt.Chart(share_df)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("share:Q", stack="normalize", title="GDP 份额",
                    axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="Nation", symbolOpacity=1),
            ),
            tooltip=[
                alt.Tooltip("civilization:N", title="Nation"),
                alt.Tooltip("year:Q",         title="Year"),
                alt.Tooltip("share:Q",         title="份额", format=".1%"),
                alt.Tooltip("gdp:Q",           title="GDP",  format=".2f"),
            ],
            order=alt.Order("gdp:Q", sort="descending"),
        )
        .properties(width="container", height=320)
        .interactive()
    )
    st.altair_chart(share_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Leader Timeline")

    leader_rows = []
    for y in sorted(power_df["year"].unique()):
        dy = power_df[power_df["year"] == y]
        leader = dy.loc[dy["power_raw"].idxmax(), "civilization"]
        leader_rows.append({"year": y, "leader": leader})
    leader_df = pd.DataFrame(leader_rows)

    leader_chart = (
        alt.Chart(leader_df)
        .mark_point(size=90, filled=True)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("leader:N", title="Leader",
                    sort=alt.EncodingSortField(field="year")),
            color=alt.Color(
                "leader:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="Nation", symbolOpacity=1),
            ),
            tooltip=["year:Q", "leader:N"],
        )
        .properties(width="container", height=max(120, len(CIVS) * 55))
        .interactive()
    )
    st.altair_chart(leader_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Final Rankings")
    final_power = power_df[power_df["year"] == power_df["year"].max()].copy()
    final_power = final_power.sort_values("power_raw", ascending=False)
    final_power["排名"] = range(1, len(final_power) + 1)
    display_cols = ["排名", "civilization", "gdp", "gdp_per_capita",
                    "military_str", "territories", "tech_composite", "trade_income"]
    rename_map = {
        "civilization": "文明", "gdp": "GDP", "gdp_per_capita": "人均GDP",
        "military_str": "军事实力", "territories": "领土", "tech_composite": "技术综合",
        "trade_income": "贸易收益",
    }
    st.dataframe(
        final_power[display_cols].rename(columns=rename_map).reset_index(drop=True),
        use_container_width=True,
    )

    if events_on and not event_df.empty:
        with st.expander("📋 随机扰动事件记录"):
            st.dataframe(
                event_df[["year", "era", "event", "target", "description"]].sort_values("year"),
                use_container_width=True, height=240,
            )


# ══════════════════════════════════════════════
# Tab 2：经济轨迹
# ══════════════════════════════════════════════
with tab2:

    # 四格布局
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            line_chart(df, "gdp", "GDP（相对单位）", COLOR_MAP, height=280, title="GDP 总量"),
            use_container_width=True,
        )
    with c2:
        st.altair_chart(
            line_chart(df, "gdp_per_capita", "人均 GDP", COLOR_MAP, height=280, title="人均 GDP"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(
            line_chart(df, "population", "人口（百万）", COLOR_MAP, height=280, title="人口规模"),
            use_container_width=True,
        )
    with c4:
        # 计算 GDP 增长率（对数差分）
        growth_df = df.copy().sort_values(["civilization", "year"])
        growth_df["gdp_growth"] = growth_df.groupby("civilization")["gdp"].pct_change() * 100
        growth_df = growth_df.dropna(subset=["gdp_growth"])
        st.altair_chart(
            line_chart(growth_df, "gdp_growth", "GDP 增长率（%）", COLOR_MAP,
                       height=280, title="GDP 增长率"),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader("Avg GDP by Era")
    era_map = {"MEDIEVAL": "农业时代", "DISCOVERY": "扩张时代",
               "MERCANTILE": "商业时代", "INDUSTRIAL": "工业时代"}
    df["era_label"] = df["era"].map(era_map)
    era_avg = df.groupby(["civilization", "era_label"])["gdp"].mean().reset_index()
    era_avg.columns = ["civilization", "era_label", "avg_gdp"]

    era_order = ["农业时代", "扩张时代", "商业时代", "工业时代"]
    era_chart = (
        alt.Chart(era_avg)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("era_label:N", title="发展阶段",
                    sort=era_order,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("avg_gdp:Q", title="阶段平均 GDP"),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="Nation", symbolOpacity=1),
            ),
            xOffset="civilization:N",
            tooltip=["civilization:N", "era_label:N",
                     alt.Tooltip("avg_gdp:Q", title="均值GDP", format=".2f")],
        )
        .properties(width="container", height=320, title="各阶段平均 GDP（分组柱状图）")
    )
    st.altair_chart(era_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Final Snapshot")
    metric_opt = st.radio(
        "指标",
        ["gdp", "gdp_per_capita", "population", "tech_composite"],
        horizontal=True,
        format_func=lambda x: {
            "gdp": "GDP总量", "gdp_per_capita": "人均GDP",
            "population": "人口", "tech_composite": "技术水平",
        }[x],
    )
    bar = (
        alt.Chart(final_df.sort_values(metric_opt, ascending=False))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("civilization:N", sort=alt.EncodingSortField(field=metric_opt, order="descending"),
                    title="Nation"),
            y=alt.Y(f"{metric_opt}:Q", title=metric_opt),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=None,
            ),
            tooltip=["civilization:N", f"{metric_opt}:Q"],
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(bar, use_container_width=True)

    st.markdown("---")

    st.subheader("GDP Breakdown  (base / trade / colonial)")
    decomp_df = df.copy()
    decomp_df["base_gdp"] = (
        decomp_df["gdp"] - decomp_df["trade_income"] - decomp_df["colonial_income"]
    ).clip(lower=0)
    decomp_long = pd.melt(
        decomp_df,
        id_vars=["civilization", "year"],
        value_vars=["base_gdp", "trade_income", "colonial_income"],
        var_name="来源",
        value_name="数值",
    )
    decomp_long["来源"] = decomp_long["来源"].map({
        "base_gdp": "国内基础", "trade_income": "贸易收益", "colonial_income": "殖民地收益",
    })
    ncivs = len(CIVS)
    facet_cols = 3 if ncivs >= 3 else ncivs
    panel_w = max(180, min(300, 900 // facet_cols))
    decomp_area = (
        alt.Chart(decomp_long)
        .mark_area(opacity=0.88)
        .encode(
            x=alt.X("year:Q", title="Year", axis=alt.Axis(labelFontSize=9)),
            y=alt.Y("数值:Q", title="GDP", stack="zero", axis=alt.Axis(labelFontSize=9)),
            color=alt.Color(
                "来源:N",
                scale=alt.Scale(
                    domain=["国内基础", "贸易收益", "殖民地收益"],
                    range=["#264653", "#2a9d8f", "#e9c46a"],
                ),
                legend=alt.Legend(title="来源", symbolOpacity=1),
            ),
            tooltip=["civilization:N", "year:Q", "来源:N",
                     alt.Tooltip("数值:Q", title="GDP", format=".3f")],
        )
        .properties(width=panel_w, height=160)
        .facet(facet=alt.Facet("civilization:N", header=alt.Header(titleFontSize=12)), columns=facet_cols)
    )
    st.altair_chart(decomp_area, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 3：贸易与领土
# ══════════════════════════════════════════════
with tab3:

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            line_chart(df, "trade_income", "贸易收益", COLOR_MAP, height=300, title="Trade Income"),
            use_container_width=True,
        )
    with c2:
        st.altair_chart(
            line_chart(df, "colonial_income", "殖民地收益", COLOR_MAP, height=300, title="殖民地收益演变"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(
            line_chart(df, "territories", "Territory", COLOR_MAP, height=300, title="Territory"),
            use_container_width=True,
        )
    with c4:
        st.altair_chart(
            line_chart(df, "trade_openness", "Openness", COLOR_MAP, height=300, title="Openness"),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader("Trade × GDP (final)")
    scatter = (
        alt.Chart(final_df)
        .mark_circle(opacity=0.85, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("trade_income:Q", title="Trade Income"),
            y=alt.Y("gdp:Q",          title="GDP"),
            size=alt.Size("population:Q", scale=alt.Scale(range=[150, 1500]),
                          legend=alt.Legend(title="人口", symbolOpacity=1)),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="Nation", symbolOpacity=1),
            ),
            tooltip=["civilization:N", "gdp:Q", "trade_income:Q",
                     "population:Q", "territories:Q", "trade_openness:Q"],
        )
        .properties(width="container", height=380)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")

    st.subheader("Resource Complementarity")
    trade_matrix = compute_trade_matrix(mode_name)
    heatmap = (
        alt.Chart(trade_matrix)
        .mark_rect(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("进口方:N", title="进口方"),
            y=alt.Y("出口方:N", title="出口方"),
            color=alt.Color(
                "互补度:Q",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="互补度", symbolOpacity=1),
            ),
            tooltip=["出口方:N", "进口方:N", alt.Tooltip("互补度:Q", format=".3f")],
        )
        .properties(width="container", height=max(250, len(CIVS) * 60))
    )
    text_layer = (
        alt.Chart(trade_matrix)
        .mark_text(fontSize=12, fontWeight="bold", color="white")
        .encode(
            x="进口方:N",
            y="出口方:N",
            text=alt.Text("互补度:Q", format=".2f"),
            opacity=alt.condition(
                alt.datum["互补度"] > 0.05, alt.value(1), alt.value(0)
            ),
        )
    )
    st.altair_chart((heatmap + text_layer).properties(width="container"), use_container_width=True)


# ══════════════════════════════════════════════
# Tab 4：国家关系
# ══════════════════════════════════════════════
with tab4:
    st.subheader("GDP Ratio")

    focus_civ = st.selectbox("选择参照文明", CIVS, key="focus_relation")
    focus_df  = df[df["civilization"] == focus_civ][["year", "gdp"]].rename(columns={"gdp": "gdp_focus"})

    ratio_rows = []
    for other in CIVS:
        if other == focus_civ:
            continue
        other_df = df[df["civilization"] == other][["year", "gdp"]].rename(columns={"gdp": "gdp_other"})
        merged = focus_df.merge(other_df, on="year")
        merged["ratio"] = merged["gdp_focus"] / merged["gdp_other"].replace(0, np.nan)
        merged["对比"] = f"{focus_civ} / {other}"
        ratio_rows.append(merged[["year", "ratio", "对比"]])
    ratio_df = pd.concat(ratio_rows, ignore_index=True)

    ratio_chart = (
        alt.Chart(ratio_df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(opacity=0, size=180))
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("ratio:Q", title="GDP ratio (focus/other)"),
            color=alt.Color("对比:N", legend=alt.Legend(title="Ratio", symbolOpacity=1)),
            tooltip=["对比:N", "year:Q", alt.Tooltip("ratio:Q", format=".2f")],
        )
    )
    rule = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
        color="gray", strokeDash=[6, 3], strokeWidth=1.5
    ).encode(y="y:Q")
    st.altair_chart(
        alt.layer(era_bands(), ratio_chart, rule)
        .properties(width="container", height=320)
        .interactive(),
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Military")
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            line_chart(df, "military_str", "Military", COLOR_MAP, height=280, title="Military"),
            use_container_width=True,
        )
    with c2:
        # 军事 / GDP 比率（军事效率）
        mil_eff = df.copy()
        mil_eff["mil_eff"] = mil_eff["military_str"] / mil_eff["gdp"].replace(0, np.nan)
        st.altair_chart(
            line_chart(mil_eff, "mil_eff", "军事/GDP 比", COLOR_MAP, height=280,
                       title="Military / GDP"),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader("Tech Focus")

    tech_color_scale = alt.Scale(
        domain=list(TECH_LABELS.keys()),
        range=list(TECH_COLORS.values()),
    )
    decision_chart = (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=0.3)
        .encode(
            x=alt.X("year:Q",            title="Year",
                    axis=alt.Axis(format="d", tickCount=10)),
            y=alt.Y("civilization:N",    title="Nation",
                    sort=list(reversed(CIVS))),
            color=alt.Color(
                "decision_tech:N",
                scale=tech_color_scale,
                legend=alt.Legend(
                    title="Tech Focus",
                    symbolOpacity=1,
                    labelExpr=(
                        "datum.value === 'agriculture' ? '🌾 农业' : "
                        "datum.value === 'navigation'  ? '⛵ 航海' : "
                        "datum.value === 'military'    ? '⚔️ 军事' : "
                        "datum.value === 'industry'    ? '🏭 工业' : "
                        "'💰 商业'"
                    ),
                ),
            ),
            tooltip=[
                "civilization:N", "year:Q",
                alt.Tooltip("decision_tech:N",   title="Tech Focus"),
                alt.Tooltip("decision_expand:Q", title="Expand"),
                alt.Tooltip("decision_trade:N",  title="Trade Policy"),
            ],
        )
        .properties(width="container", height=max(120, len(CIVS) * 48))
    )
    st.altair_chart(decision_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Trade Policy")

    trade_color_scale = alt.Scale(
        domain=["open", "balanced", "closed"],
        range=["#2a9d8f", "#e9c46a", "#e63946"],
    )
    trade_policy_chart = (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=0.3)
        .encode(
            x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d", tickCount=10)),
            y=alt.Y("civilization:N", title="Nation", sort=list(reversed(CIVS))),
            color=alt.Color(
                "decision_trade:N",
                scale=trade_color_scale,
                legend=alt.Legend(
                    title="Trade Policy", symbolOpacity=1,
                    labelExpr=(
                        "datum.value === 'open' ? '开放' : "
                        "datum.value === 'balanced' ? '均衡' : '封闭'"
                    ),
                ),
            ),
            tooltip=[
                "civilization:N", "year:Q",
                alt.Tooltip("decision_trade:N", title="Trade Policy"),
                alt.Tooltip("trade_openness:Q", title="实际开放度", format=".2f"),
            ],
        )
        .properties(width="container", height=max(120, len(CIVS) * 48))
    )
    st.altair_chart(trade_policy_chart, use_container_width=True)

    # ── 扩张等级演变 ───────────────────────────
    st.subheader("Expansion")
    expand_chart = (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=0.3)
        .encode(
            x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d", tickCount=10)),
            y=alt.Y("civilization:N", title="Nation", sort=list(reversed(CIVS))),
            color=alt.Color(
                "decision_expand:O",
                scale=alt.Scale(domain=[0, 1, 2], range=["#d1e8ff", "#457b9d", "#1d3557"]),
                legend=alt.Legend(
                    title="Expand", symbolOpacity=1,
                    labelExpr=(
                        "datum.value === 0 ? '保守' : "
                        "datum.value === 1 ? '温和扩张' : '激进殖民'"
                    ),
                ),
            ),
            tooltip=[
                "civilization:N", "year:Q",
                alt.Tooltip("decision_expand:O", title="Expand"),
                alt.Tooltip("territories:Q",     title="当前领土", format=".2f"),
            ],
        )
        .properties(width="container", height=max(120, len(CIVS) * 48))
    )
    st.altair_chart(expand_chart, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 5：技术路线
# ══════════════════════════════════════════════
with tab5:
    tech_sel = st.selectbox(
        "Tech domain",
        list(TECH_LABELS.keys()),
        format_func=lambda x: TECH_LABELS[x],
    )
    st.altair_chart(
        line_chart(df, f"tech_{tech_sel[:3]}", f"{TECH_LABELS[tech_sel]}（0-10）",
                   COLOR_MAP, height=300, title=f"{TECH_LABELS[tech_sel]} 发展轨迹"),
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("All Tech Domains")
    field_pairs = [
        ("tech_agri", "🌾 农业技术"),
        ("tech_nav",  "⛵ 航海技术"),
        ("tech_mil",  "⚔️ 军事技术"),
        ("tech_ind",  "🏭 工业技术"),
    ]
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    for col, (field, label) in zip([r1c1, r1c2, r2c1, r2c2], field_pairs):
        with col:
            st.altair_chart(
                line_chart(df, field, "水平(0-10)", COLOR_MAP, height=220, title=label),
                use_container_width=True,
            )
    st.altair_chart(
        line_chart(df, "tech_com", "水平(0-10)", COLOR_MAP, height=220, title="💰 商业技术"),
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Tech Radar")
    radar_civs = st.multiselect("Nations", CIVS, default=CIVS[:min(4, len(CIVS))])
    if radar_civs:
        domains_cn = {"tech_agri": "农业", "tech_nav": "航海", "tech_mil": "军事",
                      "tech_ind": "工业", "tech_com": "商业"}

        def make_radar_df(year_val):
            rows = []
            dy = df[df["year"] == year_val]
            for civ in radar_civs:
                row = dy[dy["civilization"] == civ]
                if row.empty:
                    continue
                for field, cn in domains_cn.items():
                    rows.append({
                        "civilization": civ, "domain": cn,
                        "value": row[field].values[0], "year": year_val,
                    })
            return pd.DataFrame(rows)

        col_r1, col_r2 = st.columns(2)
        for col_w, yr in [(col_r1, df["year"].min()), (col_r2, df["year"].max())]:
            rdf = make_radar_df(yr)
            radar = (
                alt.Chart(rdf)
                .mark_line(point=True, strokeWidth=2, filled=False)
                .encode(
                    theta=alt.Theta("domain:N", sort=list(domains_cn.values())),
                    radius=alt.Radius("value:Q", scale=alt.Scale(domain=[0, 10])),
                    color=alt.Color(
                        "civilization:N",
                        scale=alt.Scale(domain=list(COLOR_MAP.keys()),
                                        range=list(COLOR_MAP.values())),
                        legend=alt.Legend(title="Nation", symbolOpacity=1),
                    ),
                    tooltip=["civilization:N", "domain:N", "value:Q"],
                )
            )
            radar_fill = radar.mark_arc(opacity=0.1, innerRadius=0)
            era_label = "Start" if yr == df["year"].min() else "End"
            with col_w:
                st.altair_chart(
                    alt.layer(radar_fill, radar)
                    .properties(width=340, height=340, title=f"{era_label}"),
                    use_container_width=True,
                )

    st.markdown("---")

    st.subheader("Industrial Tech Leader")
    ind_leader_rows = []
    for y in sorted(df["year"].unique()):
        dy = df[df["year"] == y]
        leader = dy.loc[dy["tech_ind"].idxmax(), "civilization"]
        ind_leader_rows.append({"year": y, "leader": leader})
    ind_leader_df = pd.DataFrame(ind_leader_rows)
    ind_chart = (
        alt.Chart(ind_leader_df)
        .mark_point(size=80, filled=True)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("leader:N", title="Industrial leader"),
            color=alt.Color(
                "leader:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=None,
            ),
            tooltip=["year:Q", "leader:N"],
        )
        .properties(width="container", height=max(100, len(CIVS) * 50))
        .interactive()
    )
    st.altair_chart(ind_chart, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 6：模式对比
# ══════════════════════════════════════════════
with tab6:

    if st.button("Run all modes", type="primary"):
        all_results = run_all_modes(seed, events_on, noise_lvl)

        # 整合终局数据
        comparison_rows = []
        for m_name, m_df in all_results.items():
            final_m = m_df[m_df["year"] == m_df["year"].max()]
            for _, row in final_m.iterrows():
                arch = row["civilization"].split("·")[0]
                comparison_rows.append({
                    "Mode": m_name,
                    "Nation": row["civilization"],
                    "原型": arch,
                    "GDP": round(row["gdp"], 2),
                    "人均GDP": round(row["gdp_per_capita"], 4),
                    "技术综合": round(row["tech_composite"], 3),
                    "领土": round(row["territories"], 2),
                    "贸易收益": round(row["trade_income"], 3),
                })
        comp_df = pd.DataFrame(comparison_rows)

        # 各模式冠军
        st.subheader("各模式终局冠军")
        champ_rows = []
        for m_name, m_df in all_results.items():
            final_m = m_df[m_df["year"] == m_df["year"].max()]
            champ = final_m.nlargest(1, "gdp").iloc[0]
            power = final_m.copy()
            power["power"] = power["gdp"] * power["military_str"] * np.sqrt(power["territories"])
            power_champ = power.nlargest(1, "power").iloc[0]
            champ_rows.append({
                "Mode": m_name,
                "GDP #1": champ["civilization"],
                "GDP": round(champ["gdp"], 2),
                "Power #1": power_champ["civilization"],
            })
        st.dataframe(pd.DataFrame(champ_rows), use_container_width=True)

        st.markdown("---")

        gdp_chart = (
            alt.Chart(comp_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Mode:N"),
                y=alt.Y("GDP:Q"),
                color=alt.Color("原型:N", legend=alt.Legend(title="Nation", symbolOpacity=1)),
                xOffset="Nation:N",
                tooltip=["Mode:N", "Nation:N",
                         alt.Tooltip("GDP:Q", format=".2f"),
                         alt.Tooltip("人均GDP:Q", format=".4f")],
            )
            .properties(width="container", height=320)
        )
        st.altair_chart(gdp_chart, use_container_width=True)

        st.markdown("---")

        heatmap_df = comp_df.groupby(["原型", "Mode"])["GDP"].sum().reset_index()
        heatmap_chart = (
            alt.Chart(heatmap_df)
            .mark_rect(stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X("Mode:N"),
                y=alt.Y("原型:N", title="Nation"),
                color=alt.Color("GDP:Q", scale=alt.Scale(scheme="orangered"),
                                legend=alt.Legend(title="GDP", symbolOpacity=1)),
                tooltip=["原型:N", "Mode:N", alt.Tooltip("GDP:Q", format=".2f")],
            )
            .properties(width="container", height=240)
        )
        text_hm = (
            alt.Chart(heatmap_df)
            .mark_text(fontSize=11, fontWeight="bold", color="white")
            .encode(x="Mode:N", y="原型:N", text=alt.Text("GDP:Q", format=".1f"))
        )
        st.altair_chart((heatmap_chart + text_hm), use_container_width=True)

        st.markdown("---")
        st.dataframe(comp_df.sort_values(["Mode", "GDP"], ascending=[True, False]),
                     use_container_width=True)
    else:
        st.info("Run all 5 modes and compare (30-60s).")


# ══════════════════════════════════════════════
# Tab 7：ML 策略分析
# ══════════════════════════════════════════════
with tab7:
    rl_agent_rows = [
        {"Agent": CIVS[i] if i < len(CIVS) else "—", "Reward": r}
        for i, r in enumerate(["max GDP", "max Power (GDP×mil×terr)", "max Trade"])
    ]
    st.dataframe(pd.DataFrame(rl_agent_rows), hide_index=True, use_container_width=False)

    if st.button("Train RL agents", type="primary"):
        with st.spinner(f"Training {rl_eps} episodes..."):
            try:
                rl_df, curves, _ = run_with_rl(mode_name, seed, rl_eps, overrides_json)
                st.success("Done.")

                # 训练曲线
                st.subheader("Training Curve")
                curve_rows = []
                for civ_name, vals in curves.items():
                    for ep, v in enumerate(vals):
                        curve_rows.append({"episode": ep + 1, "gdp": v, "agent": civ_name})
                curve_df = pd.DataFrame(curve_rows)

                if not curve_df.empty:
                    curve_df["smooth"] = (
                        curve_df.groupby("agent")["gdp"]
                        .transform(lambda x: x.rolling(5, min_periods=1).mean())
                    )
                    raw_line = (
                        alt.Chart(curve_df)
                        .mark_line(opacity=0.3, strokeWidth=1)
                        .encode(
                            x=alt.X("episode:Q", title="Episode"),
                            y=alt.Y("gdp:Q",     title="Final GDP"),
                            color=alt.Color("agent:N", legend=alt.Legend(title="Agent", symbolOpacity=1)),
                        )
                    )
                    smooth_line = (
                        alt.Chart(curve_df)
                        .mark_line(strokeWidth=2.5)
                        .encode(
                            x="episode:Q",
                            y=alt.Y("smooth:Q", title="Final GDP"),
                            color=alt.Color("agent:N", legend=None),
                            tooltip=["agent:N", "episode:Q",
                                     alt.Tooltip("smooth:Q", title="GDP (5-ep avg)", format=".2f")],
                        )
                    )
                    st.altair_chart(
                        (raw_line + smooth_line)
                        .properties(width="container", height=300,
                                    title="Training curve (bold = 5-ep moving avg)"),
                        use_container_width=True,
                    )

                st.subheader("RL vs Rule-based: GDP")
                rl_target_civs = CIVS[:min(3, len(CIVS))]
                compare_rows = []
                for civ in rl_target_civs:
                    d_rl = rl_df[rl_df["civilization"] == civ].copy()
                    d_rl["类型"] = "RL 策略"
                    d_rule = df[df["civilization"] == civ].copy()
                    d_rule["类型"] = "规则式策略"
                    compare_rows.extend([d_rl, d_rule])
                cmp_df = pd.concat(compare_rows, ignore_index=True)

                cmp_chart = (
                    alt.Chart(cmp_df)
                    .mark_line(strokeWidth=2.2)
                    .encode(
                        x=alt.X("year:Q", title="Year"),
                        y=alt.Y("gdp:Q",  title="GDP"),
                        color=alt.Color(
                            "civilization:N",
                            scale=alt.Scale(domain=list(COLOR_MAP.keys()),
                                            range=list(COLOR_MAP.values())),
                            legend=alt.Legend(title="Nation", symbolOpacity=1),
                        ),
                        strokeDash=alt.StrokeDash(
                            "类型:N",
                            scale=alt.Scale(
                                domain=["RL 策略", "规则式策略"],
                                range=[[1, 0], [6, 3]],
                            ),
                            legend=alt.Legend(title="策略类型", symbolOpacity=1),
                        ),
                        tooltip=["civilization:N", "类型:N", "year:Q", "gdp:Q"],
                    )
                    .properties(width="container", height=320,
                                title="solid=RL, dashed=rule-based")
                    .interactive()
                )
                st.altair_chart(
                    alt.layer(era_bands(), cmp_chart), use_container_width=True
                )

                st.markdown("---")
                st.subheader("Agent Decisions")
                rl_agents = CIVS[:min(3, len(CIVS))]
                rl_dec_df = rl_df[rl_df["civilization"].isin(rl_agents)].copy()

                # 技术重点热力图
                hm_tech = (
                    alt.Chart(rl_dec_df)
                    .mark_rect(stroke="white", strokeWidth=0.4)
                    .encode(
                        x=alt.X("year:O", title="Year",
                                axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                        y=alt.Y("civilization:N", title="Agent"),
                        color=alt.Color(
                            "decision_tech:N",
                            scale=alt.Scale(
                                domain=list(TECH_LABELS.keys()),
                                range=list(TECH_COLORS.values()),
                            ),
                            legend=alt.Legend(title="Tech Focus", symbolOpacity=1,
                                             labelExpr=(
                                                 "datum.label == 'agriculture' ? '🌾 农业' : "
                                                 "datum.label == 'navigation'  ? '⛵ 航海' : "
                                                 "datum.label == 'military'    ? '⚔️ 军事' : "
                                                 "datum.label == 'industry'    ? '🏭 工业' : '💰 商业'"
                                             )),
                        ),
                        tooltip=["civilization:N", "year:Q", "decision_tech:N",
                                 alt.Tooltip("gdp:Q", title="GDP", format=".2f")],
                    )
                    .properties(width="container", height=max(60, 30 * len(rl_agents)),
                                title="技术投资方向（颜色=被强化的技术领域）")
                )
                st.altair_chart(hm_tech, use_container_width=True)

                # 贸易政策热力图
                TRADE_COLORS_MAP = {"open": "#2a9d8f", "balanced": "#e9c46a", "closed": "#e63946"}
                hm_trade = (
                    alt.Chart(rl_dec_df)
                    .mark_rect(stroke="white", strokeWidth=0.4)
                    .encode(
                        x=alt.X("year:O", title="Year",
                                axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                        y=alt.Y("civilization:N", title="Agent"),
                        color=alt.Color(
                            "decision_trade:N",
                            scale=alt.Scale(
                                domain=list(TRADE_COLORS_MAP.keys()),
                                range=list(TRADE_COLORS_MAP.values()),
                            ),
                            legend=alt.Legend(title="Trade Policy", symbolOpacity=1,
                                             labelExpr=(
                                                 "datum.label == 'open'   ? '开放' : "
                                                 "datum.label == 'closed' ? '封闭' : '均衡'"
                                             )),
                        ),
                        tooltip=["civilization:N", "year:Q", "decision_trade:N",
                                 alt.Tooltip("trade_income:Q", title="Trade Income", format=".3f")],
                    )
                    .properties(width="container", height=max(60, 30 * len(rl_agents)),
                                title="贸易政策选择（绿=开放 / 黄=均衡 / 红=封闭）")
                )
                st.altair_chart(hm_trade, use_container_width=True)

                # 扩张等级热力图
                EXPAND_COLORS_MAP = {0: "#264653", 1: "#e9c46a", 2: "#e76f51"}
                hm_expand = (
                    alt.Chart(rl_dec_df)
                    .mark_rect(stroke="white", strokeWidth=0.4)
                    .encode(
                        x=alt.X("year:O", title="Year",
                                axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                        y=alt.Y("civilization:N", title="Agent"),
                        color=alt.Color(
                            "decision_expand:O",
                            scale=alt.Scale(
                                domain=[0, 1, 2],
                                range=["#264653", "#e9c46a", "#e76f51"],
                            ),
                            legend=alt.Legend(title="Expand", symbolOpacity=1,
                                             labelExpr=(
                                                 "datum.label == '0' ? '保守' : "
                                                 "datum.label == '1' ? '温和' : '激进'"
                                             )),
                        ),
                        tooltip=["civilization:N", "year:Q", "decision_expand:O",
                                 alt.Tooltip("territories:Q", title="领土", format=".2f")],
                    )
                    .properties(width="container", height=max(60, 30 * len(rl_agents)),
                                title="Expansion (dark=conservative / yellow=moderate / orange=aggressive)")
                )
                st.altair_chart(hm_expand, use_container_width=True)

                st.markdown("---")
                st.subheader("Action Frequency")
                freq_rows = []
                for agent in rl_agents:
                    d = rl_df[rl_df["civilization"] == agent]
                    for tech_val, cnt in d["decision_tech"].value_counts().items():
                        freq_rows.append({
                            "Agent": agent, "Dim": "Tech",
                            "Choice": TECH_LABELS.get(tech_val, tech_val), "Count": int(cnt),
                        })
                    for trade_val, cnt in d["decision_trade"].value_counts().items():
                        freq_rows.append({
                            "Agent": agent, "Dim": "Trade",
                            "Choice": TRADE_LABELS.get(trade_val, str(trade_val)), "Count": int(cnt),
                        })
                    for exp_val, cnt in d["decision_expand"].value_counts().items():
                        freq_rows.append({
                            "Agent": agent, "Dim": "Expand",
                            "Choice": EXPAND_LABELS.get(int(exp_val), str(exp_val)), "Count": int(cnt),
                        })
                if freq_rows:
                    freq_df = pd.DataFrame(freq_rows)
                    freq_chart = (
                        alt.Chart(freq_df)
                        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                        .encode(
                            x=alt.X("Count:Q"),
                            y=alt.Y("Choice:N", sort="-x"),
                            color=alt.Color(
                                "Agent:N",
                                scale=alt.Scale(domain=list(COLOR_MAP.keys()),
                                               range=list(COLOR_MAP.values())),
                                legend=alt.Legend(title="Agent", symbolOpacity=1),
                            ),
                            row=alt.Row("Dim:N", header=alt.Header(labelFontSize=12)),
                            tooltip=["Agent:N", "Dim:N", "Choice:N", "Count:Q"],
                        )
                        .properties(width=420, height=100)
                        .resolve_scale(y="independent")
                    )
                    st.altair_chart(freq_chart, use_container_width=False)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Train RL agents (~15-40s).")

st.markdown("---")
st.caption("ECON0302G | Cobb-Douglas + Q-Learning | Vega-Altair")
