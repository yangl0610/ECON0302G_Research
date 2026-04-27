"""
app.py — Streamlit 可视化仪表板（Vega-Altair 版）
--------------------------------------------------
运行方式：
    streamlit run app.py

所有图表均使用 Vega-Altair（通过 VegaLite 规范渲染），
声明式语法将数据映射到视觉属性，无需手动管理坐标系。

标签页：
  🌍 世界概览    — GDP 份额面积图 + 经济规模散点图
  📈 经济轨迹    — GDP / 人均 GDP / 人口 / 军事 4 面板折线图
  🔬 技术竞赛    — 各技术领域折线图 + 技术结构雷达图
  🚢 贸易与殖民  — 贸易收益 + 殖民收益 + 开放度
  🔀 反事实分析  — 4 组历史假设实验对比
  🤖 ML 策略分析 — Q-learning 训练曲线 + 策略涌现行为
"""

import streamlit as st
import altair as alt
from vega_datasets import data as vega_data
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import SimulationEngine, build_default_civs
from src.strategies import make_strategy
from src.geography import build_control_df

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="世界经济形成模拟",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 辅助：历史时期背景色带（Altair mark_rect 图层）
# ─────────────────────────────────────────────
ERA_BAND_DF = pd.DataFrame([
    {"era": "中世纪",    "start": 1000, "end": 1400, "color": "#8ecae6"},
    {"era": "地理大发现","start": 1400, "end": 1600, "color": "#ffb703"},
    {"era": "重商主义",  "start": 1600, "end": 1750, "color": "#fb8500"},
    {"era": "工业革命",  "start": 1750, "end": 1860, "color": "#8338ec"},
])

def era_bands() -> alt.Chart:
    """
    返回一个半透明的时代背景色带图层。
    通过 Altair 的 layer() 叠加到主折线图上，
    让读者一眼看出每条轨迹处于哪个历史时期。
    """
    return (
        alt.Chart(ERA_BAND_DF)
        .mark_rect(opacity=0.08)
        .encode(
            x=alt.X("start:Q", title="年份"),
            x2="end:Q",
            color=alt.Color(
                "era:N",
                scale=alt.Scale(
                    domain=["中世纪", "地理大发现", "重商主义", "工业革命"],
                    range=["#8ecae6", "#ffb703", "#fb8500", "#8338ec"],
                ),
                legend=alt.Legend(title="历史时期"),
            ),
        )
    )


def line_with_bands(
    df: pd.DataFrame,
    y_field: str,
    y_title: str,
    color_field: str = "civilization:N",
    color_map: dict = None,
    height: int = 300,
    title: str = "",
) -> alt.LayerChart:
    """
    折线图 + 时代背景色带的通用构造器。

    Altair 的 layer() 将多个图表叠加为同一坐标系：
      layer[0] = 半透明色带（era_bands）
      layer[1] = 折线（每个文明一条）

    color_map: {文明名: 颜色代码} 字典，用于固定颜色
    """
    # 折线图主体
    line_kwargs = dict(
        x=alt.X("year:Q", title="年份"),
        y=alt.Y(f"{y_field}:Q", title=y_title),
        tooltip=["civilization:N", "year:Q", f"{y_field}:Q"],
    )
    if color_map:
        civs  = list(color_map.keys())
        hexes = list(color_map.values())
        line_kwargs["color"] = alt.Color(
            color_field,
            scale=alt.Scale(domain=civs, range=hexes),
            legend=alt.Legend(title="文明"),
        )
    else:
        line_kwargs["color"] = alt.Color(color_field, legend=alt.Legend(title="文明"))

    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.2, point=False)
        .encode(**line_kwargs)
    )

    return (
        alt.layer(era_bands(), lines)
        .properties(width="container", height=height, title=title)
        .interactive()  # 允许缩放/拖拽
    )


# ─────────────────────────────────────────────
# 缓存：避免每次交互重跑模拟
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="正在运行历史模拟...")
def run_baseline(seed: int, events: bool, noise: float):
    engine = SimulationEngine(events_enabled=events, noise_std=noise, seed=seed)
    engine.run()
    return engine.get_history_df(), engine.get_event_df()


@st.cache_data(show_spinner="正在训练 RL 智能体并运行模拟...")
def run_with_rl(seed: int, n_episodes: int):
    civs = build_default_civs()
    strategy_map = {}
    for civ in civs:
        if civ.name == "西北欧（荷英）":
            strategy_map[civ.name] = make_strategy("RL_gdp")
        elif civ.name == "伊比利亚（葡西）":
            strategy_map[civ.name] = make_strategy("RL_power")
        elif civ.name == "奥斯曼帝国":
            strategy_map[civ.name] = make_strategy("RL_trade")
        else:
            strategy_map[civ.name] = make_strategy(civ.strategy_name)

    engine = SimulationEngine(
        civs=civs, strategy_map=strategy_map,
        events_enabled=True, seed=seed, training_mode=True,
    )
    curves       = engine.train_rl_agents(n_episodes=n_episodes)
    engine.run()
    return engine.get_history_df(), curves, engine.get_strategy_summary()


@st.cache_data(show_spinner="正在运行反事实实验...")
def run_counterfactuals(seed: int):
    results = {}

    # 基准线
    e0 = SimulationEngine(civs=build_default_civs(), events_enabled=False, seed=seed)
    e0.run(); results["历史基准"] = e0.get_history_df()

    # 反事实 1：中国高海岸
    civs1 = build_default_civs()
    for c in civs1:
        if c.name == "中华帝国":
            c.geography.coast_access = 0.90
            c.geography.strategic_location = 0.70
    e1 = SimulationEngine(civs=civs1, events_enabled=False, seed=seed)
    e1.run(); results["中国高海岸（郑和路线）"] = e1.get_history_df()

    # 反事实 2：中国工业先行策略
    civs2 = build_default_civs()
    sm2 = {}
    for c in civs2:
        sm2[c.name] = make_strategy("IndustrialPioneer" if c.name == "中华帝国" else c.strategy_name)
    e2 = SimulationEngine(civs=civs2, strategy_map=sm2, events_enabled=False, seed=seed)
    e2.run(); results["中国工业先行策略"] = e2.get_history_df()

    # 反事实 3：西北欧无煤炭
    civs3 = build_default_civs()
    for c in civs3:
        if c.name == "西北欧（荷英）":
            c.resources.coal = 0.1
    e3 = SimulationEngine(civs=civs3, events_enabled=False, seed=seed)
    e3.run(); results["西北欧无煤炭资源"] = e3.get_history_df()

    return results


# ─────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🎮 模拟控制")
    st.markdown("---")
    seed       = st.slider("随机种子", 1, 200, 42)
    events_on  = st.checkbox("开启历史随机事件", value=True)
    noise_lvl  = st.slider("历史偶然性强度（噪声）", 0.0, 0.08, 0.025, step=0.005)
    st.markdown("---")
    rl_eps = st.slider("Q-learning 训练轮数", 20, 150, 60, step=10)
    st.markdown("---")
    ALL_CIVS = ["中华帝国","西欧诸国","伊比利亚（葡西）","西北欧（荷英）",
                "奥斯曼帝国","印度次大陆","撒哈拉以南非洲","美洲文明"]
    show_civs = st.multiselect(
        "显示哪些文明",
        ALL_CIVS,
        default=["中华帝国","西欧诸国","伊比利亚（葡西）","西北欧（荷英）","奥斯曼帝国"],
    )
    st.markdown("---")
    st.caption("📊 图表引擎：Vega-Altair")
    st.caption("📚 经济模型：Cobb-Douglas + Q-Learning")

# ─────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────
st.title("🌍 世界经济形成过程模拟")
st.markdown("""
> **研究问题**：大航海时代的航线选择、工业革命时期的技术路线——
> 这些关键节点的不同决策，如何塑造了今天的世界经济格局？

本模拟通过简化版"文明博弈"，结合 **Q-Learning 强化学习**，
推演地理禀赋 × 策略选择 × 历史偶然性对世界经济分化的贡献。
""")
st.markdown("---")

df, event_df = run_baseline(seed, events_on, noise_lvl)

# 颜色映射（固定，保证各图一致）
_color_rows = df.drop_duplicates("civilization")[["civilization","color"]]
COLOR_MAP   = dict(zip(_color_rows["civilization"], _color_rows["color"]))

df_flt = df[df["civilization"].isin(show_civs)] if show_civs else df

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 世界概览",
    "📈 经济轨迹",
    "🔬 技术竞赛",
    "🚢 贸易与殖民",
    "🔀 反事实分析",
    "🤖 ML 策略分析",
])


# ═════════════════════════════════════════════
# Tab 1：世界概览
# ═════════════════════════════════════════════
with tab1:
    st.header("🌍 世界经济格局演变概览")

    # 指标摘要
    final = df[df["year"] == df["year"].max()]
    top   = final.nlargest(1, "gdp").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1850年经济最强", top["civilization"])
    c2.metric("最高 GDP",       f"{top['gdp']:.1f}")
    c3.metric("全球 GDP 总量",  f"{final['gdp'].sum():.1f}")
    c4.metric("模拟跨度",       "1000 — 1850 年")

    st.markdown("---")

    # ── 全球 GDP 份额堆叠面积图 ──────────────────────────────
    # 用 Altair mark_area + stack="normalize" 展示各文明占全球份额
    st.subheader("全球 GDP 份额演变（权力转移图）")
    st.caption("纵轴为 100%；某色块面积越大，该文明在全球经济中的份额越高。"
               "工业时代西北欧的份额急剧扩大，清晰呈现『大分流』。")

    # 计算份额
    df_share = df.copy()
    df_share["share"] = df_share.groupby("year")["gdp"].transform(lambda x: x / x.sum())
    df_share_flt = df_share[df_share["civilization"].isin(show_civs or ALL_CIVS)]

    # Altair 堆叠面积图
    share_chart = (
        alt.Chart(df_share_flt)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("year:Q", title="年份", scale=alt.Scale(domain=[1000, 1860])),
            y=alt.Y("share:Q", stack="normalize", title="GDP 全球份额",
                    axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="文明"),
            ),
            tooltip=[
                alt.Tooltip("civilization:N", title="文明"),
                alt.Tooltip("year:Q",         title="年份"),
                alt.Tooltip("share:Q",         title="份额", format=".1%"),
                alt.Tooltip("gdp:Q",           title="GDP",  format=".2f"),
            ],
            order=alt.Order("gdp:Q", sort="descending"),
        )
        .properties(width="container", height=380, title="全球 GDP 份额演变（1000-1850）")
        .interactive()
    )
    st.altair_chart(share_chart, use_container_width=True)

    # ── 世界领土控制地图（真实地理 + 年份滑块）────────────────────
    st.subheader("领土控制世界地图")
    st.caption(
        "每个国家/地区按控制它的文明着色。"
        "殖民扩张随 territories 标量增长，按历史优先级依次染色。"
        "拖动滑块观察不同年份的世界格局演变。"
    )

    # 年份滑块
    available_years = sorted(df["year"].unique().tolist())
    map_year = st.select_slider(
        "选择年份",
        options=available_years,
        value=available_years[-1],
        format_func=lambda y: f"{y} AD",
    )

    # 1. 根据当年领土数据构建 id → civilization 映射表
    map_year_df  = df[df["year"] == map_year]
    control_df   = build_control_df(map_year_df)

    # 2. 底图：用 vega_data.world_110m 的 TopoJSON 渲染国家轮廓
    #    mark_geoshape 将每个 TopoJSON feature 渲染为多边形；
    #    未被任何文明控制的国家显示为浅灰色。
    world_topo = alt.topo_feature(vega_data.world_110m.url, "countries")

    base_map = (
        alt.Chart(world_topo)
        .mark_geoshape(fill="#d6e8f0", stroke="#aac4d8", strokeWidth=0.3)
        .project("naturalEarth1")
    )

    # 3. 领土着色层：transform_lookup 把 TopoJSON feature.id 与
    #    control_df 的 id 列关联，按 civilization 字段着色。
    #    lookup key 是 TopoJSON 里的 id 字段（ISO 3166-1 数值）。
    territory_map = (
        alt.Chart(world_topo)
        .mark_geoshape(stroke="white", strokeWidth=0.25)
        .transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                data=control_df,
                key="id",
                fields=["civilization"],
            ),
        )
        .encode(
            color=alt.condition(
                "datum.civilization !== null",
                alt.Color(
                    "civilization:N",
                    scale=alt.Scale(
                        domain=list(COLOR_MAP.keys()),
                        range=list(COLOR_MAP.values()),
                    ),
                    legend=alt.Legend(title="控制文明"),
                ),
                alt.value("#d6e8f0"),   # 未被控制 → 保持底图蓝灰色
            ),
            tooltip=[
                alt.Tooltip("civilization:N", title="控制文明"),
            ],
        )
        .project("naturalEarth1")
    )

    # 4. GDP 气泡层：叠加在地图上，显示各文明本土的经济规模
    gdp_bubbles = (
        alt.Chart(map_year_df)
        .mark_circle(opacity=0.75, stroke="white", strokeWidth=1.5)
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            size=alt.Size(
                "gdp:Q",
                scale=alt.Scale(range=[60, 2800]),
                legend=alt.Legend(title="GDP"),
            ),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("civilization:N",   title="文明"),
                alt.Tooltip("gdp:Q",            title="GDP",      format=".2f"),
                alt.Tooltip("gdp_per_capita:Q", title="人均 GDP", format=".3f"),
                alt.Tooltip("population:Q",     title="人口(M)",  format=".1f"),
                alt.Tooltip("territories:Q",    title="领土倍数", format=".2f"),
                alt.Tooltip("tech_composite:Q", title="技术综合", format=".2f"),
            ],
        )
        .project("naturalEarth1")
    )

    # 5. 文明名标签
    labels = (
        alt.Chart(map_year_df)
        .mark_text(fontSize=9, fontWeight="bold", color="#222", dy=-13)
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            text="civilization:N",
        )
        .project("naturalEarth1")
    )

    world_map = (
        alt.layer(base_map, territory_map, gdp_bubbles, labels)
        .properties(
            width="container",
            height=430,
            title=f"世界领土控制格局 — {map_year} AD",
        )
    )
    st.altair_chart(world_map, use_container_width=True)

    # ── 事件日志 ──
    if events_on and not event_df.empty:
        with st.expander("📋 历史随机事件记录（点击展开）"):
            st.dataframe(
                event_df[["year","era","event","target","description"]].sort_values("year"),
                use_container_width=True, height=280,
            )


# ═════════════════════════════════════════════
# Tab 2：经济轨迹
# ═════════════════════════════════════════════
with tab2:
    st.header("📈 经济发展轨迹详细分析")

    # ── 4 指标折线图（2×2 用 st.columns 实现，避免 hconcat 宽度失效）──
    st.subheader("四维度经济发展轨迹")
    st.caption("背景色带区分历史时期；鼠标悬停可查看具体数值；可拖拽缩放。")

    def make_panel(y_field, y_title, show_legend=False):
        """折线图 + 时代色带，width='container' 在独立列中能正确撑满。"""
        lines = (
            alt.Chart(df_flt)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("year:Q", title="年份"),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color(
                    "civilization:N",
                    scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                    legend=alt.Legend(title="文明") if show_legend else None,
                ),
                tooltip=["civilization:N", "year:Q", f"{y_field}:Q"],
            )
        )
        return (
            alt.layer(era_bands(), lines)
            .properties(width="container", height=280)
            .interactive()
        )

    # 每个 st.columns 列各占 50%，altair width="container" 才能正确填满列宽
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(make_panel("gdp", "GDP（相对单位）", show_legend=True),
                        use_container_width=True)
    with c2:
        st.altair_chart(make_panel("gdp_per_capita", "人均 GDP"),
                        use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(make_panel("population", "人口（百万）"),
                        use_container_width=True)
    with c4:
        st.altair_chart(make_panel("military_str", "军事实力"),
                        use_container_width=True)

    # ── 大分流放大图 ─────────────────────────────────────
    st.subheader('🔍 "大分流"放大：1700-1850 年人均 GDP')
    st.caption("工业革命（紫色背景）期间，西北欧与其他地区的人均 GDP 产生急剧分化。")

    late_df = df_flt[df_flt["year"] >= 1700]
    diverge_chart = line_with_bands(
        late_df, "gdp_per_capita", "人均 GDP",
        color_map=COLOR_MAP, height=320,
        title="大分流：人均 GDP 分化（1700-1850）",
    )
    st.altair_chart(diverge_chart, use_container_width=True)

    # ── 1850年横向对比条形图 ──────────────────────────────
    st.subheader("1850年终值横向对比")
    metric_opt = st.radio(
        "选择指标",
        ["gdp", "gdp_per_capita", "population", "tech_composite"],
        horizontal=True,
        format_func=lambda x: {
            "gdp":"GDP总量","gdp_per_capita":"人均GDP",
            "population":"人口","tech_composite":"技术水平",
        }[x],
    )
    final_all = df[df["year"] == df["year"].max()].sort_values(metric_opt, ascending=False)
    bar_chart = (
        alt.Chart(final_all)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("civilization:N",
                    sort=alt.EncodingSortField(field=metric_opt, order="descending"),
                    title="文明"),
            y=alt.Y(f"{metric_opt}:Q", title=metric_opt),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=None,
            ),
            tooltip=["civilization:N", f"{metric_opt}:Q", "strategy:N"],
        )
        .properties(width="container", height=320, title=f"1850年 {metric_opt} 排名")
    )
    st.altair_chart(bar_chart, use_container_width=True)


# ═════════════════════════════════════════════
# Tab 3：技术竞赛
# ═════════════════════════════════════════════
with tab3:
    st.header("🔬 技术竞赛与研发路径")

    TECH_FIELDS = {
        "tech_agri": "🌾 农业技术",
        "tech_nav":  "⛵ 航海技术",
        "tech_mil":  "⚔️ 军事技术",
        "tech_ind":  "🏭 工业技术",
        "tech_com":  "💰 商业技术",
    }

    # ── 技术领域选择折线图 ────────────────────────────────
    tech_sel = st.selectbox("选择技术领域", list(TECH_FIELDS.keys()),
                            format_func=lambda x: TECH_FIELDS[x])

    tech_chart = line_with_bands(
        df_flt, tech_sel, f"{TECH_FIELDS[tech_sel]}（0-10）",
        color_map=COLOR_MAP, height=320,
        title=f"{TECH_FIELDS[tech_sel]} 发展轨迹",
    )
    st.altair_chart(tech_chart, use_container_width=True)

    # ── 五大技术领域全景对比（2+2+1 布局，每图独立列撑满宽度）──
    st.subheader("五大技术领域全景对比")
    st.caption("可以看出各文明的『技术专精』路径：葡西的航海、英国的工业、荷兰的商业……")

    def tech_line(field, label):
        lines = (
            alt.Chart(df_flt)
            .mark_line(strokeWidth=1.8)
            .encode(
                x=alt.X("year:Q", title="年份"),
                y=alt.Y(f"{field}:Q", title="水平(0-10)",
                        scale=alt.Scale(domain=[0, 10])),
                color=alt.Color(
                    "civilization:N",
                    scale=alt.Scale(domain=list(COLOR_MAP.keys()),
                                    range=list(COLOR_MAP.values())),
                    legend=None,
                ),
                tooltip=["civilization:N", "year:Q", f"{field}:Q"],
            )
        )
        return (
            alt.layer(era_bands(), lines)
            .properties(width="container", height=220, title=label)
            .interactive()
        )

    row_a1, row_a2 = st.columns(2)
    fields_list = list(TECH_FIELDS.items())
    with row_a1:
        st.altair_chart(tech_line(fields_list[0][0], fields_list[0][1]),
                        use_container_width=True)
    with row_a2:
        st.altair_chart(tech_line(fields_list[1][0], fields_list[1][1]),
                        use_container_width=True)

    row_b1, row_b2 = st.columns(2)
    with row_b1:
        st.altair_chart(tech_line(fields_list[2][0], fields_list[2][1]),
                        use_container_width=True)
    with row_b2:
        st.altair_chart(tech_line(fields_list[3][0], fields_list[3][1]),
                        use_container_width=True)

    # 工业技术单独全宽（最重要，值得更大展示）
    st.altair_chart(tech_line(fields_list[4][0], fields_list[4][1]),
                    use_container_width=True)

    # ── 技术结构雷达图（极坐标面积图）────────────────────
    st.subheader("技术结构雷达图对比")
    st.caption("展示 1000 年和 1850 年各文明的技术专精方向。"
               "面积越大 = 综合技术越强；形状不对称 = 专精某一领域。")

    radar_civs = st.multiselect(
        "选择对比文明（建议 2-4 个）",
        options=df["civilization"].unique().tolist(),
        default=(show_civs or ALL_CIVS)[:3],
    )

    if radar_civs:
        domains_cn = {"tech_agri":"农业","tech_nav":"航海","tech_mil":"军事",
                      "tech_ind":"工业","tech_com":"商业"}

        def make_radar_df(year_val):
            rows = []
            dy = df[df["year"] == year_val]
            for civ in radar_civs:
                row = dy[dy["civilization"] == civ]
                if row.empty: continue
                for field, cn in domains_cn.items():
                    rows.append({
                        "civilization": civ,
                        "domain": cn,
                        "value": row[field].values[0],
                        "year": year_val,
                    })
            return pd.DataFrame(rows)

        col_r1, col_r2 = st.columns(2)
        for col_w, yr in [(col_r1, df["year"].min()), (col_r2, df["year"].max())]:
            rdf = make_radar_df(yr)
            # Altair 极坐标：theta = 技术领域，radius = 技术值（0-10）
            # width/height 给固定值而非 "container"，极坐标布局不支持弹性宽度
            radar = (
                alt.Chart(rdf)
                .mark_line(point=True, strokeWidth=2.2, filled=False)
                .encode(
                    theta=alt.Theta("domain:N", sort=list(domains_cn.values())),
                    radius=alt.Radius("value:Q", scale=alt.Scale(domain=[0, 10])),
                    color=alt.Color(
                        "civilization:N",
                        scale=alt.Scale(domain=list(COLOR_MAP.keys()),
                                        range=list(COLOR_MAP.values())),
                        legend=alt.Legend(title="文明"),
                    ),
                    tooltip=["civilization:N", "domain:N", "value:Q"],
                )
            )
            radar_fill = radar.mark_arc(opacity=0.12, innerRadius=0)
            with col_w:
                st.altair_chart(
                    alt.layer(radar_fill, radar)
                    .properties(width=340, height=340, title=f"{yr} 年技术结构雷达图"),
                    use_container_width=True,
                )

    # ── 工业技术领先者时间线 ──────────────────────────────
    st.subheader("工业技术领先者历史时间线")
    leader_rows = []
    for y in sorted(df["year"].unique()):
        dy = df[df["year"] == y]
        leader = dy.loc[dy["tech_ind"].idxmax(), "civilization"]
        leader_rows.append({"year": y, "leader": leader})
    leader_df = pd.DataFrame(leader_rows)

    leader_chart = (
        alt.Chart(leader_df)
        .mark_point(size=80, filled=True)
        .encode(
            x=alt.X("year:Q", title="年份"),
            y=alt.Y("leader:N", title="领先文明",
                    sort=alt.EncodingSortField(field="year")),
            color=alt.Color(
                "leader:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=None,
            ),
            tooltip=["year:Q","leader:N"],
        )
        .properties(width="container", height=220,
                    title="工业技术历史领先者（哪个文明最先突破？）")
    )
    st.altair_chart(leader_chart, use_container_width=True)


# ═════════════════════════════════════════════
# Tab 4：贸易与殖民
# ═════════════════════════════════════════════
with tab4:
    st.header("🚢 贸易网络与殖民扩张")

    # 上行：贸易收益 + 殖民收益
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.altair_chart(
            line_with_bands(df_flt, "trade_income", "贸易收益", color_map=COLOR_MAP,
                            height=360, title="贸易收益演变"),
            use_container_width=True,
        )
    with col_t2:
        st.altair_chart(
            line_with_bands(df_flt, "colonial_income", "殖民地收益", color_map=COLOR_MAP,
                            height=360, title="殖民地收益演变"),
            use_container_width=True,
        )

    # 贸易开放度
    st.altair_chart(
        line_with_bands(df_flt, "trade_openness", "贸易开放度（0-1）",
                        color_map=COLOR_MAP, height=360,
                        title="贸易开放度演变（0=闭关锁国，1=完全自由贸易）"),
        use_container_width=True,
    )

    # 领土扩张（堆叠面积图）
    st.subheader("殖民版图扩张")
    terr_chart = (
        alt.Chart(df_flt)
        .mark_area(opacity=0.7)
        .encode(
            x=alt.X("year:Q", title="年份"),
            y=alt.Y("territories:Q", title="领土（1=本土）", stack=None),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="文明"),
            ),
            tooltip=["civilization:N","year:Q","territories:Q"],
        )
        .properties(width="container", height=360, title="各文明控制领土扩张")
        .interactive()
    )
    st.altair_chart(alt.layer(era_bands(), terr_chart), use_container_width=True)

    # ── 贸易 vs GDP 散点图（1850年截面）──────────────────
    st.subheader("1850年：贸易收益 vs GDP（贸易对经济规模的贡献）")
    scatter_trade = (
        alt.Chart(final)
        .mark_circle(size=160, opacity=0.85)
        .encode(
            x=alt.X("trade_income:Q", title="贸易收益"),
            y=alt.Y("gdp:Q",          title="GDP"),
            color=alt.Color(
                "civilization:N",
                scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                legend=alt.Legend(title="文明"),
            ),
            size=alt.Size("population:Q", scale=alt.Scale(range=[100,1200]),
                          legend=alt.Legend(title="人口")),
            tooltip=["civilization:N","gdp:Q","trade_income:Q","population:Q","strategy:N"],
        )
        .properties(width="container", height=400,
                    title="贸易收益与 GDP 总量的关系（1850年）")
    )
    st.altair_chart(scatter_trade, use_container_width=True)


# ═════════════════════════════════════════════
# Tab 5：反事实分析
# ═════════════════════════════════════════════
with tab5:
    st.header("🔀 反事实历史实验")
    st.markdown("""
    **核心问题**：大分流是历史的必然还是偶然？

    通过修改关键参数（地理条件、策略选择），推演替代历史路径，
    评估地理禀赋与决策风格各自的贡献。以下实验均关闭随机事件，
    排除随机扰动，使轨迹差异纯粹来自参数变化。
    """)

    cf_results = run_counterfactuals(seed)

    # 选择分析文明
    cf_civ = st.selectbox("选择要分析的文明",
                          ["中华帝国", "西北欧（荷英）", "伊比利亚（葡西）"])

    # 合并各场景为单 DataFrame
    cf_rows = []
    for scenario, sdf in cf_results.items():
        d = sdf[sdf["civilization"] == cf_civ].copy()
        d["scenario"] = scenario
        cf_rows.append(d)
    cf_df = pd.concat(cf_rows, ignore_index=True)

    # Altair 折线图对比各场景
    cf_chart = (
        alt.Chart(cf_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("year:Q", title="年份"),
            y=alt.Y("gdp:Q",  title="GDP"),
            color=alt.Color("scenario:N", legend=alt.Legend(title="场景")),
            strokeDash=alt.StrokeDash("scenario:N",
                legend=alt.Legend(title="场景"),
                scale=alt.Scale(
                    domain=list(cf_results.keys()),
                    range=[[1,0],[6,2],[4,2,1,2],[2,2],[8,2]],
                )),
            tooltip=["scenario:N","year:Q","gdp:Q"],
        )
        .properties(width="container", height=380,
                    title=f"{cf_civ} — 反事实路径对比")
        .interactive()
    )
    st.altair_chart(alt.layer(era_bands(), cf_chart), use_container_width=True)

    # 终值对比表
    st.subheader("1850年终值对比（各反事实场景）")
    final_year = list(cf_results.values())[0]["year"].max()
    cf_table_rows = []
    for sname, sdf in cf_results.items():
        d = sdf[(sdf["civilization"] == cf_civ) & (sdf["year"] == final_year)]
        if not d.empty:
            cf_table_rows.append({
                "场景": sname,
                "GDP": round(d["gdp"].values[0], 2),
                "人均GDP": round(d["gdp_per_capita"].values[0], 4),
                "工业技术": round(d["tech_ind"].values[0], 2),
                "领土": round(d["territories"].values[0], 2),
            })
    cf_tbl = pd.DataFrame(cf_table_rows)
    base_gdp = cf_tbl[cf_tbl["场景"] == "历史基准"]["GDP"].values
    if len(base_gdp):
        cf_tbl["vs基准(%)"] = ((cf_tbl["GDP"] / base_gdp[0] - 1) * 100).round(1)
    st.dataframe(cf_tbl, use_container_width=True)

    # 全局 GDP 总量对比
    st.subheader("全球 GDP 总量对比（各反事实场景）")
    world_rows = []
    for sname, sdf in cf_results.items():
        wg = sdf.groupby("year")["gdp"].sum().reset_index()
        wg["scenario"] = sname
        world_rows.append(wg)
    world_df = pd.concat(world_rows, ignore_index=True)

    world_chart = (
        alt.Chart(world_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("year:Q", title="年份"),
            y=alt.Y("gdp:Q",  title="全球 GDP 总量"),
            color=alt.Color("scenario:N", legend=alt.Legend(title="场景")),
            tooltip=["scenario:N","year:Q","gdp:Q"],
        )
        .properties(width="container", height=320,
                    title="全球 GDP 总量（各反事实场景对比）")
        .interactive()
    )
    st.altair_chart(alt.layer(era_bands(), world_chart), use_container_width=True)

    st.info("""
    📌 **解读提示**：
    - **中国高海岸**：郑和下西洋若坚持下去，中国获得与葡萄牙相当的海岸优势，
      大发现时代的轨迹会有多大改变？
    - **中国工业先行**：如果清朝在 1700 年代主动推进工业化，而非等到被动冲击？
    - **西北欧无煤炭**：英国工业革命的地理必然性——煤炭是充分还是必要条件？
    """)


# ═════════════════════════════════════════════
# Tab 6：ML 策略分析
# ═════════════════════════════════════════════
with tab6:
    st.header("🤖 机器学习策略涌现分析")
    st.markdown("""
    使用 **Q-Learning 强化学习** 训练三个智能体，以不同目标函数优化：
    | 智能体 | 驱动文明 | 奖励函数 | 预期涌现策略 |
    |--------|----------|---------|------------|
    | RL_GDP   | 西北欧（荷英）  | GDP 增长率最大化 | 工业先行（类英国）|
    | RL_Power | 伊比利亚（葡西）| 综合国力最大化   | 航海扩张（类葡西）|
    | RL_Trade | 奥斯曼帝国      | 贸易收益最大化   | 商业开放（类荷兰）|

    **核心假设**：历史上国家的决策行为，是对某种隐含目标的优化结果。
    如果 RL 智能体学到的策略与历史原型相似，则支持这一假说。
    """)

    if st.button("🚀 开始训练 RL 智能体", type="primary"):
        with st.spinner(f"训练 {rl_eps} 轮，约需 15-30 秒..."):
            rl_df, curves, strat_summary = run_with_rl(seed, rl_eps)
        st.success("训练完成！")

        # ── 训练学习曲线 ───────────────────────────────
        st.subheader("Q-Learning 训练曲线")
        st.caption("纵轴为每轮模拟结束时（1850年）的 GDP；曲线上升说明智能体在学习有效策略。")

        curve_rows = []
        for civ_name, vals in curves.items():
            for ep, v in enumerate(vals):
                curve_rows.append({"episode": ep + 1, "gdp": v, "agent": civ_name})
        curve_df = pd.DataFrame(curve_rows)

        if not curve_df.empty:
            # 原始值（细线）+ 移动平均（粗线）
            curve_df["smooth"] = (
                curve_df.groupby("agent")["gdp"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean())
            )
            raw_line = (
                alt.Chart(curve_df)
                .mark_line(opacity=0.3, strokeWidth=1)
                .encode(
                    x=alt.X("episode:Q", title="训练轮次（Episode）"),
                    y=alt.Y("gdp:Q", title="1850年 GDP"),
                    color=alt.Color("agent:N", legend=alt.Legend(title="智能体")),
                )
            )
            smooth_line = (
                alt.Chart(curve_df)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x="episode:Q",
                    y=alt.Y("smooth:Q", title="1850年 GDP"),
                    color=alt.Color("agent:N", legend=None),
                    tooltip=["agent:N","episode:Q",
                             alt.Tooltip("smooth:Q", title="平滑GDP", format=".2f")],
                )
            )
            st.altair_chart(
                (raw_line + smooth_line)
                .properties(width="container", height=320,
                            title="Q-Learning 训练曲线（实线=5轮移动平均）"),
                use_container_width=True,
            )

        # ── 策略动作分布（水平条形图）────────────────────
        st.subheader("RL 策略涌现行为：偏好哪种决策模式？")
        st.caption("训练结束后，智能体在 Q-table 各状态上最偏好的动作分布。")

        if strat_summary:
            dist_rows = []
            for civ_name, info in strat_summary.items():
                for action, freq in info["action_distribution"].items():
                    dist_rows.append({
                        "agent":   civ_name,
                        "reward":  info["reward_type"],
                        "action":  action,
                        "freq":    freq,
                    })
            dist_df = pd.DataFrame(dist_rows)

            dist_chart = (
                alt.Chart(dist_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("action:N",  title="策略模式", sort="-x"),
                    x=alt.X("freq:Q",    title="偏好频率", axis=alt.Axis(format=".0%")),
                    color=alt.Color("agent:N", legend=alt.Legend(title="智能体")),
                    row=alt.Row("agent:N", title=""),
                    tooltip=["agent:N","reward:N","action:N",
                             alt.Tooltip("freq:Q", format=".1%")],
                )
                .properties(width="container", height=120,
                            title="RL 智能体策略偏好分布")
            )
            st.altair_chart(dist_chart, use_container_width=True)

            # 对照表
            st.subheader("RL 策略 vs. 历史原型：验证假说")
            expected_map = {
                "power": ("航海扩张", "葡萄牙/西班牙（MaritimeExpansionist）"),
                "gdp":   ("工业发展", "英国（IndustrialPioneer）"),
                "trade": ("商业贸易", "荷兰（TradeHub）"),
            }
            match_rows = []
            for civ_name, info in strat_summary.items():
                d = info["action_distribution"]
                top = max(d, key=d.get)
                rt  = info["reward_type"]
                exp_action, exp_hist = expected_map.get(rt, ("N/A","N/A"))
                match_rows.append({
                    "RL智能体": civ_name, "奖励函数": rt,
                    "实际最常选": top, "预期对应": exp_action,
                    "历史原型": exp_hist,
                    "Q-table大小": info["q_table_size"],
                    "假说吻合": "✅" if top == exp_action else "❓",
                })
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True)

        # ── RL vs 规则式 GDP 对比 ─────────────────────────
        st.subheader("RL 策略 vs 规则式策略：GDP 轨迹对比")
        rl_target_civs = ["西北欧（荷英）","伊比利亚（葡西）","奥斯曼帝国"]

        compare_rows = []
        for civ in rl_target_civs:
            d_rl   = rl_df[rl_df["civilization"] == civ].copy()
            d_rl["type"] = "RL 策略"
            d_rule = df[df["civilization"] == civ].copy()
            d_rule["type"] = "规则式策略"
            compare_rows.extend([d_rl, d_rule])
        cmp_df = pd.concat(compare_rows, ignore_index=True)

        cmp_chart = (
            alt.Chart(cmp_df)
            .mark_line(strokeWidth=2.2)
            .encode(
                x=alt.X("year:Q", title="年份"),
                y=alt.Y("gdp:Q",  title="GDP"),
                color=alt.Color("civilization:N",
                    scale=alt.Scale(domain=list(COLOR_MAP.keys()),
                                    range=list(COLOR_MAP.values())),
                    legend=alt.Legend(title="文明"),
                ),
                strokeDash=alt.StrokeDash(
                    "type:N",
                    scale=alt.Scale(domain=["RL 策略","规则式策略"], range=[[1,0],[6,3]]),
                    legend=alt.Legend(title="策略类型"),
                ),
                tooltip=["civilization:N","type:N","year:Q","gdp:Q"],
            )
            .properties(width="container", height=380,
                        title="RL 策略 vs 规则式策略：GDP 对比（实线=RL，虚线=规则式）")
            .interactive()
        )
        st.altair_chart(alt.layer(era_bands(), cmp_chart), use_container_width=True)

        st.info("""
        📌 **解读**：如果 RL 策略（实线）优于规则式策略（虚线），
        说明 ML 发现了人类历史决策者未能充分利用的"最优路径"；
        如果低于，则说明规则式策略已经相当接近最优，
        支持"历史并非随机漫步，而是某种隐性优化"的观点。
        """)
    else:
        st.info("点击上方按钮开始训练 RL 智能体（约需 15-30 秒）。")

# ── 底部说明 ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("""
**模型说明** | 课程：ECON0302G 世界经济导论 | 图表引擎：Vega-Altair

经济模型：Cobb-Douglas 生产函数（Y = A·L^α·K^β·T^γ） + 比较优势贸易 + Logistic 人口动态
ML 方法：表格型 Q-Learning（Bellman 方程更新），状态空间 4096 维，动作空间 5 维
历史校准参考：麦迪森世界经济历史统计数据库（Maddison Project Database 2020）
反事实实验：修改地理/策略参数，排除随机事件，推演替代历史路径
""")
