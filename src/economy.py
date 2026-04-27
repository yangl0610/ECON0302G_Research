"""
economy.py
----------
历史经济模型的核心计算逻辑。

理论基础：
  - 生产函数采用柯布-道格拉斯形式：Y = A * L^α * K^β * T^γ
    其中 A = 全要素生产率（受地理、资源、时代影响）
         L = 劳动力（人口代理）
         K = 资本存量
         T = 技术综合指数
  - 资本动态：ΔK = s·Y − δ·K  （储蓄积累减去折旧）
  - 人口动态：Logistic 增长（受农业技术和粮食资源限制）
  - 贸易收益：基于比较优势 + 贸易开放度 + 网络效应
  - 殖民收益：航海技术 × 军事实力 × 殖民地规模
  - 技术扩散：贸易越密切，越容易学习对方技术
"""

import numpy as np
from typing import List, Dict
from .civilization import Civilization, Era


# ─────────────────────────────────────────────
# 柯布-道格拉斯参数
# ─────────────────────────────────────────────
ALPHA = 0.35   # 劳动（人口）产出弹性
BETA  = 0.40   # 资本产出弹性
GAMMA = 0.25   # 技术产出弹性

DEPRECIATION  = 0.06   # 每回合资本折旧率（~0.4%/年）
BASE_SAVINGS   = 0.22   # 默认储蓄率（可被策略调整）


# ─────────────────────────────────────────────
# 全要素生产率（TFP）
# ─────────────────────────────────────────────
def compute_tfp(civ: Civilization, era: Era) -> float:
    """
    计算全要素生产率 A。
    TFP 综合了地理、资源和时代背景，是"先天条件"的体现。

    历史含义：同样的劳动和资本投入，
    葡萄牙的港口条件比内陆国家能产出更多贸易收益（coast_access 高）；
    英国拥有丰富煤炭，工业时代 TFP 大幅领先（coal 高）。
    """
    geo     = civ.geography
    res_val = civ.resources.era_value(era)  # 时代加权资源价值

    # 基础 TFP = 气候 × 0.3 + 地理贸易潜力 × 0.4 + 资源值 × 0.3
    base_tfp = (
        0.30 * geo.climate_score
      + 0.40 * geo.trade_potential()
      + 0.30 * min(res_val, 1.5) / 1.5
    )

    # 时代修正：工业时代煤炭比任何其他资源都重要
    era_bonus = 0.0
    if era == Era.DISCOVERY:
        era_bonus = 0.15 * geo.coast_access       # 大航海时代，海岸优势放大
    elif era == Era.INDUSTRIAL:
        era_bonus = 0.30 * (civ.resources.coal / 3.0)  # 煤炭国的时代红利

    return max(0.1, base_tfp + era_bonus)


# ─────────────────────────────────────────────
# GDP 生产函数
# ─────────────────────────────────────────────
def compute_gdp(civ: Civilization, era: Era) -> float:
    """
    柯布-道格拉斯生产函数：Y = A * L^α * K^β * T^γ

    注：这里用人口代理劳动力（忽略劳动参与率的历史变化），
    用技术综合指数代理"技术进步"对生产效率的乘数效应。
    """
    A = compute_tfp(civ, era)
    L = max(civ.population, 0.1)
    K = max(civ.capital, 0.1)
    T = max(civ.technology.composite(era), 0.1)

    return A * (L ** ALPHA) * (K ** BETA) * (T ** GAMMA)


# ─────────────────────────────────────────────
# 资本动态
# ─────────────────────────────────────────────
def update_capital(civ: Civilization, savings_rate: float) -> float:
    """
    资本积累方程：K(t+1) = K(t) + s·Y(t) - δ·K(t)
    储蓄率 s 由策略决定（激进扩张策略会降低储蓄，保守策略会提高）
    """
    investment = savings_rate * civ.gdp
    depreciation = DEPRECIATION * civ.capital
    return max(civ.capital + investment - depreciation, 0.1)


# ─────────────────────────────────────────────
# 人口动态（Logistic 增长）
# ─────────────────────────────────────────────
def update_population(civ: Civilization, era: Era) -> float:
    """
    Logistic 人口增长模型：
      P(t+1) = P(t) * [1 + r * (1 - P/K)]

    K（承载上限）由农业技术和粮食资源决定，
    体现"农业技术提升 → 养活更多人口 → 更大劳动力 → 更高 GDP"的历史逻辑。
    """
    # 基础增长率（每回合，非年化）
    base_rate = 0.04 if era == Era.MEDIEVAL else 0.06
    # 农业技术提升增长率（技术每提升 1 点，增长率 +0.3%）
    tech_bonus = 0.003 * civ.technology.agriculture
    r = base_rate + tech_bonus

    # 人口承载上限 = f(农业潜力, 粮食资源, 农业技术)
    carrying_capacity = (
        civ.geography.agri_potential() * 150.0          # 地理上限
      * (1.0 + 0.15 * civ.technology.agriculture)       # 技术扩大上限
      * (0.5 + 0.5 * civ.resources.food / 2.0)         # 资源调整
    )

    growth_factor = 1.0 + r * (1.0 - civ.population / max(carrying_capacity, 1.0))
    return max(civ.population * growth_factor, 0.1)


# ─────────────────────────────────────────────
# 贸易收益
# ─────────────────────────────────────────────
def compute_trade_income(civ: Civilization, others: List[Civilization], era: Era) -> float:
    """
    贸易收益 = Σ(与每个伙伴的双边贸易利得)

    核心机制：
      1. 比较优势：双方资源禀赋差异越大，贸易利得越大
         （葡萄牙有香料，英国有毛纺品，互补性强）
      2. 开放度加权：越开放的国家越能获取贸易收益
      3. 距离惩罚（用导航技术代理）：航海技术越高，距离成本越低
      4. 网络效应：贸易伙伴越多，信息和技术扩散越快

    历史背景：荷兰通过极高开放度（trade_openness~0.9）和
    商业技术在 17 世纪成为全球贸易中心，GDP 超出其人口规模
    """
    if era == Era.MEDIEVAL:
        # 中世纪贸易规模有限，主要是区域陆路贸易
        trade_scale = 0.05
    elif era == Era.DISCOVERY:
        trade_scale = 0.12   # 大航海时代，海上贸易爆发
    elif era == Era.MERCANTILE:
        trade_scale = 0.20   # 重商主义：贸易是国家核心战略
    else:
        trade_scale = 0.18   # 工业时代：更多靠生产，但贸易仍重要

    total_income = 0.0
    for other in others:
        if other.name == civ.name:
            continue

        # 比较优势：资源禀赋的互补程度
        res_a = np.array([civ.resources.food,   civ.resources.metal,
                          civ.resources.wood,   civ.resources.luxury])
        res_b = np.array([other.resources.food, other.resources.metal,
                          other.resources.wood, other.resources.luxury])
        # 用向量差的 L1 范数衡量禀赋差异（差异越大 → 互补性越强 → 贸易利得越大）
        complementarity = np.sum(np.abs(res_a - res_b)) / 8.0

        # 双边开放度（取双方均值）
        bilateral_openness = (civ.trade_openness + other.trade_openness) / 2.0

        # 导航技术降低贸易成本（取双方中较低值为瓶颈）
        nav_factor = min(civ.technology.navigation, other.technology.navigation) / 10.0
        nav_factor = 0.3 + 0.7 * nav_factor  # 保证基础贸易不为 0

        # 商业技术提升谈判和金融效率
        com_factor = (civ.technology.commerce + other.technology.commerce) / 20.0

        bilateral_trade = (
            trade_scale
          * complementarity
          * bilateral_openness
          * nav_factor
          * (1.0 + com_factor)
          * min(civ.gdp, other.gdp)  # 贸易量受制于较小一方的经济规模
        )
        total_income += bilateral_trade

    # 战略位置加成：位于贸易要道的国家额外收取"过路费"
    location_bonus = 1.0 + 0.4 * civ.geography.strategic_location
    return total_income * location_bonus * civ.trade_openness


# ─────────────────────────────────────────────
# 殖民收益
# ─────────────────────────────────────────────
def compute_colonial_income(civ: Civilization, era: Era) -> float:
    """
    殖民地带来的财富流入。
    历史背景：16-18 世纪，西班牙从美洲获得大量白银，
    葡萄牙控制香料贸易路线，英国在印度建立贸易站。

    殖民收益需要：高航海技术 + 足够军事实力 + 已有领土基础
    但殖民投资也有成本（维持军队和行政）
    """
    if era == Era.MEDIEVAL:
        # 中世纪还没有跨洋殖民，只有陆地扩张
        expansion_factor = 0.02
    elif era == Era.DISCOVERY:
        expansion_factor = 0.10   # 大发现时代：殖民收益最高
    elif era == Era.MERCANTILE:
        expansion_factor = 0.08   # 重商主义：殖民体系已成熟
    else:
        expansion_factor = 0.05   # 工业时代：工业收益超越殖民收益

    # 能够维持殖民地需要：航海技术 + 军事力量
    capability = (
        0.5 * (civ.technology.navigation / 10.0)
      + 0.5 * (civ.military_str / 10.0)
    )

    # 超出本土的领土每单位产生收益
    extra_territory = max(civ.territories - 1.0, 0.0)

    return expansion_factor * capability * extra_territory * civ.gdp


# ─────────────────────────────────────────────
# 技术更新
# ─────────────────────────────────────────────
def update_technology(
    civ: Civilization,
    others: List[Civilization],
    era: Era,
    tech_focus: str,          # 本回合重点投资的技术领域
    investment_share: float,  # 分配给技术投资的 GDP 比例
) -> None:
    """
    技术进步来自两个来源：
      1. 主动投资（内生增长）：将 GDP 的一部分投入技术研发
      2. 技术扩散（外溢学习）：从贸易伙伴处学习，开放度越高扩散越快

    历史含义：
      - 英国工业革命部分来自荷兰金融技术的扩散（1688 年光荣革命后）
      - 日本明治维新是技术扩散的极端案例（主动引进西方技术）
      - 中国在中世纪向伊斯兰世界传播了造纸、印刷等技术
    """
    t = civ.technology

    # ── 投资回报：重点领域获得集中投资，其他领域维持基础增长 ──
    # 不同技术在不同时代投资效率不同（工业时代投资工业技术效果更好）
    base_growth   = 0.015  # 每回合基础技术增长（维持性投资）
    focused_bonus = 0.12   # 重点投资带来的额外增长
    era_multiplier = {
        Era.MEDIEVAL:   {"agriculture": 1.5, "navigation": 0.8, "military": 1.2, "industry": 0.2, "commerce": 1.0},
        Era.DISCOVERY:  {"agriculture": 0.8, "navigation": 2.0, "military": 1.0, "industry": 0.3, "commerce": 1.2},
        Era.MERCANTILE: {"agriculture": 0.7, "navigation": 1.2, "military": 0.9, "industry": 0.8, "commerce": 2.0},
        Era.INDUSTRIAL: {"agriculture": 0.6, "navigation": 0.8, "military": 0.8, "industry": 2.5, "commerce": 1.0},
    }[era]

    for domain in ["agriculture", "navigation", "military", "industry", "commerce"]:
        current = getattr(t, domain)
        if current >= 10.0:
            continue  # 技术上限

        is_focus = (domain == tech_focus)
        growth = (base_growth + (focused_bonus if is_focus else 0.0)) * era_multiplier[domain]

        # 技术投资占 GDP 的比例影响增长速度
        growth *= (0.5 + 1.5 * investment_share)

        # 技术越高，边际增长越难（边际报酬递减）
        diminishing = 1.0 / (1.0 + 0.15 * current)

        new_val = min(current + growth * diminishing, 10.0)
        setattr(t, domain, new_val)

    # ── 技术扩散：从贸易伙伴处学习 ──────────────────────────────
    # 前提：双方都开放（trade_openness 高），且贸易量较大
    diffusion_rate = 0.008 * civ.trade_openness
    if diffusion_rate < 0.001:
        return  # 闭关锁国，无扩散

    for other in others:
        if other.name == civ.name:
            continue
        bilateral_openness = civ.trade_openness * other.trade_openness
        if bilateral_openness < 0.05:
            continue

        for domain in ["agriculture", "navigation", "military", "industry", "commerce"]:
            my_level    = getattr(t, domain)
            their_level = getattr(other.technology, domain)
            # 只有对方技术领先才能学到东西（向后扩散无意义）
            if their_level > my_level:
                gap = their_level - my_level
                learned = diffusion_rate * bilateral_openness * gap * 0.3
                setattr(t, domain, min(my_level + learned, 10.0))


# ─────────────────────────────────────────────
# 领土/殖民扩张
# ─────────────────────────────────────────────
def update_territories(civ: Civilization, era: Era, expansion_level: int) -> float:
    """
    扩张逻辑：
      expansion_level = 0：保守，只防守本土
      expansion_level = 1：温和扩张
      expansion_level = 2：激进殖民

    扩张成功率取决于：航海技术 × 军事实力 × 地理条件
    大发现时代成功率更高（时代窗口）
    """
    if expansion_level == 0:
        # 领土自然有轻微侵蚀（维护成本）
        return max(civ.territories * 0.998, 1.0)

    nav_cap = civ.technology.navigation / 10.0
    mil_cap = civ.military_str / 10.0
    geo_cap = civ.geography.coast_access

    era_bonus = {
        Era.MEDIEVAL:   0.5,
        Era.DISCOVERY:  1.5,  # 大发现时代扩张窗口期
        Era.MERCANTILE: 1.0,
        Era.INDUSTRIAL: 0.7,
    }[era]

    success_prob = nav_cap * 0.4 + mil_cap * 0.4 + geo_cap * 0.2
    gain_rate = 0.05 if expansion_level == 1 else 0.12

    territory_gain = gain_rate * success_prob * era_bonus
    return civ.territories + territory_gain


# ─────────────────────────────────────────────
# 一回合完整经济更新
# ─────────────────────────────────────────────
def apply_turn(
    civ: Civilization,
    all_civs: List[Civilization],
    era: Era,
    decision: Dict,
    noise_std: float = 0.02,  # 随机扰动（模拟历史偶然性）
    rng: np.random.Generator = None,
) -> None:
    """
    一回合的完整经济演算，按顺序：
      1. 更新领土（扩张决策）
      2. 更新技术（投资 + 扩散）
      3. 更新资本（储蓄 - 折旧）
      4. 更新人口（Logistic 增长）
      5. 计算贸易和殖民收益
      6. 计算 GDP（加上贸易和殖民溢价）
      7. 更新开放度和军事实力

    decision 字典包含：
      tech_focus       : 重点技术领域（str）
      expansion_level  : 0/1/2
      trade_policy     : "open" / "balanced" / "closed"
      savings_rate     : float (0.1~0.4)
      tech_investment  : float (GDP 中用于技术的比例)
    """
    if rng is None:
        rng = np.random.default_rng()

    tech_focus      = decision.get("tech_focus", "commerce")
    expansion_level = decision.get("expansion_level", 1)
    trade_policy    = decision.get("trade_policy", "balanced")
    savings_rate    = decision.get("savings_rate", BASE_SAVINGS)
    tech_inv_share  = decision.get("tech_investment", 0.08)

    # Trade-policy cost: open trade crowds out domestic capital formation;
    # closed policy protects it at the cost of lower trade income and diffusion.
    # Multipliers are applied here so they hold regardless of strategy type.
    SAVINGS_MODIFIER = {"open": 0.88, "balanced": 1.00, "closed": 1.10}
    savings_rate *= SAVINGS_MODIFIER[trade_policy]

    # open also amplifies external price shocks (imported volatility)
    SHOCK_MODIFIER = {"open": 1.40, "balanced": 1.00, "closed": 0.75}

    # 1. 领土扩张
    civ.territories = update_territories(civ, era, expansion_level)

    # 2. 技术进步（内生 + 扩散）
    update_technology(civ, all_civs, era, tech_focus, tech_inv_share)

    # 3. 资本积累
    civ.capital = update_capital(civ, savings_rate)

    # 4. 人口增长
    civ.population = update_population(civ, era)

    # 5. 计算贸易和殖民收益
    civ.trade_income    = compute_trade_income(civ, all_civs, era)
    civ.colonial_income = compute_colonial_income(civ, era)

    # 6. 计算 GDP，再叠加贸易和殖民收益
    base_gdp = compute_gdp(civ, era)
    civ.gdp  = base_gdp + civ.trade_income + civ.colonial_income

    # 随机扰动（开放经济对外部冲击更敏感）
    effective_noise = noise_std * SHOCK_MODIFIER[trade_policy]
    shock = 1.0 + rng.normal(0, effective_noise)
    civ.gdp = max(civ.gdp * shock, 0.01)

    # 7. 更新贸易开放度（向目标值缓慢调整，不能突变）
    target_openness = {"open": 0.85, "balanced": 0.50, "closed": 0.15}[trade_policy]
    civ.trade_openness += 0.15 * (target_openness - civ.trade_openness)

    # 军事实力 = 军事技术 × 经济规模（相对化）
    civ.military_str = civ.technology.military * (1.0 + 0.1 * (civ.gdp / 5.0))
