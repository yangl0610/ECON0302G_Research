"""
civilization.py
---------------
文明（国家）的核心数据结构。
每个 Civilization 对象代表一个历史上的国家/地区，
包含其地理禀赋、资源、技术水平和经济状态。
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


# ─────────────────────────────────────────────
# 历史时期划分
# ─────────────────────────────────────────────
class Era(Enum):
    MEDIEVAL   = 0   # 中世纪      1000-1400
    DISCOVERY  = 1   # 地理大发现  1400-1600
    MERCANTILE = 2   # 重商主义    1600-1750
    INDUSTRIAL = 3   # 工业革命    1750-1850

ERA_LABELS = {
    Era.MEDIEVAL:   "中世纪 (1000-1400)",
    Era.DISCOVERY:  "地理大发现 (1400-1600)",
    Era.MERCANTILE: "重商主义 (1600-1750)",
    Era.INDUSTRIAL: "工业革命 (1750-1850)",
}

# 每个时期的起止年份，用于时间轴标注
ERA_YEARS = {
    Era.MEDIEVAL:   (1000, 1400),
    Era.DISCOVERY:  (1400, 1600),
    Era.MERCANTILE: (1600, 1750),
    Era.INDUSTRIAL: (1750, 1850),
}

# 每个时期的回合数（每回合 = 约 10-20 年）
TURNS_PER_ERA = {
    Era.MEDIEVAL:   20,
    Era.DISCOVERY:  10,
    Era.MERCANTILE:  8,
    Era.INDUSTRIAL:  7,
}


# ─────────────────────────────────────────────
# 地理禀赋
# ─────────────────────────────────────────────
@dataclass
class Geography:
    """
    文明的地理条件，直接影响贸易潜力和农业产出。
    所有字段取值范围 0.0 ~ 1.0。

    coast_access       : 海岸线/港口条件，决定航海和贸易能力
    terrain_quality    : 耕地质量，决定农业上限
    climate_score      : 气候适宜度，影响综合生产率
    river_density      : 内河密度，降低内陆运输成本
    strategic_location : 是否位于历史贸易要道（如香料之路、地中海）
    """
    coast_access:        float
    terrain_quality:     float
    climate_score:       float
    river_density:       float
    strategic_location:  float

    def trade_potential(self) -> float:
        """决定该文明在贸易网络中的天然优势"""
        return 0.45 * self.coast_access + 0.35 * self.strategic_location + 0.20 * self.river_density

    def agri_potential(self) -> float:
        """决定粮食产量上限，从而限制人口容量"""
        return 0.60 * self.terrain_quality + 0.40 * self.climate_score


# ─────────────────────────────────────────────
# 资源禀赋
# ─────────────────────────────────────────────
@dataclass
class Resources:
    """
    自然资源初始禀赋（相对单位，0.0 ~ 3.0）。
    这些值在整个模拟中基本固定（代表地质条件），
    但会被技术进步放大（开采效率提升）。

    food   : 粮食产能（农业土地、水源）
    metal  : 金属矿藏（铁、铜——工具、武器原料）
    wood   : 林木资源（造船、建筑）
    luxury : 奢侈品（香料、丝绸、贵金属——贸易利润来源）
    coal   : 煤炭储量（工业革命后才变得关键）
    """
    food:   float
    metal:  float
    wood:   float
    luxury: float
    coal:   float

    def era_value(self, era: Era) -> float:
        """
        同样的资源，在不同历史时期价值不同。
        例如煤炭在工业革命前几乎没用，之后极为关键。
        """
        weights = {
            Era.MEDIEVAL:   [0.45, 0.20, 0.20, 0.15, 0.00],
            Era.DISCOVERY:  [0.25, 0.20, 0.25, 0.30, 0.00],
            Era.MERCANTILE: [0.20, 0.15, 0.20, 0.40, 0.05],
            Era.INDUSTRIAL: [0.10, 0.20, 0.15, 0.10, 0.45],
        }
        w = weights[era]
        vals = [self.food, self.metal, self.wood, self.luxury, self.coal]
        return sum(w[i] * vals[i] for i in range(5))


# ─────────────────────────────────────────────
# 技术水平
# ─────────────────────────────────────────────
@dataclass
class TechLevel:
    """
    五个技术领域的水平，0.0 ~ 10.0 分制。
    技术可以通过投资提升，也会从贸易伙伴处扩散学习。

    agriculture : 农业技术（提高粮食产量和人口容量）
    navigation  : 航海技术（决定探索/殖民能力，贸易成本）
    military    : 军事技术（影响领土扩张和防御能力）
    industry    : 工业技术（1750年后成为核心生产力）
    commerce    : 商业技术（提高贸易效率和金融能力）
    """
    agriculture: float = 1.0
    navigation:  float = 1.0
    military:    float = 1.0
    industry:    float = 0.3   # 工业初始很低
    commerce:    float = 1.0

    def composite(self, era: Era) -> float:
        """
        综合技术指数：按历史时期对各领域加权求和。
        反映"什么时代，哪种技术最重要"的历史规律。
        """
        weights = {
            Era.MEDIEVAL:   [0.40, 0.05, 0.25, 0.00, 0.30],
            Era.DISCOVERY:  [0.15, 0.35, 0.20, 0.00, 0.30],
            Era.MERCANTILE: [0.10, 0.20, 0.15, 0.15, 0.40],
            Era.INDUSTRIAL: [0.05, 0.10, 0.10, 0.55, 0.20],
        }
        w = weights[era]
        vals = [self.agriculture, self.navigation, self.military,
                self.industry, self.commerce]
        return sum(w[i] * vals[i] for i in range(5))

    def as_dict(self) -> Dict[str, float]:
        return {
            "agriculture": self.agriculture,
            "navigation":  self.navigation,
            "military":    self.military,
            "industry":    self.industry,
            "commerce":    self.commerce,
        }

    def as_array(self) -> np.ndarray:
        return np.array([self.agriculture, self.navigation, self.military,
                         self.industry, self.commerce], dtype=np.float32)


# ─────────────────────────────────────────────
# 文明主体
# ─────────────────────────────────────────────
class Civilization:
    """
    模拟中的一个国家/文明智能体。
    每回合由策略代理（Strategy）决定行动，
    由经济模型（economy.py）更新状态。
    """

    def __init__(
        self,
        name: str,
        geography: Geography,
        resources: Resources,
        strategy_name: str,
        color: str,
        lat: float = 0.0,   # 地图纬度，用于可视化
        lon: float = 0.0,   # 地图经度
    ):
        self.name          = name
        self.geography     = geography
        self.resources     = resources
        self.strategy_name = strategy_name
        self.color         = color
        self.lat           = lat
        self.lon           = lon

        # ── 经济状态 ──────────────────────────────
        # 人口（百万人），根据地理和粮食资源初始化
        self.population   = self._init_population()
        # GDP（相对单位，1000 AD 全球约为 100）
        self.gdp          = self._init_gdp()
        # 物质资本存量（约为 GDP 的 2.5 倍，反映历史初始积累）
        self.capital      = self.gdp * 2.5

        self.technology   = TechLevel()

        # ── 政治/战略状态 ─────────────────────────
        self.territories    = 1.0   # 控制领土相对大小（1=本土，>1含殖民地）
        self.trade_openness = 0.40  # 贸易开放度：0=闭关锁国，1=完全自由贸易
        self.military_str   = 1.0   # 军事实力（影响扩张和防御）

        # ── 当期收入分项（每回合更新）────────────
        self.trade_income    = 0.0
        self.colonial_income = 0.0

        # ── 历史记录（用于绘图）─────────────────
        self.history: Dict[str, List] = {
            "year":            [],
            "era":             [],
            "gdp":             [],
            "gdp_per_capita":  [],
            "population":      [],
            "tech_composite":  [],
            "trade_income":    [],
            "colonial_income": [],
            "territories":     [],
            "trade_openness":  [],
            "military_str":    [],
            # 各技术领域详细轨迹
            "tech_agri":  [],
            "tech_nav":   [],
            "tech_mil":   [],
            "tech_ind":   [],
            "tech_com":       [],
            # 每回合决策快照
            "decision_tech":   [],   # 重点技术领域
            "decision_expand": [],   # 扩张等级 0/1/2
            "decision_trade":  [],   # 贸易政策
        }

    # ─── 初始化辅助 ──────────────────────────────────

    def _init_population(self) -> float:
        """1000 AD 人口估算（百万人）。基准值 6M，按地理条件和粮食资源缩放。"""
        base = 6.0
        geo_factor  = 0.5 + 0.5 * self.geography.agri_potential()
        res_factor  = 0.5 + 0.5 * min(self.resources.food / 2.0, 1.0)
        return base * geo_factor * res_factor

    def _init_gdp(self) -> float:
        """GDP 初始值（相对单位），人均 GDP 约 0.10 单位。"""
        return self.population * 0.10

    # ─── 属性 ────────────────────────────────────────

    @property
    def gdp_per_capita(self) -> float:
        return self.gdp / max(self.population, 0.01)

    # ─── 状态记录 ─────────────────────────────────────

    def record(self, year: int, era: Era, decision: dict = None) -> None:
        """将当前状态快照追加到历史记录，每回合调用一次。decision 为本回合策略决策。"""
        h = self.history
        t = self.technology
        h["year"].append(year)
        h["era"].append(era.name)
        h["gdp"].append(round(self.gdp, 3))
        h["gdp_per_capita"].append(round(self.gdp_per_capita, 4))
        h["population"].append(round(self.population, 2))
        h["tech_composite"].append(round(t.composite(era), 3))
        h["trade_income"].append(round(self.trade_income, 3))
        h["colonial_income"].append(round(self.colonial_income, 3))
        h["territories"].append(round(self.territories, 2))
        h["trade_openness"].append(round(self.trade_openness, 2))
        h["military_str"].append(round(self.military_str, 3))
        h["tech_agri"].append(round(t.agriculture, 2))
        h["tech_nav"].append(round(t.navigation, 2))
        h["tech_mil"].append(round(t.military, 2))
        h["tech_ind"].append(round(t.industry, 2))
        h["tech_com"].append(round(t.commerce, 2))
        dec = decision or {}
        h["decision_tech"].append(dec.get("tech_focus", "agriculture"))
        h["decision_expand"].append(dec.get("expansion_level", 0))
        h["decision_trade"].append(dec.get("trade_policy", "balanced"))

    # ─── 供 RL 智能体使用的状态向量 ────────────────────

    def state_vector(self, world_avg_gdp: float, era: Era) -> np.ndarray:
        """
        将当前状态压缩为归一化向量，供 Q-learning 智能体观测。
        12 维特征：5 项技术 + GDP相对值 + 开放度 + 领土 + 4 项地理
        """
        t = self.technology
        return np.array([
            t.agriculture / 10.0,
            t.navigation  / 10.0,
            t.military    / 10.0,
            t.industry    / 10.0,
            t.commerce    / 10.0,
            self.gdp / (world_avg_gdp + 1e-6),  # 相对GDP排名代理
            self.trade_openness,
            min(self.territories / 5.0, 1.0),
            self.geography.coast_access,
            self.geography.terrain_quality,
            self.resources.coal / 3.0,
            float(era.value) / 3.0,
        ], dtype=np.float32)
