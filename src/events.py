"""
events.py
---------
历史随机事件系统。

核心设计思想：
  历史的演进不是纯粹决定论的，存在大量"偶然事件"
  （黑死病、技术意外发明、自然灾害、政治剧变）。
  这些事件对 GDP、人口、技术和贸易造成冲击，
  是评估"历史偶然性"影响的关键实验变量。

实验设计：
  - 开启事件（events_enabled=True）：观察偶然性的实际影响
  - 关闭事件（events_enabled=False）：纯粹的决定论基准线
  两者的轨迹差异 = 历史偶然性的量化贡献

事件按作用范围分为：
  - 局部事件（Local）：只影响目标文明
  - 区域事件（Regional）：影响一个地理区域
  - 全局事件（Global）：影响所有文明
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
import numpy as np
from .civilization import Civilization, Era


# ─────────────────────────────────────────────
# 事件数据结构
# ─────────────────────────────────────────────
@dataclass
class HistoricalEvent:
    """
    一个历史事件的完整描述。

    name         : 事件名（用于可视化标注）
    year_range   : (start, end) 年份范围，事件只在此期间可能触发
    probability  : 每回合触发概率（0~1）
    scope        : "local" / "regional" / "global"
    target_filter: 可选过滤器，决定哪些文明受影响（None = 所有）
    effects      : 应用函数 f(civ, rng) → 修改 civ 状态
    description  : 事件历史背景说明
    """
    name:          str
    year_range:    tuple          # (min_year, max_year)
    probability:   float          # 每回合触发概率
    scope:         str            # "local" | "regional" | "global"
    effects:       Callable       # f(civ: Civilization, rng: np.random.Generator)
    description:   str = ""
    target_filter: Optional[Callable] = None  # f(civ) -> bool


# ─────────────────────────────────────────────
# 效果函数（直接修改文明状态）
# ─────────────────────────────────────────────
def _black_death(civ: Civilization, rng: np.random.Generator):
    """
    黑死病（1347-1353）：欧洲死亡约 1/3 人口。
    短期：人口暴跌，GDP 下降，资本受损。
    长期（模型未显式处理）：幸存者人均 GDP 反而上升（劳动力稀缺）。
    """
    death_rate = rng.uniform(0.25, 0.45)
    civ.population *= (1 - death_rate)
    civ.gdp        *= (1 - death_rate * 0.7)   # GDP 下降幅度略小（部分资本留存）
    civ.capital    *= (1 - death_rate * 0.3)


def _technological_breakthrough(civ: Civilization, rng: np.random.Generator):
    """
    技术突破（印刷机、蒸汽机等）：某一技术领域意外跃升。
    历史背景：谷登堡印刷机（1440）使知识扩散速度提升 10 倍，
    瓦特蒸汽机（1769）让工业技术突破临界点。
    """
    domain = rng.choice(["navigation", "industry", "commerce", "agriculture"])
    boost  = rng.uniform(0.5, 1.5)
    current = getattr(civ.technology, domain)
    setattr(civ.technology, domain, min(current + boost, 10.0))


def _trade_route_disruption(civ: Civilization, rng: np.random.Generator):
    """
    贸易路线中断（奥斯曼封锁地中海、海盗猖獗）。
    历史背景：1453 年君士坦丁堡陷落，东西贸易路线受阻，
    这直接推动了葡萄牙探索绕非洲到达印度的新航路。
    """
    disruption = rng.uniform(0.3, 0.6)
    civ.trade_income    *= (1 - disruption)
    civ.trade_openness  *= (1 - disruption * 0.3)


def _natural_disaster(civ: Civilization, rng: np.random.Generator):
    """
    自然灾害（旱灾、洪水、火山爆发）。
    主要影响农业产出和人口，短期重创经济。
    """
    severity = rng.uniform(0.05, 0.25)
    civ.population *= (1 - severity * 0.5)
    civ.gdp        *= (1 - severity)
    civ.resources.food *= (1 - severity * 0.3)   # 长期降低土地产出


def _resource_discovery(civ: Civilization, rng: np.random.Generator):
    """
    新资源发现（美洲白银矿、北海渔场、煤矿勘探）。
    历史背景：1545 年波托西银矿发现，使西班牙财政收入翻倍，
    但也导致欧洲通货膨胀（"价格革命"）。
    """
    resource = rng.choice(["metal", "luxury", "coal", "food"])
    boost = rng.uniform(0.2, 0.8)
    current = getattr(civ.resources, resource)
    setattr(civ.resources, resource, min(current + boost, 3.0))


def _political_instability(civ: Civilization, rng: np.random.Generator):
    """
    政治动荡（内战、王位继承危机、农民起义）。
    影响：经济活动萎缩，资本外逃，军事实力下降。
    """
    severity = rng.uniform(0.1, 0.35)
    civ.gdp        *= (1 - severity)
    civ.capital    *= (1 - severity * 0.5)
    civ.military_str *= (1 - severity * 0.4)


def _great_famine(civ: Civilization, rng: np.random.Generator):
    """大饥荒：人口和农业技术双重打击"""
    civ.population *= rng.uniform(0.70, 0.90)
    civ.technology.agriculture *= rng.uniform(0.85, 0.95)


def _financial_innovation(civ: Civilization, rng: np.random.Generator):
    """
    金融创新（证券交易所、中央银行、债券市场）。
    历史背景：荷兰东印度公司（1602）是第一家股份制公司，
    英格兰银行（1694）开创现代中央银行制度，
    均大幅提升了商业技术和贸易效率。
    """
    boost = rng.uniform(0.3, 0.8)
    civ.technology.commerce = min(civ.technology.commerce + boost, 10.0)
    civ.trade_openness = min(civ.trade_openness + 0.1, 1.0)


def _colonial_resistance(civ: Civilization, rng: np.random.Generator):
    """
    殖民地反抗（起义、成本上升）：减少殖民收入，消耗军事资源。
    历史背景：七年战争后英国在北美的统治危机，
    荷兰东印度公司维护成本最终超过收益。
    """
    loss = rng.uniform(0.15, 0.40)
    civ.territories     = max(civ.territories * (1 - loss * 0.3), 1.0)
    civ.colonial_income *= (1 - loss)
    civ.military_str    *= (1 - loss * 0.2)


# ─────────────────────────────────────────────
# 事件库
# ─────────────────────────────────────────────
def build_event_pool() -> List[HistoricalEvent]:
    """
    返回历史事件池。
    每个事件包含触发时间窗口、概率、作用范围和效果。
    注意：同一事件在不同模拟 run 中可能不触发，
    这正是"历史偶然性"的来源。
    """
    return [
        # ── 全局事件 ─────────────────────────────
        HistoricalEvent(
            name="黑死病",
            year_range=(1340, 1360),
            probability=0.70,        # 历史上确实发生了，高概率
            scope="global",
            target_filter=lambda c: c.geography.coast_access > 0.3,  # 沿海城市传播更快
            effects=_black_death,
            description="鼠疫大流行，欧洲死亡约 1/3 人口（1347-1353）",
        ),
        HistoricalEvent(
            name="小冰期气候恶化",
            year_range=(1550, 1700),
            probability=0.40,
            scope="global",
            effects=lambda c, r: setattr(c, "gdp", c.gdp * r.uniform(0.92, 0.98)),
            description="全球气温下降，农业歉收，社会动荡（1550-1850）",
        ),

        # ── 技术突破（局部，随机触发）─────────────
        HistoricalEvent(
            name="航海技术突破",
            year_range=(1400, 1520),
            probability=0.25,
            scope="local",
            target_filter=lambda c: c.geography.coast_access > 0.5,
            effects=lambda c, r: (
                setattr(c.technology, "navigation",
                        min(c.technology.navigation + r.uniform(0.5, 1.2), 10.0))
            ),
            description="罗盘改良、星盘精度提升、卡拉维尔帆船设计突破",
        ),
        HistoricalEvent(
            name="印刷术革命",
            year_range=(1440, 1500),
            probability=0.30,
            scope="local",
            effects=_technological_breakthrough,
            description="谷登堡印刷机（1440）使知识扩散速度大幅提升",
        ),
        HistoricalEvent(
            name="蒸汽机突破",
            year_range=(1760, 1800),
            probability=0.35,
            scope="local",
            target_filter=lambda c: c.resources.coal > 0.8,  # 有煤才能用蒸汽机
            effects=lambda c, r: (
                setattr(c.technology, "industry",
                        min(c.technology.industry + r.uniform(1.0, 2.0), 10.0))
            ),
            description="瓦特蒸汽机（1769）及其改进，工业革命核心技术",
        ),
        HistoricalEvent(
            name="金融创新",
            year_range=(1600, 1720),
            probability=0.20,
            scope="local",
            target_filter=lambda c: c.trade_openness > 0.5,
            effects=_financial_innovation,
            description="东印度公司、证券交易所、中央银行等金融制度创新",
        ),

        # ── 贸易与政治冲击 ────────────────────────
        HistoricalEvent(
            name="奥斯曼封锁贸易路线",
            year_range=(1450, 1500),
            probability=0.50,
            scope="regional",
            target_filter=lambda c: c.geography.strategic_location > 0.6,
            effects=_trade_route_disruption,
            description="君士坦丁堡陷落（1453），地中海贸易路线受阻，推动大航海探索",
        ),
        HistoricalEvent(
            name="新大陆资源发现",
            year_range=(1490, 1560),
            probability=0.40,
            scope="local",
            target_filter=lambda c: c.technology.navigation > 3.0,
            effects=_resource_discovery,
            description="哥伦布航行（1492）、波托西银矿（1545）等资源发现",
        ),
        HistoricalEvent(
            name="政治动荡",
            year_range=(1000, 1850),
            probability=0.12,        # 长期低概率背景事件
            scope="local",
            effects=_political_instability,
            description="内战、王朝更替、宗教战争等政治危机",
        ),
        HistoricalEvent(
            name="殖民地反抗",
            year_range=(1680, 1850),
            probability=0.15,
            scope="local",
            target_filter=lambda c: c.territories > 2.0,
            effects=_colonial_resistance,
            description="殖民地起义和独立运动，增加维护成本",
        ),
        HistoricalEvent(
            name="自然灾害",
            year_range=(1000, 1850),
            probability=0.10,
            scope="local",
            effects=_natural_disaster,
            description="旱灾、洪水、火山爆发等自然灾害",
        ),
        HistoricalEvent(
            name="大饥荒",
            year_range=(1000, 1850),
            probability=0.08,
            scope="local",
            target_filter=lambda c: c.technology.agriculture < 4.0,
            effects=_great_famine,
            description="大规模粮食短缺，人口骤减",
        ),
    ]


# ─────────────────────────────────────────────
# 事件触发引擎
# ─────────────────────────────────────────────
class EventSystem:
    """负责每回合检查哪些事件触发，并记录历史"""

    def __init__(self, enabled: bool = True, rng: np.random.Generator = None):
        self.enabled = enabled
        self.rng     = rng or np.random.default_rng()
        self.pool    = build_event_pool()
        # 已触发事件日志，用于可视化标注
        self.log: List[Dict] = []

    def process_turn(
        self,
        civs: List[Civilization],
        year: int,
        era: Era,
    ) -> List[Dict]:
        """
        处理本回合的事件触发。
        返回本回合触发的事件列表（用于 UI 标注）。
        """
        if not self.enabled:
            return []

        triggered = []
        for event in self.pool:
            # 检查年份窗口
            if not (event.year_range[0] <= year <= event.year_range[1]):
                continue
            # 随机触发
            if self.rng.random() > event.probability:
                continue

            # 确定受影响的文明
            if event.scope == "global":
                targets = civs
            elif event.scope == "regional":
                # 区域事件：随机选 2-3 个相邻文明（用 filter 近似）
                candidates = [c for c in civs if event.target_filter is None or event.target_filter(c)]
                n = min(len(candidates), self.rng.integers(2, 4))
                targets = list(self.rng.choice(candidates, size=n, replace=False)) if candidates else []
            else:  # local
                candidates = [c for c in civs if event.target_filter is None or event.target_filter(c)]
                if not candidates:
                    continue
                targets = [self.rng.choice(candidates)]

            # 应用效果
            for civ in targets:
                event.effects(civ, self.rng)

            record = {
                "year":     year,
                "era":      era.name,
                "event":    event.name,
                "targets":  [c.name for c in targets],
                "scope":    event.scope,
                "description": event.description,
            }
            triggered.append(record)
            self.log.append(record)

        return triggered
