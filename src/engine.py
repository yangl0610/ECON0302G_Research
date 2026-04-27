"""
engine.py
---------
模拟引擎（SimulationEngine）。

职责：
  1. 管理世界上所有文明和它们的策略代理
  2. 驱动历史时间轴（逐回合推进）
  3. 协调经济更新、技术扩散、事件触发
  4. 支持 Q-learning 智能体的训练模式
  5. 将所有历史数据汇总为 pandas DataFrame，供可视化使用

使用方式：
  engine = SimulationEngine(civs, strategies, seed=42)
  engine.run()                  # 跑完整模拟
  df = engine.get_history_df() # 获取历史 DataFrame
  events = engine.event_log     # 获取事件日志
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from .civilization import Civilization, Era, TURNS_PER_ERA, ERA_YEARS
from .economy import apply_turn
from .strategies import BaseStrategy, QLearningStrategy
from .events import EventSystem


# ─────────────────────────────────────────────
# 预设文明配置（历史上的 8 个主要文明/地区）
# ─────────────────────────────────────────────
from .civilization import Geography, Resources

def build_default_civs() -> List[Civilization]:
    """
    返回 8 个历史文明的初始配置。
    地理参数和资源参数基于历史研究粗略校准：
      - 中国：高地理质量、高粮食，但低煤炭（大部分煤炭在内陆）
      - 西欧：高海岸、战略位置优越，工业时代前中期
      - 奥斯曼：战略位置极高（控制东西贸易要道）
      - 印度：高农业、高奢侈品（香料、棉布）
      - 葡/西（伊比利亚）：极高海岸，优先航海
      - 荷/英（西北欧）：高贸易战略位置，有煤炭
      - 非洲（撒哈拉以南）：高奢侈品（黄金）但低航海
      - 美洲（阿兹特克/印加）：高金属（白银），与外界隔绝

    纬度/经度仅用于世界地图可视化。
    """
    from .civilization import Geography, Resources

    return [
        Civilization(
            name="中华帝国",
            geography=Geography(
                coast_access=0.45,       # 有东南沿海，但海禁政策
                terrain_quality=0.85,    # 黄河/长江流域，极优农业
                climate_score=0.75,
                river_density=0.80,      # 大运河体系
                strategic_location=0.55, # 丝绸之路东端，但非核心节点
            ),
            resources=Resources(food=2.5, metal=1.5, wood=1.2, luxury=1.8, coal=0.8),
            strategy_name="AgrarianConservative",
            color="#e63946",
            lat=35.0, lon=105.0,
        ),
        Civilization(
            name="西欧诸国",
            geography=Geography(
                coast_access=0.70,
                terrain_quality=0.65,
                climate_score=0.60,
                river_density=0.65,
                strategic_location=0.60,
            ),
            resources=Resources(food=1.8, metal=1.6, wood=1.5, luxury=0.8, coal=1.2),
            strategy_name="Mercantilist",
            color="#457b9d",
            lat=48.0, lon=8.0,
        ),
        Civilization(
            name="伊比利亚（葡西）",
            geography=Geography(
                coast_access=0.90,       # 大西洋+地中海双面海岸
                terrain_quality=0.50,
                climate_score=0.65,
                river_density=0.40,
                strategic_location=0.70, # 连接地中海和大西洋
            ),
            resources=Resources(food=1.2, metal=1.0, wood=1.3, luxury=0.6, coal=0.3),
            strategy_name="MaritimeExpansionist",
            color="#f4a261",
            lat=40.0, lon=-5.0,
        ),
        Civilization(
            name="西北欧（荷英）",
            geography=Geography(
                coast_access=0.85,
                terrain_quality=0.55,
                climate_score=0.55,
                river_density=0.75,      # 莱茵河、泰晤士河
                strategic_location=0.80, # 北海贸易中心
            ),
            resources=Resources(food=1.4, metal=1.3, wood=1.2, luxury=0.5, coal=1.8),  # 英国煤矿
            strategy_name="TradeHub",
            color="#2a9d8f",
            lat=52.0, lon=4.0,
        ),
        Civilization(
            name="奥斯曼帝国",
            geography=Geography(
                coast_access=0.65,
                terrain_quality=0.60,
                climate_score=0.65,
                river_density=0.55,
                strategic_location=0.95, # 控制欧亚贸易要道（博斯普鲁斯海峡）
            ),
            resources=Resources(food=1.5, metal=1.2, wood=0.9, luxury=1.4, coal=0.4),
            strategy_name="Mercantilist",
            color="#e9c46a",
            lat=39.0, lon=35.0,
        ),
        Civilization(
            name="印度次大陆",
            geography=Geography(
                coast_access=0.60,
                terrain_quality=0.80,    # 恒河平原
                climate_score=0.70,
                river_density=0.70,
                strategic_location=0.65, # 印度洋贸易中心
            ),
            resources=Resources(food=2.2, metal=1.0, wood=1.4, luxury=2.0, coal=0.6),
            strategy_name="AgrarianConservative",
            color="#a8dadc",
            lat=22.0, lon=80.0,
        ),
        Civilization(
            name="撒哈拉以南非洲",
            geography=Geography(
                coast_access=0.35,
                terrain_quality=0.60,
                climate_score=0.50,
                river_density=0.55,
                strategic_location=0.30,
            ),
            resources=Resources(food=1.8, metal=0.8, wood=1.5, luxury=1.5, coal=0.2),
            strategy_name="AgrarianConservative",
            color="#6d4c41",
            lat=2.0, lon=22.0,
        ),
        Civilization(
            name="美洲文明",
            geography=Geography(
                coast_access=0.50,
                terrain_quality=0.70,
                climate_score=0.65,
                river_density=0.60,
                strategic_location=0.10, # 与旧大陆隔绝，大发现后急剧变化
            ),
            resources=Resources(food=2.0, metal=0.7, wood=1.8, luxury=0.8, coal=0.4),
            strategy_name="AgrarianConservative",
            color="#9c6644",
            lat=15.0, lon=-85.0,
        ),
    ]


# ─────────────────────────────────────────────
# 模拟引擎主体
# ─────────────────────────────────────────────
class SimulationEngine:
    """
    主模拟引擎。

    参数：
      civs          : 文明列表（若为 None 则使用默认 8 个文明）
      strategy_map  : {文明名: Strategy 对象} 的字典
                      若某文明未在字典中，使用其默认 strategy_name
      events_enabled: 是否开启历史随机事件
      noise_std     : 随机扰动强度（模拟历史偶然性）
      seed          : 随机种子（控制可复现性）
      training_mode : 是否为 RL 训练模式（会调用 strategy.update()）
    """

    def __init__(
        self,
        civs: Optional[List[Civilization]] = None,
        strategy_map: Optional[Dict[str, BaseStrategy]] = None,
        events_enabled: bool = True,
        noise_std: float = 0.025,
        seed: int = 42,
        training_mode: bool = False,
    ):
        self.civs           = civs if civs is not None else build_default_civs()
        self.strategy_map   = strategy_map or {}
        self.events_enabled = events_enabled
        self.noise_std      = noise_std
        self.seed           = seed
        self.training_mode  = training_mode

        self.rng = np.random.default_rng(seed)

        # 将默认策略名解析为策略对象
        from .strategies import make_strategy
        for civ in self.civs:
            if civ.name not in self.strategy_map:
                self.strategy_map[civ.name] = make_strategy(civ.strategy_name)

        self.event_system = EventSystem(enabled=events_enabled, rng=self.rng)

        # 当前模拟进度
        self.current_year = 1000
        self.current_era  = Era.MEDIEVAL
        self._era_index   = 0

        # 汇总记录
        self.turn_log: List[Dict] = []

    # ─── 时期管理 ──────────────────────────────────────

    def _get_era(self, year: int) -> Era:
        """根据年份返回历史时期"""
        if year < 1400:
            return Era.MEDIEVAL
        elif year < 1600:
            return Era.DISCOVERY
        elif year < 1750:
            return Era.MERCANTILE
        else:
            return Era.INDUSTRIAL

    def _year_step(self, era: Era) -> int:
        """每回合推进的年数"""
        steps = {
            Era.MEDIEVAL:   20,   # 中世纪每步 20 年（变化慢）
            Era.DISCOVERY:  10,
            Era.MERCANTILE: 10,
            Era.INDUSTRIAL: 10,
        }
        return steps[era]

    # ─── 单回合执行 ────────────────────────────────────

    def step(self) -> Dict:
        """
        执行一个回合：
          1. 触发随机事件
          2. 每个文明由策略代理做决策
          3. 经济模型更新所有状态
          4. RL 智能体执行 Q 值更新（训练模式）
          5. 记录本回合数据
        """
        year = self.current_year
        era  = self._get_era(year)
        self.current_era = era

        # 1. 随机事件
        events_this_turn = self.event_system.process_turn(self.civs, year, era)

        # 2. 每个文明做决策并执行经济更新
        decisions_this_turn = {}
        for civ in self.civs:
            strategy = self.strategy_map[civ.name]

            # QL 策略需要传 training 参数
            if isinstance(strategy, QLearningStrategy):
                decision = strategy.decide(civ, self.civs, era, year, training=self.training_mode)
            else:
                decision = strategy.decide(civ, self.civs, era, year)

            decisions_this_turn[civ.name] = decision

            # 执行经济更新
            apply_turn(civ, self.civs, era, decision, noise_std=self.noise_std, rng=self.rng)

        # 3. RL 智能体更新 Q 值（必须在所有文明都完成经济更新后）
        if self.training_mode:
            world_avg_gdp = np.mean([c.gdp for c in self.civs])
            for civ in self.civs:
                strategy = self.strategy_map[civ.name]
                if isinstance(strategy, QLearningStrategy):
                    state_vec = civ.state_vector(world_avg_gdp, era)
                    strategy.update(civ, state_vec, era)

        # 4. 记录所有文明的当前状态
        for civ in self.civs:
            civ.record(year, era)

        # 5. 汇总本回合日志
        turn_record = {
            "year":   year,
            "era":    era.name,
            "events": events_this_turn,
        }
        self.turn_log.append(turn_record)

        # 推进时间
        self.current_year += self._year_step(era)

        return turn_record

    # ─── 完整运行 ──────────────────────────────────────

    def run(self, end_year: int = 1850) -> None:
        """运行模拟直到目标年份"""
        while self.current_year <= end_year:
            self.step()

    # ─── 训练 RL 智能体 ────────────────────────────────

    def train_rl_agents(self, n_episodes: int = 80) -> Dict[str, List[float]]:
        """
        对所有 QLearningStrategy 智能体进行多轮训练。

        每个 episode = 一次完整的历史模拟（1000-1850 年）。
        每个 episode 结束后重置文明状态，衰减探索率。

        返回：各 RL 智能体的每轮总 GDP 训练曲线（用于展示学习过程）
        """
        from copy import deepcopy

        # 找出所有 QL 策略
        rl_agents = {
            name: strat
            for name, strat in self.strategy_map.items()
            if isinstance(strat, QLearningStrategy)
        }
        if not rl_agents:
            return {}

        # 保存初始文明状态，每个 episode 后重置
        initial_civs = deepcopy(self.civs)
        training_curves: Dict[str, List[float]] = {name: [] for name in rl_agents}

        print(f"开始训练 {len(rl_agents)} 个 RL 智能体，共 {n_episodes} 轮...")

        for episode in range(n_episodes):
            # 重置文明状态
            self.civs = deepcopy(initial_civs)
            self.current_year = 1000
            self.event_system.log.clear()
            self.turn_log.clear()
            self.rng = np.random.default_rng(self.seed + episode)  # 不同 episode 用不同随机种子
            self.event_system.rng = self.rng

            # 重置 RL 策略的 prev_state（开始新 episode）
            for strat in rl_agents.values():
                strat._prev_state  = None
                strat._prev_action = None

            # 跑完一次模拟
            self.run()

            # 记录训练曲线（各 RL 文明的最终 GDP）
            for civ_name, strat in rl_agents.items():
                matching = [c for c in self.civs if c.name == civ_name]
                if matching:
                    training_curves[civ_name].append(matching[0].gdp)

            # 衰减探索率（随训练进行，逐渐减少随机探索）
            for strat in rl_agents.values():
                strat.decay_epsilon()

            if (episode + 1) % 20 == 0:
                avg_gdps = {n: training_curves[n][-1] for n in rl_agents}
                print(f"  Episode {episode+1}/{n_episodes}: {avg_gdps}")

        print("训练完成！")

        # 训练完成后标记
        for strat in rl_agents.values():
            strat.is_trained = True
            strat.epsilon    = 0.03  # 推理时保留少量探索

        # 恢复初始状态，准备正式模拟
        self.civs = deepcopy(initial_civs)
        self.current_year = 1000
        self.training_mode = False

        return training_curves

    # ─── 数据导出 ──────────────────────────────────────

    def get_history_df(self) -> pd.DataFrame:
        """
        将所有文明的历史记录合并为一个 DataFrame。
        每行 = 一个文明在某一年份的状态快照。
        这是可视化模块的主要数据来源。
        """
        rows = []
        for civ in self.civs:
            h = civ.history
            n = len(h["year"])
            for i in range(n):
                rows.append({
                    "civilization":   civ.name,
                    "strategy":       civ.strategy_name,
                    "color":          civ.color,
                    "lat":            civ.lat,
                    "lon":            civ.lon,
                    "year":           h["year"][i],
                    "era":            h["era"][i],
                    "gdp":            h["gdp"][i],
                    "gdp_per_capita": h["gdp_per_capita"][i],
                    "population":     h["population"][i],
                    "tech_composite": h["tech_composite"][i],
                    "trade_income":   h["trade_income"][i],
                    "colonial_income":h["colonial_income"][i],
                    "territories":    h["territories"][i],
                    "trade_openness": h["trade_openness"][i],
                    "military_str":   h["military_str"][i],
                    "tech_agri":      h["tech_agri"][i],
                    "tech_nav":       h["tech_nav"][i],
                    "tech_mil":       h["tech_mil"][i],
                    "tech_ind":       h["tech_ind"][i],
                    "tech_com":       h["tech_com"][i],
                })
        return pd.DataFrame(rows)

    def get_event_df(self) -> pd.DataFrame:
        """返回所有触发事件的记录 DataFrame"""
        rows = []
        for rec in self.event_system.log:
            for target in rec["targets"]:
                rows.append({
                    "year":        rec["year"],
                    "era":         rec["era"],
                    "event":       rec["event"],
                    "target":      target,
                    "scope":       rec["scope"],
                    "description": rec["description"],
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["year", "era", "event", "target", "scope", "description"]
        )

    def get_strategy_summary(self) -> Dict:
        """
        返回每个 QL 策略的动作分布分析，
        用于可视化"ML 智能体偏好哪种历史决策模式"。
        """
        summary = {}
        for name, strat in self.strategy_map.items():
            if isinstance(strat, QLearningStrategy):
                summary[name] = {
                    "reward_type":         strat.reward_type,
                    "action_distribution": strat.action_distribution(),
                    "q_table_size":        len(strat.q_table),
                }
        return summary
