"""
strategies.py
-------------
策略代理（Strategy Agent）系统。

本模块实现两类策略：

A) 规则式策略（Rule-Based）
   直接编码历史上已知的国家决策风格：
   - MaritimeExpansionist : 葡萄牙/西班牙模式 — 优先航海+殖民
   - Mercantilist         : 法国/重商主义模式 — 保护贸易 + 商业技术
   - IndustrialPioneer    : 英国模式 — 工业投资 + 适度开放
   - AgrarianConservative : 中国/印度模式 — 农业立国 + 保守外交
   - TradeHub             : 荷兰/热那亚模式 — 极度开放 + 商业金融

B) Q-Learning 策略（Machine Learning）
   用强化学习在模拟环境中训练，根据不同奖励函数
   "自发涌现"出不同的行为模式，用于与规则式策略对比。

   奖励函数类型：
   - RL_GDP_Maximizer     : 最大化 GDP 增长率
   - RL_Power_Maximizer   : 最大化（GDP + 领土 + 军事）综合国力
   - RL_Trade_Maximizer   : 最大化贸易收益

   核心问题：不同目标函数训练出的策略，是否会收敛到
   不同的历史原型？这正是本项目想验证的 ML 假设。
"""

import numpy as np
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional
from .civilization import Civilization, Era


# ─────────────────────────────────────────────
# 决策空间定义
# ─────────────────────────────────────────────
TECH_DOMAINS = ["agriculture", "navigation", "military", "industry", "commerce"]

# 离散动作：5个技术重心 × 3个扩张等级 × 3个贸易政策 = 45种组合
# 为简化 Q-table，将其压缩为 5 种"战略模式"
STRATEGY_MODES = {
    0: {"tech_focus": "navigation",  "expansion_level": 2, "trade_policy": "open",     "savings_rate": 0.18, "tech_investment": 0.10, "label": "航海扩张"},
    1: {"tech_focus": "commerce",    "expansion_level": 0, "trade_policy": "open",     "savings_rate": 0.25, "tech_investment": 0.10, "label": "商业贸易"},
    2: {"tech_focus": "industry",    "expansion_level": 1, "trade_policy": "balanced", "savings_rate": 0.28, "tech_investment": 0.12, "label": "工业发展"},
    3: {"tech_focus": "agriculture", "expansion_level": 0, "trade_policy": "closed",   "savings_rate": 0.20, "tech_investment": 0.08, "label": "农业保守"},
    4: {"tech_focus": "military",    "expansion_level": 2, "trade_policy": "closed",   "savings_rate": 0.15, "tech_investment": 0.07, "label": "军事扩张"},
}
N_ACTIONS = len(STRATEGY_MODES)


# ─────────────────────────────────────────────
# 抽象基类
# ─────────────────────────────────────────────
class BaseStrategy(ABC):
    """所有策略的公共接口"""

    @abstractmethod
    def decide(self, civ: Civilization, all_civs, era: Era, year: int) -> Dict:
        """
        给定当前状态，返回决策字典。
        决策字典格式见 economy.apply_turn()。
        """
        pass

    def name(self) -> str:
        return self.__class__.__name__


# ─────────────────────────────────────────────
# 规则式策略 A：航海扩张型（葡萄牙/西班牙）
# ─────────────────────────────────────────────
class MaritimeExpansionist(BaseStrategy):
    """
    历史原型：15-17 世纪的葡萄牙和西班牙。
    核心逻辑：以航海技术为基础，通过殖民贸易而非本土工业积累财富。
    关键决策：
      - 优先投资航海技术（探索新航路）
      - 激进的领土扩张（建立殖民帝国）
      - 保持相对开放（需要贸易港口）
    弱点：忽视工业技术，工业革命时代落后
    """

    def decide(self, civ: Civilization, all_civs, era: Era, year: int) -> Dict:
        # 工业时代被迫转型，但转型较慢
        if era == Era.INDUSTRIAL:
            return {
                "tech_focus":      "industry",
                "expansion_level": 1,
                "trade_policy":    "open",
                "savings_rate":    0.22,
                "tech_investment": 0.10,
            }
        return {
            "tech_focus":      "navigation",
            "expansion_level": 2,
            "trade_policy":    "open",
            "savings_rate":    0.18,
            "tech_investment": 0.10,
        }


# ─────────────────────────────────────────────
# 规则式策略 B：重商主义型（法国/奥斯曼）
# ─────────────────────────────────────────────
class Mercantilist(BaseStrategy):
    """
    历史原型：17-18 世纪的法国（柯尔贝主义）、奥斯曼帝国。
    核心逻辑：国家主导贸易，通过保护性关税维持贸易顺差，
    国内制造业受保护，商业技术发达。
    关键决策：
      - 主投商业技术（组织贸易、金融工具）
      - 温和保护主义（不完全开放也不闭关）
      - 适度扩张（控制贸易节点而非大规模殖民）
    """

    def decide(self, civ: Civilization, all_civs, era: Era, year: int) -> Dict:
        # 工业时代重商主义国家开始转向工业投资，但很晚
        if era == Era.INDUSTRIAL and year > 1800:
            tech = "industry"
        else:
            tech = "commerce"

        return {
            "tech_focus":      tech,
            "expansion_level": 1,
            "trade_policy":    "balanced",
            "savings_rate":    0.24,
            "tech_investment": 0.10,
        }


# ─────────────────────────────────────────────
# 规则式策略 C：工业先驱型（英国）
# ─────────────────────────────────────────────
class IndustrialPioneer(BaseStrategy):
    """
    历史原型：18-19 世纪的英国。
    核心逻辑：率先完成工业化，以工业出口获取贸易主导权。
    关键决策：
      - 工业时代前：积累商业和航海技术
      - 工业时代：全力投入工业技术
      - 始终维持较高开放度（作为全球贸易中心）
      - 高储蓄率支撑工业投资
    历史现实：英国在 1750-1850 年人均 GDP 增长约 4 倍，
    实现"大分流"中的决定性飞跃
    """

    def decide(self, civ: Civilization, all_civs, era: Era, year: int) -> Dict:
        # 工业时代之前，积累商业和航海基础
        if era in (Era.MEDIEVAL, Era.DISCOVERY):
            tech = "navigation"
        elif era == Era.MERCANTILE:
            tech = "commerce"
        else:  # INDUSTRIAL
            tech = "industry"

        # 工业时代高储蓄率（历史上英国储蓄率约 15-25%）
        savings = 0.28 if era == Era.INDUSTRIAL else 0.22

        return {
            "tech_focus":      tech,
            "expansion_level": 1 if era != Era.INDUSTRIAL else 0,
            "trade_policy":    "open",
            "savings_rate":    savings,
            "tech_investment": 0.12 if era == Era.INDUSTRIAL else 0.09,
        }


# ─────────────────────────────────────────────
# 规则式策略 D：农业保守型（中国/印度）
# ─────────────────────────────────────────────
class AgrarianConservative(BaseStrategy):
    """
    历史原型：宋-清时代的中国、莫卧儿印度。
    核心逻辑：以高效农业支撑庞大人口，偏向内向发展，
    不主动追求海外扩张。
    关键决策：
      - 长期优先农业技术（保障粮食安全和税基）
      - 较低贸易开放度（清朝海禁、朝贡贸易体系）
      - 不扩张或很少扩张（郑和下西洋是特例，后被放弃）
    历史悖论：中国 1000-1500 年 GDP 全球最大，
    却在工业革命中落后——这正是"高水平均衡陷阱"的体现
    """

    def decide(self, civ: Civilization, all_civs, era: Era, year: int) -> Dict:
        # 检查是否有"改革窗口"（受外来冲击后短暂尝试开放）
        # 简化处理：工业时代晚期才开始被迫开放
        if era == Era.INDUSTRIAL and year > 1830:
            # 类似鸦片战争后被动开放
            return {
                "tech_focus":      "industry",
                "expansion_level": 0,
                "trade_policy":    "balanced",
                "savings_rate":    0.20,
                "tech_investment": 0.08,
            }

        return {
            "tech_focus":      "agriculture",
            "expansion_level": 0,
            "trade_policy":    "closed",
            "savings_rate":    0.22,
            "tech_investment": 0.06,
        }


# ─────────────────────────────────────────────
# 规则式策略 E：贸易枢纽型（荷兰/热那亚）
# ─────────────────────────────────────────────
class TradeHub(BaseStrategy):
    """
    历史原型：17 世纪荷兰、中世纪热那亚、威尼斯。
    核心逻辑：充分发挥战略地理位置，成为国际贸易中转站，
    通过金融创新（证券交易所、保险、债券）放大贸易收益。
    关键决策：
      - 极高贸易开放度（越开放，中转费越高）
      - 优先商业和金融技术
      - 几乎不扩张领土（资源有限，专注贸易）
    荷兰案例：17 世纪控制全球 50% 的海运，
    人均 GDP 超过英国，但人口只有法国的 1/6
    """

    def decide(self, civ: Civilization, all_civs, era: Era, year: int) -> Dict:
        # 工业时代商业模式受冲击，需要转型
        if era == Era.INDUSTRIAL:
            tech = "industry"
            policy = "open"
        else:
            tech = "commerce"
            policy = "open"

        return {
            "tech_focus":      tech,
            "expansion_level": 0,
            "trade_policy":    policy,
            "savings_rate":    0.26,
            "tech_investment": 0.11,
        }


# ─────────────────────────────────────────────
# Q-Learning 策略（机器学习）
# ─────────────────────────────────────────────
class QLearningStrategy(BaseStrategy):
    """
    使用表格型 Q-learning 训练的策略。

    状态空间设计：
      将 12 维连续状态向量离散化为 4^6 = 4096 种状态
      （取 6 个最重要的特征，每个分 4 个等级）
      这是 Q-table 大小和训练速度之间的权衡。

    动作空间：
      5 种战略模式（见 STRATEGY_MODES）

    训练方式：
      在 SimulationEngine 中进行 N 个 episode 的 self-play 训练，
      每个文明独立学习，不知道其他文明的内部状态（部分可观测）。

    奖励函数：
      可配置为 "gdp"（最大化 GDP 增长）、
                "power"（最大化综合国力）、
                "trade"（最大化贸易收益）
    """

    def __init__(
        self,
        reward_type: str = "gdp",   # "gdp" | "power" | "trade"
        learning_rate: float = 0.15,
        discount:      float = 0.90,
        epsilon:       float = 0.20,  # ε-greedy 探索率
        label: str = None,
    ):
        self.reward_type   = reward_type
        self.lr            = learning_rate
        self.gamma         = discount
        self.epsilon       = epsilon
        self.label_name    = label or f"RL_{reward_type}"

        # Q-table: 字典形式，只存访问过的状态（稀疏）
        # key: 离散状态元组, value: ndarray(N_ACTIONS,)
        self.q_table: Dict[tuple, np.ndarray] = {}

        self.is_trained = False
        self._prev_state: Optional[tuple] = None
        self._prev_action: Optional[int]  = None
        self._prev_gdp:    float = 0.0
        self._prev_trade:  float = 0.0
        self._prev_power:  float = 0.0

    def name(self) -> str:
        return self.label_name

    # ─── 状态离散化 ──────────────────────────────

    def _discretize(self, state_vec: np.ndarray) -> tuple:
        """
        将连续状态向量压缩为离散索引元组。
        取最关键的 6 个维度，每个分成 4 个等级（0-3）。

        选取的 6 个特征（对应 civilization.state_vector 的索引）：
          0  agriculture_tech
          1  navigation_tech
          3  industry_tech
          4  commerce_tech
          5  relative_gdp
          11 era_index
        """
        key_indices = [0, 1, 3, 4, 5, 11]
        thresholds  = [0.25, 0.50, 0.75]  # 4 级别：[0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
        disc = []
        for i in key_indices:
            v = float(np.clip(state_vec[i], 0.0, 1.0))
            level = sum(v >= th for th in thresholds)
            disc.append(level)
        return tuple(disc)

    def _get_q(self, state: tuple) -> np.ndarray:
        """获取 Q 值（不存在则初始化为小随机值，鼓励探索）"""
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(0.0, 0.1, N_ACTIONS)
        return self.q_table[state]

    # ─── 动作选择（ε-greedy）─────────────────────

    def _choose_action(self, state: tuple, training: bool) -> int:
        """
        训练时用 ε-greedy：以概率 ε 随机探索，以 1-ε 选最优动作。
        推理时直接选 Q 值最高的动作（贪婪策略）。
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self._get_q(state)))

    # ─── 奖励计算 ─────────────────────────────────

    def _compute_reward(self, civ: Civilization) -> float:
        """
        根据 reward_type 计算本回合奖励。

        奖励设计的核心问题：
          - 纯 GDP 奖励 → 智能体学会工业优先（类似英国）
          - 领土/军事奖励 → 智能体学会航海扩张（类似西班牙）
          - 贸易收益奖励 → 智能体学会开放贸易（类似荷兰）
        通过对比这三种涌现策略与规则式策略的相似度，
        可以验证"历史上的国家行为是否是对某种目标函数的优化"
        """
        if self.reward_type == "gdp":
            # GDP 增长率（对数差分，防止绝对值主导）
            reward = np.log(civ.gdp + 1) - np.log(self._prev_gdp + 1)
        elif self.reward_type == "power":
            # 综合国力 = GDP × 军事 × 领土
            power = civ.gdp * civ.military_str * civ.territories
            reward = np.log(power + 1) - np.log(self._prev_power + 1)
        elif self.reward_type == "trade":
            # 贸易收益绝对值 + 增量奖励
            reward = civ.trade_income - self._prev_trade + 0.1 * civ.trade_income
        else:
            reward = civ.gdp - self._prev_gdp

        return float(reward)

    # ─── Q-learning 更新 ──────────────────────────

    def update(self, civ: Civilization, state_vec: np.ndarray, era: Era) -> None:
        """
        在每回合结束后调用，执行 Q 值更新（Bellman 方程）：
          Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        """
        if self._prev_state is None:
            return

        reward = self._compute_reward(civ)
        new_state = self._discretize(state_vec)

        prev_q = self._get_q(self._prev_state)
        next_q = self._get_q(new_state)

        # Bellman 更新
        td_target = reward + self.gamma * np.max(next_q)
        td_error  = td_target - prev_q[self._prev_action]
        prev_q[self._prev_action] += self.lr * td_error

        # 保存当前状态值供下一回合计算奖励差
        self._prev_gdp   = civ.gdp
        self._prev_trade = civ.trade_income
        self._prev_power = civ.gdp * civ.military_str * civ.territories

    # ─── 主决策接口 ──────────────────────────────

    def decide(
        self,
        civ: Civilization,
        all_civs,
        era: Era,
        year: int,
        training: bool = False,
    ) -> Dict:
        """
        返回决策字典。
        training=True 时会做 Q 值更新（训练阶段）；
        training=False 时只做推理（评估/展示阶段）。
        """
        world_avg_gdp = np.mean([c.gdp for c in all_civs])
        state_vec     = civ.state_vector(world_avg_gdp, era)
        state         = self._discretize(state_vec)

        action = self._choose_action(state, training)

        # 记录，供下回合 update() 使用
        self._prev_state  = state
        self._prev_action = action
        self._prev_gdp    = civ.gdp
        self._prev_trade  = civ.trade_income
        self._prev_power  = civ.gdp * civ.military_str * civ.territories

        return dict(STRATEGY_MODES[action])  # 返回副本，防止被修改

    # ─── 训练后降低探索率（模拟策略成熟）──────────

    def decay_epsilon(self, factor: float = 0.95) -> None:
        """随训练轮次增加，逐步降低随机探索比例"""
        self.epsilon = max(self.epsilon * factor, 0.02)

    # ─── 保存/加载 Q-table ────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "reward_type": self.reward_type}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table     = data["q_table"]
        self.reward_type = data["reward_type"]
        self.is_trained  = True
        self.epsilon     = 0.02  # 加载后用接近贪婪的策略

    # ─── 策略分析：返回每种模式的偏好频率 ──────────

    def action_distribution(self) -> Dict[str, float]:
        """
        统计在所有已访问状态上，哪种动作模式的 Q 值最高。
        用于可视化"该 RL 策略最偏爱哪种历史决策模式"。
        """
        counts = np.zeros(N_ACTIONS)
        for q_vals in self.q_table.values():
            counts[np.argmax(q_vals)] += 1
        total = counts.sum()
        if total == 0:
            return {STRATEGY_MODES[i]["label"]: 0.0 for i in range(N_ACTIONS)}
        return {STRATEGY_MODES[i]["label"]: counts[i] / total for i in range(N_ACTIONS)}


# ─────────────────────────────────────────────
# 策略工厂：按名字返回策略对象
# ─────────────────────────────────────────────
RULE_BASED_STRATEGIES = {
    "MaritimeExpansionist": MaritimeExpansionist,
    "Mercantilist":         Mercantilist,
    "IndustrialPioneer":    IndustrialPioneer,
    "AgrarianConservative": AgrarianConservative,
    "TradeHub":             TradeHub,
}

def make_strategy(name: str) -> BaseStrategy:
    """
    根据策略名称返回对应实例。
    支持规则式策略和 RL 策略：
      "RL_gdp"   → Q-learning，奖励函数为 GDP 增长
      "RL_power" → Q-learning，奖励函数为综合国力
      "RL_trade" → Q-learning，奖励函数为贸易收益
    """
    if name.startswith("RL_"):
        reward_type = name.split("_", 1)[1]
        return QLearningStrategy(reward_type=reward_type, label=name)
    if name in RULE_BASED_STRATEGIES:
        return RULE_BASED_STRATEGIES[name]()
    raise ValueError(f"Unknown strategy: {name}. "
                     f"Available: {list(RULE_BASED_STRATEGIES.keys())} + RL_gdp/RL_power/RL_trade")
