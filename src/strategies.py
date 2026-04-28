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
TECH_DOMAINS   = ["agriculture", "navigation", "military", "industry", "commerce"]
_TECH_CHOICES  = ["agriculture", "navigation", "military", "industry", "commerce"]  # 5
_TRADE_CHOICES = ["open", "balanced", "closed"]  # 3
_EXP_CHOICES   = [0, 1, 2]                       # 3
# 完整组合：5 × 3 × 3 = 45 个离散动作（不再压缩为预设模式）
N_ACTIONS      = len(_TECH_CHOICES) * len(_TRADE_CHOICES) * len(_EXP_CHOICES)

# 储蓄率和技术投资率跟随 tech_focus（各技术领域的投入偏好）
_TECH_ECON = {
    "agriculture": {"savings_rate": 0.20, "tech_investment": 0.08},
    "navigation":  {"savings_rate": 0.18, "tech_investment": 0.10},
    "military":    {"savings_rate": 0.15, "tech_investment": 0.07},
    "industry":    {"savings_rate": 0.28, "tech_investment": 0.12},
    "commerce":    {"savings_rate": 0.25, "tech_investment": 0.10},
}


def _action_to_decision(action_idx: int) -> Dict:
    """将整数动作索引（0-44）解码为完整的决策字典。"""
    nt, ne = len(_TRADE_CHOICES), len(_EXP_CHOICES)
    ti = action_idx // (nt * ne)
    ri = (action_idx % (nt * ne)) // ne
    ei = action_idx % ne
    tech = _TECH_CHOICES[ti]
    return {
        "tech_focus":      tech,
        "trade_policy":    _TRADE_CHOICES[ri],
        "expansion_level": _EXP_CHOICES[ei],
        **_TECH_ECON[tech],
    }


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
    Double Q-Learning 策略。

    改进要点（相对原始版本）：
      1. 动作空间 5 → 45（tech×trade×expand 三维独立选择）
         RL 可以自由发现任意组合，不受预设模式限制
      2. 状态维度 6 → 8（新增 military_tech、trade_openness）
         理论状态数 4^8 = 65536，稀疏字典存储
      3. Double Q-Learning：双表（q_a/q_b）交替更新
         q_a 选动作，q_b 评估目标值，减少 Q 值高估偏差
      4. 奖励裁剪到 [-5, 5]，防止异常值破坏 Q 表
      5. 竞争激励（可选）：在基础奖励上叠加
         本国 GDP 相对对手均值的优势项，
         鼓励智能体对竞争态势做出响应
    """

    def __init__(
        self,
        reward_type:        str   = "gdp",   # "gdp" | "power" | "trade"
        learning_rate:      float = 0.15,
        discount:           float = 0.90,
        epsilon:            float = 0.30,    # 初始探索率（由 engine 统一线性衰减）
        competitive:        bool  = False,   # 竞争激励开关
        competitive_weight: float = 0.25,   # 竞争奖励项的权重
        label:              str   = None,
    ):
        self.reward_type        = reward_type
        self.lr                 = learning_rate
        self.gamma              = discount
        self.epsilon            = epsilon
        self.competitive        = competitive
        self.competitive_weight = competitive_weight
        self.label_name         = label or f"RL_{reward_type}"

        # 双 Q 表：稀疏字典，key=离散状态元组, value=ndarray(N_ACTIONS,)
        self.q_a: Dict[tuple, np.ndarray] = {}
        self.q_b: Dict[tuple, np.ndarray] = {}

        self.is_trained        = False
        self._prev_state:  Optional[tuple] = None
        self._prev_action: Optional[int]   = None
        self._prev_gdp:    float = 0.0
        self._prev_trade:  float = 0.0
        self._prev_power:  float = 0.0
        self._prev_tech:   float = 0.0

    def name(self) -> str:
        return self.label_name

    @property
    def q_table(self) -> Dict:
        """向后兼容：返回 q_a（用于 len() 统计已探索状态数）"""
        return self.q_a

    # ─── 状态离散化（8 维）──────────────────────

    def _discretize(self, state_vec: np.ndarray) -> tuple:
        """
        将连续状态向量离散化为 8 个 4-level 特征的元组。
        理论最大状态数 4^8 = 65536，实际稀疏存储远少于此。

        选取特征（对应 civilization.state_vector 索引）：
          0  agriculture_tech
          1  navigation_tech
          2  military_tech      ← 相比原版新增
          3  industry_tech
          4  commerce_tech
          5  relative_gdp
          6  trade_openness     ← 相比原版新增
          11 era_index
        """
        key_indices = [0, 1, 2, 3, 4, 5, 6, 11]
        thresholds  = [0.25, 0.50, 0.75]
        disc = []
        for i in key_indices:
            v = float(np.clip(state_vec[i], 0.0, 1.0))
            disc.append(sum(v >= th for th in thresholds))
        return tuple(disc)

    def _get_q(self, state: tuple, table: Dict) -> np.ndarray:
        """从指定表中取 Q 值（不存在则用小随机值初始化，鼓励早期探索）"""
        if state not in table:
            table[state] = np.random.uniform(0.0, 0.05, N_ACTIONS)
        return table[state]

    # ─── 动作选择（ε-greedy，双表均值）─────────

    def _choose_action(self, state: tuple, training: bool) -> int:
        """训练时 ε-greedy 探索；推理时用双表均值选最优动作。"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q_avg = (self._get_q(state, self.q_a) + self._get_q(state, self.q_b)) / 2.0
        return int(np.argmax(q_avg))

    # ─── 奖励计算 ─────────────────────────────────

    def _compute_reward(self, civ: Civilization, era: Era, all_civs=None) -> float:
        """
        base reward（按 reward_type）+ 技术进度辅助奖励 + 可选竞争激励，最终裁剪到 [-5, 5]。

        技术进度辅助奖励（所有类型通用）：
          tech_bonus = 0.3 × Δtech_composite
          提供更及时的中间信号，帮助 RL 发现"先投技术后获益"的长期策略。

        竞争激励项：weight × (own_power - opp_avg_power) / opp_avg_power
        """
        tech_now   = civ.technology.composite(era)
        tech_bonus = 0.3 * (tech_now - self._prev_tech)

        if self.reward_type == "gdp":
            base = np.log(civ.gdp + 1) - np.log(self._prev_gdp + 1) + tech_bonus
        elif self.reward_type == "power":
            power = civ.gdp * civ.military_str * civ.territories
            base  = np.log(power + 1) - np.log(self._prev_power + 1) + tech_bonus
        elif self.reward_type == "trade":
            base = (civ.trade_income - self._prev_trade) + tech_bonus
        else:
            base = civ.gdp - self._prev_gdp + tech_bonus

        comp = 0.0
        if self.competitive and all_civs and len(all_civs) > 1:
            opp_power = np.mean([c.gdp * c.military_str * c.territories for c in all_civs if c.name != civ.name])
            comp      = self.competitive_weight * (civ.gdp * civ.military_str * civ.territories - opp_power) / (opp_power + 1e-6)

        return float(np.clip(base + comp, -5.0, 5.0))

    # ─── Double Q-Learning 更新 ──────────────────

    def update(self, civ: Civilization, state_vec: np.ndarray, era: Era,
               all_civs=None) -> float:
        """
        每回合结束后调用，执行 Double Q-Learning 更新。

        Double Q-Learning（Hasselt 2010）：
          随机选定"主表"qa 和"副表"qb
          - 用 qa 选择 argmax 动作
          - 用 qb 计算该动作的目标值
          - 只更新 qa
        避免单表同时用于选择和评估导致的高估问题。

        返回 TD-error 供训练监控使用。
        """
        if self._prev_state is None:
            # 第一步只记录初始值，不更新
            self._prev_gdp   = civ.gdp
            self._prev_trade = civ.trade_income
            self._prev_power = civ.gdp * civ.military_str * civ.territories
            self._prev_tech  = civ.technology.composite(era)
            return 0.0

        reward    = self._compute_reward(civ, era, all_civs)
        new_state = self._discretize(state_vec)

        # 随机决定哪张表做更新（交替以保证对称）
        if np.random.random() < 0.5:
            qa, qb = self.q_a, self.q_b
        else:
            qa, qb = self.q_b, self.q_a

        q_curr    = self._get_q(self._prev_state, qa)
        best_next = int(np.argmax(self._get_q(new_state, qa)))   # qa 选动作
        td_target = reward + self.gamma * self._get_q(new_state, qb)[best_next]  # qb 评估
        td_error  = td_target - q_curr[self._prev_action]
        q_curr[self._prev_action] += self.lr * td_error

        self._prev_gdp   = civ.gdp
        self._prev_trade = civ.trade_income
        self._prev_power = civ.gdp * civ.military_str * civ.territories
        self._prev_tech  = civ.technology.composite(era)

        return float(td_error)

    # ─── 主决策接口 ──────────────────────────────

    def decide(self, civ: Civilization, all_civs, era: Era, year: int,
               training: bool = False) -> Dict:
        world_avg_gdp = np.mean([c.gdp for c in all_civs])
        state_vec     = civ.state_vector(world_avg_gdp, era)
        state         = self._discretize(state_vec)
        action        = self._choose_action(state, training)

        self._prev_state  = state
        self._prev_action = action
        self._prev_gdp    = civ.gdp
        self._prev_trade  = civ.trade_income
        self._prev_power  = civ.gdp * civ.military_str * civ.territories
        self._prev_tech   = civ.technology.composite(era)

        return _action_to_decision(action)

    def set_epsilon(self, eps: float) -> None:
        """由 engine 统一调用，设置当前探索率。"""
        self.epsilon = float(np.clip(eps, 0.0, 1.0))

    # ─── 保存/加载 ────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "q_a": self.q_a, "q_b": self.q_b,
                "reward_type": self.reward_type,
                "competitive": self.competitive,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_a         = data.get("q_a", data.get("q_table", {}))
        self.q_b         = data.get("q_b", {})
        self.reward_type = data["reward_type"]
        self.competitive = data.get("competitive", False)
        self.is_trained  = True
        self.epsilon     = 0.02

    # ─── 策略分析 ─────────────────────────────────

    def action_distribution(self) -> Dict[str, float]:
        """
        统计在所有已访问状态上，双表均值 Q 最高的动作按 tech_focus 的分布。
        """
        counts    = np.zeros(N_ACTIONS)
        nt, ne    = len(_TRADE_CHOICES), len(_EXP_CHOICES)
        all_states = set(self.q_a) | set(self.q_b)
        for s in all_states:
            q_avg = (self._get_q(s, self.q_a) + self._get_q(s, self.q_b)) / 2.0
            counts[int(np.argmax(q_avg))] += 1
        total  = counts.sum()
        if total == 0:
            return {t: 0.0 for t in _TECH_CHOICES}
        result = {}
        for ti, tech in enumerate(_TECH_CHOICES):
            result[tech] = float(counts[ti * nt * ne : (ti + 1) * nt * ne].sum() / total)
        return result


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
