"""
archetypes.py
文明原型（Archetype）与竞争模式（Competition Mode）配置。

不再绑定任何真实历史国家，5 种原型代表 5 种抽象发展路径。
通过不同的竞争模式组合，观察不同策略在同一规则下的长期演化结果。
"""

from typing import Dict, List
from .civilization import Civilization, Geography, Resources


# ─────────────────────────────────────────────
# 5 种文明原型的参数配置
# ─────────────────────────────────────────────
_CONFIGS: Dict[str, dict] = {
    "A": {
        "geo":      dict(coast_access=0.30, terrain_quality=0.90, climate_score=0.80,
                         river_density=0.85, strategic_location=0.40),
        "res":      dict(food=2.8, metal=1.2, wood=1.5, luxury=1.0, coal=0.5),
        "strategy": "AgrarianConservative",
        "color":    "#2a9d8f",
        "desc":     "Agrarian — high population, conservative",
    },
    "B": {
        "geo":      dict(coast_access=0.92, terrain_quality=0.50, climate_score=0.60,
                         river_density=0.45, strategic_location=0.80),
        "res":      dict(food=1.0, metal=1.0, wood=1.3, luxury=0.8, coal=0.4),
        "strategy": "MaritimeExpansionist",
        "color":    "#457b9d",
        "desc":     "Maritime — coastal, expansionist",
    },
    "C": {
        "geo":      dict(coast_access=0.70, terrain_quality=0.60, climate_score=0.60,
                         river_density=0.70, strategic_location=0.70),
        "res":      dict(food=1.3, metal=1.5, wood=1.2, luxury=0.5, coal=2.2),
        "strategy": "IndustrialPioneer",
        "color":    "#e63946",
        "desc":     "Industrial — coal-rich, late surge",
    },
    "D": {
        "geo":      dict(coast_access=0.80, terrain_quality=0.50, climate_score=0.55,
                         river_density=0.65, strategic_location=0.95),
        "res":      dict(food=1.1, metal=1.0, wood=1.1, luxury=1.8, coal=0.7),
        "strategy": "TradeHub",
        "color":    "#f4a261",
        "desc":     "Trade hub — high per-capita, low military",
    },
    "E": {
        "geo":      dict(coast_access=0.65, terrain_quality=0.65, climate_score=0.65,
                         river_density=0.60, strategic_location=0.75),
        "res":      dict(food=1.6, metal=1.8, wood=1.0, luxury=1.2, coal=0.9),
        "strategy": "Mercantilist",
        "color":    "#9c6644",
        "desc":     "Mercantilist — balanced military-commerce",
    },
}

# ─────────────────────────────────────────────
# 5 种竞争模式
# ─────────────────────────────────────────────
COMPETITION_MODES: Dict[str, dict] = {
    "1v1": {
        "archetypes": ["A", "C"],
        "desc":        "A vs C",
    },
    "3-Nation": {
        "archetypes": ["A", "B", "C"],
        "desc":        "A, B, C",
    },
    "4-Nation": {
        "archetypes": ["A", "B", "C", "D"],
        "desc":        "A, B, C, D",
    },
    "5-Nation": {
        "archetypes": ["A", "B", "C", "D", "E"],
        "desc":        "A, B, C, D, E",
    },
    "1v3": {
        "archetypes": ["C", "A", "A", "A"],
        "desc":        "C vs A×3",
    },
}


def build_competition_civs(mode_name: str, overrides: dict = None) -> List[Civilization]:
    """
    根据竞争模式构建文明列表。
    overrides: {civ_name: {"geo": {...}, "res": {...}, "strategy": str}}
    覆盖指定文明的默认参数，其余保持原型默认值。
    """
    mode = COMPETITION_MODES[mode_name]
    archetype_list = mode["archetypes"]
    overrides = overrides or {}

    name_count: Dict[str, int] = {}
    for a in archetype_list:
        name_count[a] = name_count.get(a, 0) + 1

    civs = []
    name_used: Dict[str, int] = {}
    for arch_name in archetype_list:
        cfg = _CONFIGS[arch_name]
        if name_count[arch_name] > 1:
            name_used[arch_name] = name_used.get(arch_name, 0) + 1
            idx = name_used[arch_name]
            civ_name = f"{arch_name}·{chr(64 + idx)}"
            color = _adjust_color(cfg["color"], idx)
        else:
            civ_name = arch_name
            color = cfg["color"]

        ov = overrides.get(civ_name, {})
        geo_params = {**cfg["geo"],  **ov.get("geo", {})}
        res_params = {**cfg["res"],  **ov.get("res", {})}
        strategy   = ov.get("strategy", cfg["strategy"])

        civs.append(Civilization(
            name=civ_name,
            geography=Geography(**geo_params),
            resources=Resources(**res_params),
            strategy_name=strategy,
            color=color,
            lat=0.0, lon=0.0,
        ))
    return civs


def default_params(mode_name: str) -> dict:
    """返回当前模式下各文明的默认参数，供 UI 初始化用。"""
    civs = build_competition_civs(mode_name)
    result = {}
    for civ in civs:
        arch = civ.name.split("·")[0]
        cfg  = _CONFIGS[arch]
        result[civ.name] = {
            "geo":      dict(cfg["geo"]),
            "res":      dict(cfg["res"]),
            "strategy": cfg["strategy"],
            "color":    civ.color,
        }
    return result


def get_archetype_name(civ_name: str) -> str:
    """从文明名（可能带后缀）中提取原型名。"""
    return civ_name.split("·")[0]


def archetype_descs() -> Dict[str, str]:
    return {name: cfg["desc"] for name, cfg in _CONFIGS.items()}


def compute_trade_matrix(mode_name: str) -> "pd.DataFrame":
    """计算竞争模式内各文明之间的资源互补度（贸易潜力代理指标）。"""
    import pandas as pd
    civs = build_competition_civs(mode_name)
    rows = []
    for ci in civs:
        for cj in civs:
            if ci.name == cj.name:
                comp = 0.0
            else:
                ri = [ci.resources.food, ci.resources.metal, ci.resources.wood, ci.resources.luxury]
                rj = [cj.resources.food, cj.resources.metal, cj.resources.wood, cj.resources.luxury]
                comp = round(sum(abs(a - b) for a, b in zip(ri, rj)) / 8.0, 3)
            rows.append({"出口方": ci.name, "进口方": cj.name, "互补度": comp})
    return pd.DataFrame(rows)


def _adjust_color(hex_color: str, idx: int) -> str:
    """对重复原型的副本调整色调（深/浅），让它们在图上可区分。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if idx == 2:   # 加深
        r, g, b = int(r * 0.65), int(g * 0.65), int(b * 0.65)
    elif idx == 3:  # 变浅
        r = int(r + (255 - r) * 0.45)
        g = int(g + (255 - g) * 0.45)
        b = int(b + (255 - b) * 0.45)
    return f"#{r:02x}{g:02x}{b:02x}"
