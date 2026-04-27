# 运行说明

## 环境

已创建 conda 环境 `econ0302g`，Python 3.11，依赖已安装。

## 启动可视化仪表板（主要提交内容）

```bash
# 在 VSCode 终端中：
conda activate econ0302g
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

## 运行命令行实验（生成报告图表）

```bash
conda activate econ0302g

# 全部实验
python run_experiments.py

# 附加蒙特卡洛（20次）
python run_experiments.py --monte-carlo 20

# 关闭随机事件（纯决定论基准）
python run_experiments.py --no-events
```

图表保存在 `output/` 目录。

## 项目结构

```
ECON0302G_Research/
├── src/
│   ├── civilization.py   文明数据模型（地理、资源、技术）
│   ├── economy.py        经济模型（柯布-道格拉斯生产函数、贸易、人口）
│   ├── strategies.py     策略智能体（规则式 + Q-Learning）
│   ├── engine.py         模拟引擎（驱动时间轴、协调各模块）
│   └── events.py         历史随机事件系统（黑死病、技术突破等）
├── app.py                Streamlit 可视化仪表板（6个标签页）
├── run_experiments.py    命令行实验运行器（含蒙特卡洛）
└── requirements.txt
```
