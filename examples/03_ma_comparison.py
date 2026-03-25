#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyta2 trend/ma 移动均线使用示例
=================================

展示 3 种不同的调用模式：
  Mode 1 - Batch 函数批量处理（最常用，适合回测）
  Mode 2 - rolling 指标对象，逐步滚动（适合实时流式更新）
  Mode 3 - 指标对象单步 rolling，缓存输出拿 DataFrame（适合策略开发）

对比均线类型：SMA / EMA / WMA / DEMA / TEMA / HMA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------- pyta2 imports ----------
from pyta2.trend.ma import SMA, EMA, WMA, HMA, DEMA, TEMA   # batch 函数
from pyta2.trend.ma import rSMA, rEMA, rWMA, rHMA, rDEMA, rTEMA  # rolling 指标
from pyta2.utils.plot import kplot_df                         # K线绘图

# ---------- data imports ----------
from fintest.crypto.api import get_data

# ============================================================
# 加载 OHLCV 数据
# ============================================================
df = get_data('kline-1m-alpha')
# 转换时间戳为 datetime (open_time 是毫秒)
if 'open_time' in df.columns:
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')

# 确保数据量不要太大，取前 120 根
df = df.head(120).copy()
N = len(df)
close = df['close'].values



PERIOD = 10   # 均线周期（所有均线保持一致，方便对比）
ma_colors = {
    'SMA':  'steelblue',
    'EMA':  'orange',
    'WMA':  'green',
    'DEMA': 'purple',
    'TEMA': 'red',
    'HMA':  'brown',
}

# ============================================================
# Mode 1: Batch 函数 — 一次性计算整列数据
#          适合静态数据集 / 向量化回测
# ============================================================
print("=" * 50)
print("Mode 1: Batch 函数")
ma_batch = {
    'SMA':  SMA(close, PERIOD),
    'EMA':  EMA(close, PERIOD),
    'WMA':  WMA(close, PERIOD),
    'DEMA': DEMA(close, PERIOD),
    'TEMA': TEMA(close, PERIOD),
    'HMA':  HMA(close, PERIOD),
}
for name, result in ma_batch.items():
    # 找到第一个非 NaN 的位置
    first_valid = np.where(~np.isnan(result))[0][0]
    print(f"  {name}({PERIOD}): 首个有效值 index={first_valid}, val={result[first_valid]:.4f}")

# ============================================================
# Mode 2: rolling 指标对象逐步更新
#          适合实时流式数据，每次喂入最新的历史切片
# ============================================================
print("\nMode 2: rolling 指标对象（逐步滚动）")
indicators_rolling = {
    'SMA':  rSMA(PERIOD),
    'EMA':  rEMA(PERIOD),
    'WMA':  rWMA(PERIOD),
    'DEMA': rDEMA(PERIOD),
    'TEMA': rTEMA(PERIOD),
    # HMA n>1，此处使用 n=4 避免 n//2=0 的越界问题
    'HMA':  rHMA(4),
}
ma_rolling = {name: [] for name in indicators_rolling}
for i in range(N):
    for name, ind in indicators_rolling.items():
        val = ind.rolling(close[:i+1])
        ma_rolling[name].append(val)

# 转为 numpy array
ma_rolling = {name: np.array(vals) for name, vals in ma_rolling.items()}
print(f"  每种均线均完成 {N} 步逐步滚动。")
# 验证 Mode 1 与 Mode 2 结果一致（SMA 为例，对 Mode 2 使用相同 period）
sma_ok = np.allclose(ma_batch['SMA'], ma_rolling['SMA'], equal_nan=True)
print(f"  Mode1 vs Mode2 SMA 一致: {sma_ok}")

# ============================================================
# Mode 3: 指标对象携带输出缓存 → 直接导出 DataFrame
#          适合策略模块中需要随时拿历史缓存的场景
# ============================================================
print("\nMode 3: 带缓存的指标对象 → DataFrame 输出")
from pyta2.trend.ma.api import get_ma_class

# 构建一个带 buffer_size 的 EMA 指标（保留最近 20 条输出）
EMA_cached = rEMA(PERIOD, buffer_size=20)
for i in range(N):
    EMA_cached.rolling(close[:i+1])

# 导出最近 20 条结果
df_cache = EMA_cached.outputs.to_pandas()
print(f"  EMA({PERIOD}) 缓存了 {len(df_cache)} 条输出")
print(df_cache.tail(5).to_string(index=False))

# ============================================================
# 绘图：K 线图 + 6 条均线对比
# ============================================================
fig, ax = plt.subplots(figsize=(15, 7))
kplot_df(ax, df, show_volume=False, use_date=True,
         title=f'Moving Average Comparison  n={PERIOD}', ylabel='Price')

x = df['date']
# 使用 Mode 1（Batch）的结果绘制均线（数量最多也最清晰）
for name, vals in ma_batch.items():
    ax.plot(x, vals, label=name, color=ma_colors[name], linewidth=1.4, alpha=0.85)

ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()

# ============================================================
# 绘图：均线延迟对比（去掉 K 线，只看各条均线与 close 的差距）
# ============================================================
fig2, ax2 = plt.subplots(figsize=(15, 5))
ax2.plot(x, close, label='Close',  color='black', linewidth=1.0, alpha=0.5)
for name, vals in ma_batch.items():
    ax2.plot(x, vals, label=name, color=ma_colors[name], linewidth=1.4, alpha=0.85)
ax2.set_title(f'MA Lag Comparison 图 n={PERIOD}  (close vs all MAs)')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(alpha=0.3)
plt.tight_layout()

# ============================================================
# 绘图：同一函数不同周期对比 (例如 EMA 5/10/20/60)
# ============================================================
periods = [10, 20, 60]
period_colors = ['orange', 'red', 'purple']

fig3, ax3 = plt.subplots(figsize=(15, 6))
kplot_df(ax3, df, show_volume=False, use_date=True,
         title='EMA Mult-Period Comparison (EMA 10/20/60)', ylabel='Price')

for p, color in zip(periods, period_colors):
    ema_vals = EMA(close, p)
    ax3.plot(x, ema_vals, label=f'EMA-{p}', color=color, linewidth=1.5, alpha=0.9)

ax3.legend(loc='upper left', fontsize=10)
ax3.grid(alpha=0.3)
plt.tight_layout()


print("\n绘图完成，请查看弹出的图形窗口。")
plt.show()
