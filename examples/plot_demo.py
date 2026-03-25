#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyta2 绘图工具演示脚本
展示如何使用 figax, kplot 和 wiggle 绘图功能。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 直接从安装好的 pyta2 包中导入
from pyta2.utils.plot import get_figax, kplot_df, wiggle
from pyta2.utils.plot.wiggle import create_demo_data

def demo_kplot():
    """演示 K 线图绘制"""
    print("\n--- 演示 K 线图 (KPlot) ---")
    
    # 1. 生成模拟的 OHLCV 数据
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    # 简单的随机漫步生成价格
    base_price = 100
    noise = np.random.randn(n).cumsum() * 0.5
    close = base_price + noise
    opens = close - np.random.randn(n) * 0.3
    high = np.maximum(opens, close) + np.random.rand(n) * 0.5
    low = np.minimum(opens, close) - np.random.rand(n) * 0.5
    volume = np.random.rand(n) * 1000000
    
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # 2. 绘制 (增加参考线演示)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 模拟一些参考线：比如最近的价格波动高低点
    h_lines = [df['high'].max(), df['low'].min()]
    # 比如在第 50 个数据点画垂直线
    v_lines = [df['date'].iloc[50]] if True else [50] 
    
    v_ax = kplot_df(ax, df, 
                     show_volume=True, 
                     use_date=True,
                     hlines=h_lines,
                     vlines=v_lines,
                     line_color='blue',
                     use_cursor=True,
                     title="pyta2 Financial Plot Demo (with Ref Lines & Cursor)",
                     ylabel="Price")
    
    if v_ax:
        v_ax.set_ylabel("Volume")
    
    plt.tight_layout()
    print("K线图绘制完成。若在 GUI 环境中运行，图形窗口应弹出。")
    # 如果在非 GUI 环境下需要保存，可以取消下行注释
    # plt.savefig('kplot_demo.png')

def demo_wiggle():
    """演示地震数据摆动图 (Wiggle)"""
    print("\n--- 演示地震数据摆动图 (Wiggle) ---")
    
    # 1. 使用 wiggle 模块内置的数据生成器
    data, tt, xx = create_demo_data(n_time=150, n_traces=30, add_signal=True, noise_level=0.3)
    
    # 2. 绘制 (展示多子图对比)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # 基本摆动图 (蓝色填充)
    wiggle(ax1, data, tt, xx, color='blue', line_color='black', alpha=0.7)
    ax1.set_title("Standard Wiggle (Positive Fill)")
    
    # 高级双色填充 (正负振幅不同颜色)
    wiggle(ax2, data, tt, xx, 
           color='red', line_color='black', 
           fill_negative=True, negative_color='blue', 
           sf=0.2, alpha=0.6)
    ax2.set_title("Dual-Color Wiggle (Optimized Zeros)")
    
    plt.tight_layout()
    print("Wiggle 绘图完成（已使用优化后的插值算法）。")

def demo_figax():
    """演示多轴(twinx)配置工具"""
    print("\n--- 演示多轴配置 (Figax) ---")
    
    # 创建带有 3 个独立右侧轴的图表
    # 返回: (fig, ax_main, tx1, tx2, tx3)
    res = get_figax(n_tx=3, tx_colors=['red', 'blue', 'green'], offset=40)
    fig, ax_main = res[0], res[1]
    other_axes = res[2:]
    
    # 生成各不相同量级的数据
    x = np.linspace(0, 10, 100)
    ax_main.plot(x, np.sin(x), 'k-', label='Main (sin)')
    ax_main.set_ylabel("Main Wave")
    
    other_axes[0].plot(x, x**2, 'r-', label='TX1 (x^2)')
    other_axes[0].set_ylabel("Squared Value")
    
    other_axes[1].plot(x, np.exp(x/2), 'b-', label='TX2 (exp)')
    other_axes[1].set_ylabel("Exp Value")
    
    other_axes[2].plot(x, np.log1p(x), 'g-', label='TX3 (log)')
    other_axes[2].set_ylabel("Log Value")
    
    ax_main.set_title("Multi-TwinX Axis Demo")
    ax_main.legend(loc='upper left')
    
    print("Figax 多轴图配置完成。")

if __name__ == "__main__":
    # 检查是否安装了中文字体库
    try:
        import chineseize_matplotlib
    except ImportError:
        print("提示: 未安装 chineseize_matplotlib，中文显示可能受限（pip install chineseize-matplotlib）。")

    demo_kplot()
    demo_wiggle()
    demo_figax()
    
    print("\n演示程序运行完毕。")
    # 仅当检测到后端允许时才调用 show
    if plt.get_backend().lower() != 'agg':
        plt.show()
