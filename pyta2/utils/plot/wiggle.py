#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的wiggle绘图模块

这个模块提供了优化的地震数据摆动图绘制功能，包括：
1. 基本摆动图绘制
2. 自定义颜色和样式
3. 批量绘图
4. 性能优化
5. 错误处理
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings
from functools import lru_cache
import time

__all__ = ['insert_zeros', 'wiggle_input_check', 'wiggle', 'optimized_wiggle', 
           'batch_wiggle_plot', 'create_demo_data', 'plot_wiggle_comparison', 
           'performance_test']


def insert_zeros(trace: np.ndarray, tt: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    在数据轨迹和tt向量中插入零位置，基于线性拟合
    
    Parameters:
    -----------
    trace : np.ndarray
        数据轨迹
    tt : np.ndarray, optional
        时间向量，如果为None则自动生成
        
    Returns:
    --------
    tuple
        (trace_zi, tt_zi) - 插入零后的轨迹和时间向量
    """
    if tt is None:
        tt = np.arange(len(trace))

    # 找到符号变化的点
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    
    # 如果没有符号变化，直接返回原数据
    if len(zc_idx) == 0:
        return trace.copy(), tt.copy()
    
    # 计算零交叉点
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    
    # 避免除零错误
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = (y2 - y1) / (x2 - x1)
        # 过滤掉无效的零交叉点
        valid_mask = np.isfinite(a) & (np.abs(a) > 1e-10)
        if not np.any(valid_mask):
            return trace.copy(), tt.copy()
            
        zc_idx = zc_idx[valid_mask]
        x1 = x1[valid_mask]
        y1 = y1[valid_mask]
        a = a[valid_mask]
        tt_zero = x1 - y1 / a

    # 分割tt和trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0].copy()
    trace_zi = trace_split[0].copy()

    # 插入零值
    for i in range(len(tt_zero)):
        tt_zi = np.hstack((tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack((trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data: np.ndarray, 
                      tt: Optional[np.ndarray], 
                      xx: Optional[np.ndarray], 
                      sf: float, 
                      verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    为wiggle()和traces()函数检查输入参数的辅助函数
    
    Parameters:
    -----------
    data : np.ndarray
        2D数据数组
    tt : np.ndarray, optional
        时间向量
    xx : np.ndarray, optional
        空间向量
    sf : float
        拉伸因子
    verbose : bool, default=False
        是否显示详细信息
        
    Returns:
    --------
    tuple
        (data, tt, xx, ts) - 处理后的数据和参数
    """
    # 输入验证
    if not isinstance(verbose, bool):
        raise TypeError("verbose必须是一个布尔值")

    # 数据验证
    if not isinstance(data, np.ndarray):
        raise TypeError("data必须是一个numpy数组")
    
    if data.ndim != 2:
        raise ValueError("data必须是一个2D数组")
    
    if data.size == 0:
        raise ValueError("data不能为空")

    # 时间向量验证
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt自动生成:", tt)
    else:
        if not isinstance(tt, np.ndarray):
            raise TypeError("tt必须是一个numpy数组")
        if tt.ndim != 1:
            raise ValueError("tt必须是一个1D数组")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt的长度必须与data的行数相同")

    # 空间向量验证
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx自动生成:", xx)
    else:
        if not isinstance(xx, np.ndarray):
            raise TypeError("xx必须是一个numpy数组")
        if xx.ndim != 1:
            raise ValueError("xx必须是一个1D数组")
        if xx.shape[0] != data.shape[1]:
            raise ValueError("xx的长度必须与data的列数相同")
        if verbose:
            print("xx:", xx)

    # 拉伸因子验证
    if not isinstance(sf, (int, float)):
        raise TypeError("拉伸因子(sf)必须是一个数字")
    if sf <= 0:
        raise ValueError("拉伸因子(sf)必须大于0")

    # 计算轨迹水平间距
    if len(xx) > 1:
        ts = np.min(np.diff(xx))
    else:
        ts = 1.0

    # 数据标准化和缩放
    data_std = np.std(data, axis=0)
    data_max_std = np.max(data_std)
    
    if data_max_std > 0:
        data_scaled = data / data_max_std * ts * sf
    else:
        data_scaled = data.copy()

    return data_scaled, tt, xx, ts


def wiggle(ax: plt.Axes, 
           data: np.ndarray, 
           tt: Optional[np.ndarray] = None, 
           xx: Optional[np.ndarray] = None, 
           color: str = 'k', 
           line_color: str = 'k', 
           sf: float = 0.15, 
           verbose: bool = False,
           alpha: float = 0.8,
           linewidth: float = 0.5,
           fill_positive: bool = True,
           fill_negative: bool = False,
           negative_color: Optional[str] = None) -> None:
    """
    绘制地震数据的摆动图
    
    这是一个用于可视化地震数据的经典方法，通过填充正振幅区域和绘制轨迹线来显示数据。
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        绘图轴对象
    data : np.ndarray
        2D地震数据数组，行为时间，列为轨迹
    tt : np.ndarray, optional
        时间向量，如果为None则自动生成
    xx : np.ndarray, optional
        空间向量（轨迹位置），如果为None则自动生成
    color : str, default='k'
        正振幅填充颜色
    line_color : str, default='k'
        轨迹线颜色
    sf : float, default=0.15
        拉伸因子，控制轨迹的振幅缩放
    verbose : bool, default=False
        是否显示详细信息
    alpha : float, default=0.8
        填充透明度
    linewidth : float, default=0.5
        线条宽度
    fill_positive : bool, default=True
        是否填充正振幅区域
    fill_negative : bool, default=False
        是否填充负振幅区域
    negative_color : str, optional
        负振幅填充颜色，如果为None则使用color
        
    Examples:
    ---------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> 
    >>> # 创建示例数据
    >>> data = np.random.randn(100, 30)
    >>> 
    >>> # 基本用法
    >>> fig, ax = plt.subplots(figsize=(12, 8))
    >>> wiggle(ax, data)
    >>> plt.show()
    >>> 
    >>> # 自定义颜色和参数
    >>> fig, ax = plt.subplots(figsize=(12, 8))
    >>> wiggle(ax, data, color='blue', line_color='red', sf=0.2, alpha=0.6)
    >>> plt.show()
    """
    
    # 输入检查
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)
    
    # 设置负振幅颜色
    if negative_color is None:
        negative_color = color
    
    # 获取轨迹数量
    n_traces = data.shape[1]
    
    if verbose:
        print(f"绘制 {n_traces} 条轨迹")
    
    # 绘制每条轨迹
    for i in range(n_traces):
        trace = data[:, i]
        offset = xx[i]
        
        if verbose and i % 10 == 0:  # 每10条轨迹打印一次
            print(f"处理轨迹 {i+1}/{n_traces}, 偏移: {offset:.2f}")
        
        # 插入零值以获得更好的填充效果
        trace_zi, tt_zi = insert_zeros(trace, tt)
        
        # 绘制轨迹线
        ax.plot(trace_zi + offset, tt_zi, color=line_color, linewidth=linewidth)
        
        # 填充正振幅区域
        if fill_positive:
            ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                           where=trace_zi >= 0,
                           facecolor=color, alpha=alpha, interpolate=True)
        
        # 填充负振幅区域
        if fill_negative:
            ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                           where=trace_zi < 0,
                           facecolor=negative_color, alpha=alpha, interpolate=True)
    
    # 设置轴范围和方向
    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()  # 地震数据通常时间轴向下
    
    # 设置标签
    ax.set_xlabel('轨迹位置')
    ax.set_ylabel('时间')
    ax.set_title('地震数据摆动图')


def create_demo_data(n_time: int = 100, n_traces: int = 30, 
                    add_signal: bool = True, noise_level: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建演示用的地震数据
    
    Parameters:
    -----------
    n_time : int, default=100
        时间采样点数
    n_traces : int, default=30
        轨迹数量
    add_signal : bool, default=True
        是否添加模拟信号
    noise_level : float, default=0.5
        噪声水平
        
    Returns:
    --------
    tuple
        (data, tt, xx) - 数据、时间向量、空间向量
    """
    # 生成时间向量
    tt = np.linspace(0, 2.0, n_time)
    
    # 生成空间向量
    xx = np.linspace(0, 100, n_traces)
    
    # 生成基础噪声数据
    data = np.random.randn(n_time, n_traces) * noise_level
    
    if add_signal:
        # 添加一些模拟的地震信号
        for i, x in enumerate(xx):
            # 添加不同深度的反射层
            for depth in [0.3, 0.8, 1.2]:
                # 计算到达时间
                t_arrival = depth + x * 0.01  # 简单的线性关系
                if t_arrival < tt[-1]:
                    # 找到最近的时间点
                    idx = np.argmin(np.abs(tt - t_arrival))
                    # 添加高斯脉冲
                    pulse = np.exp(-((tt - t_arrival) / 0.05) ** 2)
                    data[:, i] += pulse * (1 + 0.3 * np.sin(x * 0.1))
    
    return data, tt, xx


def plot_wiggle_comparison(data: np.ndarray, 
                          tt: Optional[np.ndarray] = None, 
                          xx: Optional[np.ndarray] = None,
                          figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    创建摆动图的对比展示
    
    Parameters:
    -----------
    data : np.ndarray
        地震数据
    tt : np.ndarray, optional
        时间向量
    xx : np.ndarray, optional
        空间向量
    figsize : tuple, default=(16, 10)
        图形尺寸
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 基本摆动图
    wiggle(axes[0, 0], data, tt, xx, color='blue', line_color='black', 
           sf=0.15, alpha=0.7)
    axes[0, 0].set_title('基本摆动图', fontsize=12, fontweight='bold')
    
    # 双色摆动图
    wiggle(axes[0, 1], data, tt, xx, color='red', line_color='black', 
           sf=0.2, alpha=0.6, fill_negative=True, negative_color='blue')
    axes[0, 1].set_title('双色摆动图', fontsize=12, fontweight='bold')
    
    # 高对比度摆动图
    wiggle(axes[1, 0], data, tt, xx, color='green', line_color='darkgreen', 
           sf=0.3, alpha=0.8, linewidth=1.0)
    axes[1, 0].set_title('高对比度摆动图', fontsize=12, fontweight='bold')
    
    # 仅线条模式
    wiggle(axes[1, 1], data, tt, xx, color='purple', line_color='purple', 
           sf=0.2, fill_positive=False, linewidth=1.5)
    axes[1, 1].set_title('仅线条模式', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def batch_wiggle_plot(data_list: List[np.ndarray], 
                     titles: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (16, 12),
                     **kwargs) -> None:
    """
    批量绘制多个摆动图
    
    Parameters:
    -----------
    data_list : List[np.ndarray]
        数据列表
    titles : List[str], optional
        标题列表
    figsize : tuple, default=(16, 12)
        图形尺寸
    **kwargs
        传递给wiggle函数的其他参数
    """
    n_plots = len(data_list)
    if n_plots == 0:
        return
    
    # 计算子图布局
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, data in enumerate(data_list):
        if i >= len(axes):
            break
            
        ax = axes[i]
        wiggle(ax, data, **kwargs)
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'数据 {i+1}', fontsize=12, fontweight='bold')
    
    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 创建演示数据
    print("创建演示数据...")
    data, tt, xx = create_demo_data(n_time=150, n_traces=40, add_signal=True, noise_level=0.3)
    
    print(f"数据形状: {data.shape}")
    print(f"时间范围: {tt[0]:.2f} - {tt[-1]:.2f}")
    print(f"空间范围: {xx[0]:.2f} - {xx[-1]:.2f}")
    
    # 创建对比图
    print("生成对比图...")
    plot_wiggle_comparison(data, tt, xx)
    
    # 单个图示例
    print("生成单个图示例...")
    fig, ax = plt.subplots(figsize=(12, 8))
    wiggle(ax, data, tt, xx, color='blue', line_color='black', 
           sf=0.2, alpha=0.7, fill_negative=True, negative_color='red')
    ax.set_title('优化后的摆动图示例', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
