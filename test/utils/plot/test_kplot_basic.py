from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from pyta2.utils.plot.kplot import kplot
# import chineseize_matplotlib

def generate_sample_klinedata(n_points: int = 200, 
                              base_price: float = 100.0,
                              volatility: float = 0.02,
                              trend: float = 0.001) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成模拟的K线数据用于测试
    
    Parameters:
    -----------
    n_points : int, default=200
        生成的数据点数量
    base_price : float, default=100.0
        基础价格
    volatility : float, default=0.02
        价格波动率
    trend : float, default=0.001
        价格趋势（正值表示上涨趋势）
        
    Returns:
    --------
    tuple
        (opens, highs, lows, closes, volumes, dates) - OHLCV数据和日期
    """
    np.random.seed(42)  # 设置随机种子确保结果可重现
    
    # 生成价格序列（随机游走 + 趋势）
    price_changes = np.random.normal(trend, volatility, n_points)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # 生成OHLC数据
    opens = prices[:-1]  # 开盘价
    closes = prices[1:]  # 收盘价
    
    # 生成高低价（在开盘和收盘价基础上添加随机波动）
    high_volatility = np.random.uniform(0.005, 0.02, n_points-1)
    low_volatility = np.random.uniform(0.005, 0.02, n_points-1)
    
    highs = np.maximum(opens, closes) * (1 + high_volatility)
    lows = np.minimum(opens, closes) * (1 - low_volatility)
    
    # 生成成交量（与价格变化相关）
    price_change_ratio = np.abs(closes - opens) / opens
    base_volume = np.random.uniform(1000, 5000, n_points-1)
    volumes = base_volume * (1 + price_change_ratio * 2)
    
    # 生成日期序列
    dates = np.arange(n_points-1)
    
    return opens, highs, lows, closes, volumes, dates


if __name__ == "__main__":
    """
    测试函数 - 使用手动生成的K线数据演示绘图功能
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("🚀 开始K线图绘制测试...")
    
    try:
        # 生成模拟数据
        print("📊 正在生成模拟K线数据...")
        n_points = 200
        opens, highs, lows, closes, volumes, dates = generate_sample_klinedata(
            n_points=n_points, 
            base_price=100.0, 
            volatility=0.02, 
            trend=0.001
        )
        
        print(f"✅ 成功生成 {len(opens)} 条K线数据")
        print(f"   价格范围: {np.min(lows):.2f} - {np.max(highs):.2f}")
        print(f"   成交量范围: {np.min(volumes):.0f} - {np.max(volumes):.0f}")
        
        # 创建图形
        print("🎨 正在创建图形...")
        fig, ax = plt.subplots(figsize=(16, 8))
        volume_ax = ax.twinx()
        
        # 绘制K线图
        print("📈 正在绘制K线图...")
        kplot(ax, opens, highs, lows, closes, volumes=volumes, 
              volume_ax=volume_ax,
              width=0.8, alpha=0.9, colorup='#2E8B57', colordown='#DC143C',
              grid=True, rotate_date=True)
        
        # 设置标题和标签
        ax.set_title('K线图测试 - 模拟数据', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('价格 (元)', fontsize=12, fontweight='bold')
        volume_ax.set_ylabel('成交量', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        
        # 添加统计信息
        price_change = (closes[-1] - opens[0]) / opens[0] * 100
        max_price = np.max(highs)
        min_price = np.min(lows)
        total_volume = np.sum(volumes)
        
        stats_text = f'价格变化: {price_change:+.2f}% | 最高: {max_price:.2f} | 最低: {min_price:.2f} | 总成交量: {total_volume:,.0f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 优化布局
        plt.tight_layout()
        
        print("✅ K线图绘制完成！")
        print("📊 图表统计信息:")
        print(f"   - 数据点数: {len(opens)}")
        print(f"   - 价格变化: {price_change:+.2f}%")
        print(f"   - 价格范围: {min_price:.2f} - {max_price:.2f}")
        print(f"   - 总成交量: {total_volume:,.0f}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 绘制K线图时发生错误: {e}")
        import traceback
        traceback.print_exc()
