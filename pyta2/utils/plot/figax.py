import matplotlib.pyplot as plt
from typing import Tuple, List, Union

__all__ = ['get_figax', 'plt']

def get_figax(n_tx: int, 
              rows: int = 111, 
              figsize: Tuple[float, float] = (16, 8), 
              offset: float = 28,
              left: float = 0.03, 
              right: float = 0.97, 
              top: float = 0.97, 
              bottom: float = 0.03,
              tx_colors: List[str] = None) -> Tuple[plt.Figure, plt.Axes, ...]:
    """
    创建带有多个y轴的图表
    
    Parameters:
    -----------
    n_tx : int
        需要创建的twinx轴数量
    rows : int, default=111
        子图布局参数
    figsize : tuple, default=(16, 8)
        图形尺寸
    offset : float, default=28
        右侧轴偏移量
    left, right, top, bottom : float
        图形边距参数
    tx_colors : list, optional
        轴颜色列表，如果提供会自动设置轴颜色
        
    Returns:
    --------
    tuple
        (fig, ax, *tx_list) - 图形对象、主轴和所有twinx轴
    """
    if tx_colors is None:
        tx_colors = ['r', 'g', 'b', 'm', 'c']
    
    # 创建图形和主轴
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax = fig.add_subplot(rows)

    # 创建twinx轴列表
    tx_list = []
    for i in range(n_tx):
        tx = ax.twinx()
        
        # 设置轴位置偏移
        if i >= 1:
            tx.spines['right'].set_position(('outward', -offset * i))
        
        # 设置轴颜色
        if i < len(tx_colors):
            tx.spines['right'].set_color(tx_colors[i])
            tx.tick_params(axis='y', colors=tx_colors[i])
        
        tx_list.append(tx)

    return fig, ax, *tx_list