import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Union, Callable, Any, Tuple
from pandas import DataFrame
from .mpl_finance import candlestick2_ohlc, volume_overlay


def kplot_df(ax: plt.Axes, 
              df: DataFrame, 
              show_volume: bool = False, 
              use_date: bool = False,
              key_date: Union[str, Callable] = 'date',
              key_open: Union[str, Callable] = 'open', 
              key_high: Union[str, Callable] = 'high',
              key_low: Union[str, Callable] = 'low', 
              key_close: Union[str, Callable] = 'close',
              key_volume: Union[str, Callable] = 'volume',
              **kwargs) -> Optional[plt.Axes]:
    """
    从DataFrame绘制K线图
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        绘图轴对象
    df : pandas.DataFrame
        包含OHLC数据的DataFrame
    show_volume : bool, default=False
        是否显示成交量
    use_date : bool, default=False
        是否使用日期作为x轴
    key_date : str or callable, default='date'
        日期列的键名或提取函数
    key_open : str or callable, default='open'
        开盘价列的键名或提取函数
    key_high : str or callable, default='high'
        最高价列的键名或提取函数
    key_low : str or callable, default='low'
        最低价列的键名或提取函数
    key_close : str or callable, default='close'
        收盘价列的键名或提取函数
    key_volume : str or callable, default='volume'
        成交量列的键名或提取函数
    **kwargs
        传递给kplot函数的其他参数
        
    Returns:
    --------
    matplotlib.axes.Axes or None
        成交量轴对象（如果显示成交量）或None
    """
    def get_key_values(key: Union[str, Callable]) -> Optional[np.ndarray]: 
        """从DataFrame中提取指定列的数据"""
        if callable(key):
            return key(df)
        elif isinstance(key, str):
            if key in df.columns:
                return df[key].values
            else:
                logging.warning(f'Column "{key}" not found in DataFrame')
                return None
        else:
            raise ValueError(f'Unsupported parameter type for key: {type(key)}')
    
    # 提取OHLC数据
    opens = get_key_values(key_open)
    highs = get_key_values(key_high)
    lows = get_key_values(key_low)
    closes = get_key_values(key_close)
    
    # 验证必要数据是否存在
    if any(data is None for data in [opens, highs, lows, closes]):
        raise ValueError("Missing required OHLC data columns")
    
    # 提取日期数据
    dates = get_key_values(key_date) if use_date else None

    # 提取成交量数据
    volumes = get_key_values(key_volume) if show_volume else None
    
    # 调用核心绘图函数
    return kplot(ax, opens, highs, lows, closes, volumes=volumes, dates=dates, **kwargs)

def ensure_numpy(vec: Union[np.ndarray, Any], dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    确保输入转换为numpy数组
    
    Parameters:
    -----------
    vec : array-like
        输入数据
    dtype : numpy.dtype, optional
        目标数据类型
        
    Returns:
    --------
    numpy.ndarray
        转换后的numpy数组
    """
    # 如果是pandas对象，提取values
    if hasattr(vec, 'values'):
        vec = vec.values
    
    # 转换为numpy数组
    if dtype is not None:
        return np.array(vec, dtype=dtype)
    return np.array(vec)

def kplot(ax: plt.Axes, 
          opens: Union[np.ndarray, list], 
          highs: Union[np.ndarray, list], 
          lows: Union[np.ndarray, list], 
          closes: Union[np.ndarray, list], 
          volumes: Optional[Union[np.ndarray, list]] = None, 
          dates: Optional[Union[np.ndarray, list]] = None, 
          volume_ax: Optional[plt.Axes] = None,
          width: float = 0.9, 
          alpha: float = 0.9, 
          colorup: str = 'green', 
          colordown: str = 'red',
          date_formatter: Optional[str] = None, 
          rotate_date: bool = True,
          volume_yticklabel_off: bool = False, 
          title: Optional[str] = None,
          ylabel: Optional[str] = None,
          xlabel: Optional[str] = None,
          yscale: Optional[str] = None,
          xscale: Optional[str] = None,
          grid: bool = False) -> Optional[plt.Axes]:
    """
    绘制K线图（蜡烛图）
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        主绘图轴对象
    opens : array-like
        开盘价数据
    highs : array-like
        最高价数据
    lows : array-like
        最低价数据
    closes : array-like
        收盘价数据
    volumes : array-like, optional
        成交量数据
    dates : array-like, optional
        日期数据
    volume_ax : matplotlib.axes.Axes, optional
        成交量轴对象，如果为None则自动创建
    width : float, default=0.9
        蜡烛图宽度
    alpha : float, default=0.9
        透明度
    colorup : str, default='green'
        上涨颜色
    colordown : str, default='red'
        下跌颜色
    date_formatter : str, optional
        日期格式化字符串
    rotate_date : bool, default=True
        是否旋转日期标签
    volume_yticklabel_off : bool, default=False
        是否隐藏成交量轴标签
    grid : bool, default=False
        是否显示网格
        
    Returns:
    --------
    matplotlib.axes.Axes or None
        成交量轴对象（如果显示成交量）或None
    """
    # 数据长度验证
    data_lengths = [len(opens), len(highs), len(lows), len(closes)]
    if not all(length == data_lengths[0] for length in data_lengths):
        raise ValueError("OHLC data must have the same length")
    
    if len(opens) == 0:
        logging.info('kplot: Empty Data')
        return None

    # 数据预处理
    _opens = ensure_numpy(opens, dtype=np.float64)
    _highs = ensure_numpy(highs, dtype=np.float64)
    _lows = ensure_numpy(lows, dtype=np.float64)
    _closes = ensure_numpy(closes, dtype=np.float64)
    
    if dates is not None:
        dates = ensure_numpy(dates)
    
    # 绘制K线图
    candlestick2_ohlc(ax, _opens, _highs, _lows, _closes, dates=dates,
                     width=width, alpha=alpha, colorup=colorup, colordown=colordown)
    
    # 设置价格轴范围
    _max = np.nanmax(_highs) 
    _min = np.nanmin(_lows) 
    _span = _max - _min
    ylim = [_min - _span * 0.25, _max + _span * 0.03]
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(grid)
    
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    # 处理成交量
    if volumes is not None:
        if volume_ax is None:
            volume_ax = ax.twinx()
        
        _volumes = ensure_numpy(volumes, dtype=np.float64)
        max_volumes = np.max(_volumes)
        _top = max_volumes * (1 + 0.03 + 0.25) / 0.25
        
        # 绘制成交量
        volume_ax.axhline(y=max_volumes, color='k', lw=0.5)
        volume_overlay(volume_ax, _opens, _closes, _volumes, dates=dates, 
                      width=width, alpha=alpha,
                      colorup=colorup, colordown=colordown)
        volume_ax.set_ylim(0, _top)
        volume_ax.spines['right'].set_visible(False)
        
        if volume_yticklabel_off:
            plt.setp(volume_ax, yticklabels=[])

    # 处理日期格式化
    if dates is not None:
        if date_formatter is None:
            date_formatter = '%Y-%m-%d %H:%M'
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
        if rotate_date:
            plt.gcf().autofmt_xdate()

    return volume_ax

