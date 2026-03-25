import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fintest.crypto.binance import api
from pyta2.utils.plot import kplot_df, get_figax
from pyta2.stats.atr import ATR, DecATR
from pyta2.structure.zigzag import ZigZag, ZigZag_HL, ZigZag_Unit, ZigZag_HL_Unit

def plot_zigzag():
    data = api.get_future_klines_demo1()
    df = pd.DataFrame(data[:600])
    float_columns = ['open', 'high', 'low', 'close']
    for col in float_columns:
        df[col] = df[col].astype(float)
    df['date'] = pd.to_datetime(df['open_time']/1000)
    # 如果ATR-n太小，导致
    df['units'] = ATR(df['high'], df['low'], df['close'], 50)
    
    print(df.head())

    fig, ax, *tx = get_figax(2)
    kplot_df(ax, df, show_volume=True)
    
    delta = 0.005
    zigzag, meta_info = ZigZag(df['close'], delta, use_pct=True, return_type='dataframe', return_meta_info=True)
    zigzag_hl, meta_info_hl = ZigZag_HL(df['high'], df['low'], delta, use_pct=True, return_type='dataframe', return_meta_info=True)
    zigzag_unit, meta_info_unit = ZigZag_Unit(df['units'], df['close'], delta=3, return_type='dataframe', return_meta_info=True)
    zigzag_hl_unit, meta_info_hl_unit = ZigZag_HL_Unit(df['units'], df['high'], df['low'], df['close'], delta=3, return_type='dataframe', return_meta_info=True)
    
    print(zigzag)
    print(zigzag_hl)
    print(zigzag_unit)
    print(zigzag_hl_unit)
    
    def plot_zigzag(zigzag, meta_info):
        mask = zigzag['confirmed_at'] != -1
        zigzag = zigzag[mask]
        print(zigzag)
        
        Is = zigzag['confirmed_at'].values
        Ts = zigzag['searching_dir'].values
        Ys = df['close'].values[Is]
        
        ax.plot(Is+0.5, Ys, ls='-', marker='^', lw=1.2, label=f"{meta_info['full_name']}")
        for i, t in zip(Is, Ts):
            if t == -1:
                ax.plot(i+0.5, df['close'].iloc[i], marker='v', lw=1.2)
            else:
                ax.plot(i+0.5, df['close'].iloc[i], marker='^', lw=1.2)
    
    plot_zigzag(zigzag, meta_info)
    plot_zigzag(zigzag_hl, meta_info_hl)
    plot_zigzag(zigzag_unit, meta_info_unit)
    plot_zigzag(zigzag_hl_unit, meta_info_hl_unit)
    # tx[0].plot(zigzag, ls='--', marker='^', lw=0.7, label=f"zigzag {delta}")
    
    ax.legend(loc='upper left')
    tx[0].legend(loc='upper right')
    fig.tight_layout()


plot_zigzag()
plt.show()
