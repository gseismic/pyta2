import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyta_dev.utils.plot import kplot_df, get_figax
from fintest.crypto.binance import api
from pyta_dev.stats.atr import ATR, DecATR
from pyta_dev.structure.channel import Boll

def plot_boll():
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
    
    ub, mid, lb = Boll(df['close'], 20, 2)

    ax.plot(ub, ls='--', marker='^', lw=0.7, label=f"ub")
    ax.plot(mid, ls='--', marker='^', lw=0.7, label=f"mid")
    ax.plot(lb, ls='--', marker='^', lw=0.7, label=f"lb")
    
    ax.legend(loc='upper left')
    tx[0].legend(loc='upper right')
    fig.tight_layout()


plot_boll()
plt.show()
