import pyta2 as pyta
from pyta2.utils.plot import kplot, plt
from fintest.crypto.api import get_data
import pandas as pd

df = get_data('kline-1m-alpha')
print(df.head())

print(df['close'].values)
# df['sma10'] = pyta.SMA(df['close'].values, 10)
df['sma10-2'] = pyta.SMA(df['close'], 10)

fig, ax = plt.subplots(figsize=(10, 5))
kplot(ax, df['open'], df['high'], df['low'], df['close'], 
      title='Kline-1m-Alpha', 
      ylabel='Price',
      xlabel='Time')

ax.plot(df['sma10'], label='SMA-10')

plt.show()