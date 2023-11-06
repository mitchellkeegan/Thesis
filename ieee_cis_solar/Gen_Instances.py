import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from statsmodels.tsa.seasonal import STL
import numpy as np

instance_dir = 'Instances/smallArtificial'

# Load in forecasted base demand and solar supply (should be sitting in the root instance folder)
forecast_csv = os.path.join('Instances','Forecasts-sample.csv')

# Load in pricing data
price_df = pd.read_csv(os.path.join('Instances','PRICE_AND_DEMAND_202011_VIC1.csv'))

price_30m = price_df['RRP'].to_list()

price = []

for rrp in price_30m:
    price.append(rrp)
    price.append(rrp)

price = np.asarray(price)

T = range(2880)
D_o = list(range(30))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(T,price,'black',label='Base Time Series')
ax.set_xticks([4*24*d for d in D_o],[d for d in D_o])
ax.xaxis.set_minor_locator(FixedLocator(range(0,len(T),4)))
ax.set_xlabel('Day')
ax.set_ylabel('Grid price ($/MWh')
ax.set_title('Artificial Grid Power Prices')
ax.legend()

res = STL(price,period=2880//30).fit()
trend,seasonal,resids = res.trend, res.seasonal, res.resid

# ax_trend = fig.add_subplot(4,1,2,sharex=ax)
# ax_trend.plot(T,res.trend)
#
# ax_seasonal = fig.add_subplot(4,1,3,sharex=ax)
# ax_seasonal.plot(T,res.seasonal)
#
# ax_resid = fig.add_subplot(4,1,4,sharex=ax)
# ax_resid.plot(T,res.resid)


# Bootstrap a sample from the data
block_size = 30
assert 2880 % block_size == 0

num_blocks = 2880//30
n_samples = 100
# bootstrapped_series = np.zeros((n_samples,2880))
bootstrapped_series = (trend+seasonal) * np.ones((n_samples,2880))
rng = np.random.default_rng()
for n in range(n_samples):

    block_starts = rng.choice([block_size*i for i in range(num_blocks)],size=num_blocks)
    for i,idx in enumerate(block_starts):
        bootstrapped_series[n,i*block_size:(i+1)*block_size] += resids[idx:idx+block_size]

    if n < 3:
        ax.plot(T, bootstrapped_series[n,:],'--')

# bootstrapped_series_pd = pd.DataFrame(bootstrapped_series)
# bootstrapped_series_pd.to_csv(path_or_buf=os.path.join(instance_dir,'Backup Forecasts.csv'))
#
# stored_forecasts = pd.read_csv(os.path.join(instance_dir,'Backup Forecasts.csv')).to_numpy()[:,1:]

# with open(os.path.join(instance_dir,'Instance Generation Info.txt'),'w') as f:
#     f.write(f'{n_samples}\n')

plt.savefig('Artificial Grid Price Data.eps')
plt.show()