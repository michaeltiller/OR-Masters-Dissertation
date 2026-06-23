#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:34:31 2026

@author: michael
"""

da_Prices_df = pd.read_csv('Pricing Data/da_hrl_lmps.csv')
rt_Prices_df = pd.read_csv('Pricing Data/rt_hrl_lmps.csv')





rt_Prices_df = rt_Prices_df[['datetime_beginning_ept', 'system_energy_price_rt']]

da_Prices_df = da_Prices_df[['datetime_beginning_ept', 'system_energy_price_da']]

prices  = pd.merge(rt_Prices_df, da_Prices_df, how ='left', on = 'datetime_beginning_ept')







da_Prices_df['datetime_beginning_ept'] = pd.to_datetime(da_Prices_df['datetime_beginning_ept'])

da_Prices_df['hour'] = da_Prices_df['datetime_beginning_ept'].dt.hour
da_Prices_df['Date'] = da_Prices_df['datetime_beginning_ept'].dt.date

pricesWide = pd.pivot_table(da_Prices_df, index = 'Date', columns = 'hour', values = 'system_energy_price_da', aggfunc = 'mean')
pricesWide = pricesWide.dropna()
pricesWide = pricesWide.to_numpy()

avgPriceHourly = pricesWide.mean(axis = 0)

avgPrice = avgPriceHourly.mean()


def calcBeta(pricesWide, avgPriceHourly, avgPrice):
    N = len(pricesWide)
    return 1/N * sum(np.sqrt( (sum( (pricesWide[i,k]  - (1/24) * sum(pricesWide[i,j] for j in range(24) ))**2 for k in range(24))) /  sum((avgPriceHourly[k] - avgPrice)**2 for k in range(24))   )    for i in range(N))

beta = calcBeta(pricesWide, avgPriceHourly, avgPrice)


estimated_price = [avgPrice + beta * (avgPriceHourly[k] - avgPrice) for k in range(24)]


def calcbetafromQuantiles(pricesWide, avgPriceHourly, avgPrice, quantile):

    daily_std = np.std(pricesWide, axis=1, ddof=0)
    
    q = np.quantile(daily_std, 0.85)
    
    avgPriceHourly = np.mean(pricesWide, axis=0)
    avgPrice = np.mean(pricesWide)
    
    denom = np.sqrt(
        (1/24) * np.sum((avgPriceHourly - avgPrice)**2)
    )
    
    return q / denom


# Calc intraday prices 


prices = prices.dropna()

prices['datetime_beginning_ept'] = pd.to_datetime(prices['datetime_beginning_ept'])

prices['hour'] = prices['datetime_beginning_ept'].dt.hour
prices['date'] = prices['datetime_beginning_ept'].dt.date


prices['diff'] = prices['system_energy_price_rt'] - prices['system_energy_price_da']

hourly_deviation = prices.groupby('hour')['diff'].agg('mean')

correction = hourly_deviation.mean()

delta_corrected = (
    hourly_deviation
    -
    correction
)


target_sigma = (
    prices
    .groupby("date")["system_energy_price_rt"]
    .std()
    .mean()
)

current_sigma = (estimated_price + delta_corrected).std()



gamma = target_sigma/current_sigma


idPriceEstimate = estimated_price + gamma * delta_corrected

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(1, 25)

ax.plot(x, idPriceEstimate, label='ID Price', color='#2563EB', linewidth=2, marker='o', markersize=4, zorder=5)
ax.plot(x, estimated_price, label='DA Price', color='#DC2626', linewidth=2, marker='s', markersize=4, linestyle='--', zorder=5)

ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('Price ($/MWh)', fontsize=12)
ax.set_title('Intraday vs Estimated Price — 24-Hour Horizon', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{h:02d}:00' for h in x], rotation=45, ha='right', fontsize=9)
ax.legend(frameon=True, fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()




 
    