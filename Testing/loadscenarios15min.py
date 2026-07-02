#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:39:06 2026

@author: michael
"""


std_load = 0.06
n_scenarios = 10

# df = pd.read_csv('Pricing data/c391110f-760f-4133-bf51-f1657cb5df88.csv')
    
# df['TimeStamp'] = pd.to_datetime(df['interval_start_local'])
# df['Date'] = df['TimeStamp'].dt.date
# df['Hour'] = df['TimeStamp'].dt.hour
# df['Min'] = df['TimeStamp'].dt.minute

# df = df[['Date', 'Hour', 'Min', 'demand' ]]


# #06-06 # 06-07 $ 06-14 06-15

# specificDay = df[df['Date']== pd.Timestamp('2026-06-28').date()]


# load = np.array(specificDay['demand'])
# load = load/100   




histLoads = pd.read_csv('Pricing data/hrl_load_metered.csv')
histLoads = histLoads[['datetime_beginning_ept', 'mw']]
histLoads.columns = ['TimeStamp', 'mw'] 

 
histLoads['TimeStamp'] = pd.to_datetime(histLoads['TimeStamp'])
histLoads['Date'] = histLoads['TimeStamp'].dt.date
histLoads['Hour'] = histLoads['TimeStamp'].dt.hour


specificDay = histLoads[histLoads['Date']== pd.Timestamp('2025-12-10').date()]


load = np.array(specificDay['mw'])
load = load/2

n_hours = len(load)
x_hour = np.arange(n_hours)
x_15min = np.linspace(0, n_hours - 1, n_hours * 4)

load_15min = np.interp(x_15min, x_hour, load)




n_initial = 3000

scenarios_raw = np.random.normal(
    loc=load_15min,                        
    scale=std_load * load_15min,            
    size=(n_initial, 96)
)


km = KMeans(n_clusters=n_scenarios, random_state=42, n_init=10)

km.fit(scenarios_raw)

load_scenarios = km.cluster_centers_                   
labels = km.labels_

p = np.array([(labels == k).sum() / n_initial for k in range(n_scenarios)])

fig, ax = plt.subplots(figsize=(8, 5))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_scenarios))

for color, s in zip(colors, range(n_scenarios)):
    ax.plot(
        range(96),
        load_scenarios[s],
        color=color,
        lw=2,
        alpha=0.75,
        label=f"Scenario {s} (p={p[s]:.2f})" )

ax.plot(  range(96), load_15min, color="black",  lw=3, linestyle="--",label="Forecast" )

ax.set_title( "Generated Daily Load Scenarios", fontsize=18,  fontweight="bold",  pad=15, )

ax.set_xlabel( "T", fontsize=14,fontweight="bold" )

ax.set_ylabel( "Electrical Load (kW)", fontsize=14, fontweight="bold", )

ax.set_xticks(range(0, 96, 4))
ax.tick_params(axis="both", labelsize=11)

ax.grid(True, linestyle="--", alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper left",
    fontsize=10,
    frameon=True,
    fancybox=True,
    framealpha=0.95,
)

plt.tight_layout()
plt.show()
    
