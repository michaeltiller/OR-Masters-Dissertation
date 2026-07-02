#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:04:53 2026

@author: michael
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:17:42 2026

@author: michael
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from gurobipy import *
#from loadscenariosv2 import *
from matplotlib.patches import Patch

from collections import OrderedDict

np.random.seed(42)

scheduled_profiles = np.array([

    # Commercial pre-cooling
    [0,0,0,0,0,0,0,0,0,0,2,4,6,8,6,2,-10,-10,-6,-2,0,0,0,0],

    # Large office buildings
    [0,0,0,0,0,0,0,0,0,2,4,6,8,8,6,4,-12,-15,-12,-5,0,0,0,0],

    # Industrial scheduling
    [0,0,0,0,0,0,0,0,0,0,5,8,10,10,8,5,-18,-18,-12,-6,0,0,0,0],

    # Refrigeration preloading
    [0,0,0,0,0,0,0,0,2,4,6,8,8,6,4,2,-10,-12,-10,-6,-2,0,0,0],

    # Long duration thermal storage
    #[0,0,0,0,0,0,0,0,3,5,7,9,9,7,5,3,-15,-15,-12,-8,-3,0,0,0],

])


rt_fixed_profiles = np.array([

    # Short HVAC event
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-10,-10,8,6,4,2,0,0],

    # Two-hour reduction
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-8,-8,10,4,2,0,0,0,0],

    # Deep but brief event
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-15,-15,-10,10,8,5,2,0,0],

    # Three-hour commercial curtailment
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-20,-20,-20,8,8,8,4,4],

    # Residential thermostat aggregation
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-12,-10,-8,6,5,3,2,0],

    # Interruptible industrial load
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-25,-25,0,12,10,3,0,0,0],

])


# parameters 

alpha = 0.5
beta = 0
max_rt_DR = 5 
max_da_DR = 5



H = np.array([4, 3, 4, 5]) * 4

delta_raw = [
    np.repeat(np.array([-15, -15, +15, +15]), 4) ,     
    np.repeat(np.array([-10, -10, +20]), 4)/4,            
    np.repeat(np.array([-20, -20, +30, +10]),4 ),      
    np.repeat(np.array([+5, +5, -20, +5, +5]),4),      
]

scheduled_profiles = np.repeat(scheduled_profiles, 4, axis=1) 
rt_fixed_profiles = np.repeat(rt_fixed_profiles, 4, axis=1) 




c = np.array([3, 2, 2, 1])     

C = H.copy() 
H_max = H.max()


delta_flexible = np.zeros((len(H), H_max))
for k in range(len(H)):
    delta_flexible[k, :H[k]] = delta_raw[k]

#n_scenarios = 5
#std_load = 0.03



#p,load_scenarios =  loadscenarios(n_scenarios, std_load)



# Sets 

J = range(len(scheduled_profiles))
K = range(len(delta_raw))
T = range(96)
S = range(n_scenarios)
L = range(len(rt_fixed_profiles))


drModel= Model("DR Model")

drModel.setParam("OutputFlag", 0)

delta_L = drModel.addVars(J, T, lb = -GRB.INFINITY)

delta_L_Flex = drModel.addVars(K, T, S, lb = -GRB.INFINITY)
delta_L_Fixed = drModel.addVars(L, T, S, lb = -GRB.INFINITY)

L_net = drModel.addVars(T, S, lb = -GRB.INFINITY)


u = drModel.addVars(J, vtype = GRB.BINARY )

u_Flex = drModel.addVars(K,T, S, vtype = GRB.BINARY)
u_Fixed = drModel.addVars(L, S, vtype = GRB.BINARY )

pi = drModel.addVars( S , lb = 0)
r = drModel.addVars(T, S , lb = 0)


drModel.setObjective(alpha * quicksum(p[s] * pi[s] for s in S) + (1-alpha)* quicksum( p[s] * quicksum(r[t,s] for t in T) for s in S))


# DR Constraints #### 
drModel.addConstrs(L_net[t,s] == load_scenarios[s,t] + quicksum(delta_L[j,t] for j in J) + quicksum(delta_L_Flex[k, t,s] for k in K) + quicksum(delta_L_Fixed[l, t, s] for l in L) for t in T for s in S)

# Peak Load ####

drModel.addConstrs(pi[s] >= L_net[t,s] for t in T for s in S)

#Absolute ramp limits 
drModel.addConstrs(r[t,s]  >= L_net[t,s] - L_net[t-1,s] for t in range(1,96) for s in S)
drModel.addConstrs(r[t,s]  >= -(L_net[t,s] - L_net[t-1,s]) for t in range(1,96) for s in S)

# Load 


drModel.addConstrs(delta_L[j, t] == scheduled_profiles[j, t] * u[j] for j in J for t in T)



drModel.addConstrs(delta_L_Fixed[l, t, s] == rt_fixed_profiles[l, t] * u_Fixed[l, s] for l in L for t in T for s in S)


# Number of DA Dr resources 

drModel.addConstr(quicksum(u[j] for j in J ) <= max_da_DR) 

 

drModel.addConstrs(
    (delta_L_Flex[k, t, s] == quicksum(delta_flexible[k, t - t_prime] * u_Flex
                                    [k, t_prime, s] for t_prime in range(max(0, t - H[k] + 1), t + 1)
    ) for k in K for t in T for s in S)
)

drModel.addConstrs(
    (1 >= quicksum(  u_Flex[k, t_prime, s] for t_prime in range(max(0, t - H[k] + 1), t + 1)
    ) for k in K for t in T for s in S)
)



drModel.addConstrs(
    1 >= quicksum( u_Flex[k, t, s] for t in T) for k in K for s in S
)


drModel.addConstrs(
    (u_Flex[k, t, s] == 0
     for k in K for t in T for s in S
     if t + H[k] - 1 > len(T) - 1) 
)





# FLexibillity Constriants 

drModel.addConstrs(quicksum(u_Flex[k,t,s] for k in K for t in T) <= (1-beta) * max_rt_DR for s in S) 


drModel.addConstrs(quicksum(u_Fixed[l,s] for l in L) <= beta * max_rt_DR for s in S) 



# Hourly activation 


y = {t: 1 if t % 4 == 0 else 0 for t in T}

drModel.addConstrs(
    (u_Flex[k,t,s] <= y[t] for k in K for t in T for s in S)
)

drModel.optimize()



L_net_vals     = {(s, t): L_net[t, s].X     for t in T for s in S}
delta_L_vals   = {(j, t): delta_L[j, t].X   for j in J for t in T}
delta_L_Flex_vals = {(s, k, t): delta_L_Flex[k, t, s].X for k in K for t in T for s in S}
delta_L_Fixed_vals = {(s, l, t): delta_L_Fixed[l, t, s].X for l in L for t in T for s in S}


import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def get_subplot_grid(n_plots, max_cols=3):
    n_cols = min(max_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols


# ------------------------------------------------------------
# Common quantities
# ------------------------------------------------------------
S = sorted(set(s for s, t in L_net_vals.keys()))
T = sorted(set(t for s, t in L_net_vals.keys()))
J = sorted(set(j for j, t in delta_L_vals.keys()))
K = sorted(set(k for s, k, t in delta_L_Flex_vals.keys()))
L = sorted(set(l for s, l, t in delta_L_Fixed_vals.keys()))

n_scenarios = len(S)
n_periods = len(T)

dt_hours = 0.25                    # 15 min intervals
ticks_per_hour = int(round(1 / dt_hours))

x = np.arange(n_periods)

tick_locations = np.arange(0, n_periods, ticks_per_hour)
tick_labels = [f"{h:02d}:00" for h in range(len(tick_locations))]

dr_colors = [
    '#2196F3',
    '#4CAF50',
    '#FF9800',
    '#E91E63',
    '#9C27B0',
    '#F44336',
    '#009688',
    '#3F51B5',
    '#795548',
    '#607D8B'
]

# ============================================================
# DR CONTRIBUTIONS
# ============================================================

width = 0.25

n_rows, n_cols = get_subplot_grid(n_scenarios)

fig1, axes1 = plt.subplots(
    n_rows,
    n_cols,
    figsize=(7 * n_cols, 5 * n_rows),
    sharex=True,
    sharey=True,
    squeeze=False,
)

axes1 = axes1.flatten()

for idx, s in enumerate(S):

    ax = axes1[idx]

    da_pos = np.zeros(n_periods)
    da_neg = np.zeros(n_periods)

    for j in J:

        contrib = np.array([delta_L_vals[(j, t)] for t in T])

        if np.all(np.abs(contrib) < 1e-6):
            continue

        pos = np.clip(contrib, 0, None)
        neg = np.clip(contrib, None, 0)

        ax.bar(
            x - width,
            pos,
            width,
            bottom=da_pos,
            color=dr_colors[j],
            edgecolor='black',
            linewidth=0.5,
            label=f'DA DR {j}',
        )

        ax.bar(
            x - width,
            neg,
            width,
            bottom=da_neg,
            color=dr_colors[j],
            edgecolor='black',
            linewidth=0.5,
        )

        da_pos += pos
        da_neg += neg

    flex_pos = np.zeros(n_periods)
    flex_neg = np.zeros(n_periods)

    for k in K:

        contrib = np.array([delta_L_Flex_vals[(s, k, t)] for t in T])

        if np.all(np.abs(contrib) < 1e-6):
            continue

        pos = np.clip(contrib, 0, None)
        neg = np.clip(contrib, None, 0)

        ax.bar(
            x,
            pos,
            width,
            bottom=flex_pos,
            color=dr_colors[k],
            edgecolor='black',
            linewidth=0.5,
            hatch='///',
            label=f'Flex DR {k}',
        )

        ax.bar(
            x,
            neg,
            width,
            bottom=flex_neg,
            color=dr_colors[k],
            edgecolor='black',
            linewidth=0.5,
            hatch='///',
        )

        flex_pos += pos
        flex_neg += neg

    fixed_pos = np.zeros(n_periods)
    fixed_neg = np.zeros(n_periods)

    for l in L:

        contrib = np.array([delta_L_Fixed_vals[(s, l, t)] for t in T])

        if np.all(np.abs(contrib) < 1e-6):
            continue

        pos = np.clip(contrib, 0, None)
        neg = np.clip(contrib, None, 0)

        ax.bar(
            x + width,
            pos,
            width,
            bottom=fixed_pos,
            color=dr_colors[l],
            edgecolor='black',
            linewidth=0.5,
            hatch='xx',
            label=f'Fixed DR {l}',
        )

        ax.bar(
            x + width,
            neg,
            width,
            bottom=fixed_neg,
            color=dr_colors[l],
            edgecolor='black',
            linewidth=0.5,
            hatch='xx',
        )

        fixed_pos += pos
        fixed_neg += neg

    ax.axhline(0, color='black', ls='--', lw=0.8)

    ax.set_title(f"Scenario {s} (p={p[s]:.2f})")

    ax.set_ylabel("DR Contribution (kW)")
    ax.set_xlabel("Time")

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=45)

    ax.grid(axis='y', alpha=0.3)

for ax in axes1[n_scenarios:]:
    ax.set_visible(False)

handles = []
labels = []

for ax in axes1:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

legend = OrderedDict(zip(labels, handles))

fig1.legend(
    legend.values(),
    legend.keys(),
    loc='lower center',
    ncol=6,
    bbox_to_anchor=(0.5, 0.0),
)

fig1.suptitle("DR Load Contributions by Scenario")

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.show()

# ============================================================
# LOAD PROFILES
# ============================================================

width = 0.35

n_rows, n_cols = get_subplot_grid(n_scenarios)

fig2, axes2 = plt.subplots(
    n_rows,
    n_cols,
    figsize=(7 * n_cols, 5 * n_rows),
    sharex=True,
    squeeze=False,
)

axes2 = axes2.flatten()

for idx, s in enumerate(S):

    ax = axes2[idx]

    base = load_scenarios[s]
    net = np.array([L_net_vals[(s, t)] for t in T])

    ax.plot(
        x,
        base,
        '--o',
        color='#E53935',
        ms=2,
        lw=1.5,
        label='Base load',
    )

    ax.plot(
        x,
        net,
        '-o',
        color='#1565C0',
        ms=2,
        lw=2,
        label='Net load',
    )

    ax.axhline(0, color='black', ls='--', lw=0.8)

    ax.set_title(f"Scenario {s} (p={p[s]:.2f})")

    ax.set_ylabel("Load (kW)")
    ax.set_xlabel("Time")

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=45)

    ax.grid(axis='y', alpha=0.3)

    ax.set_ylim(bottom=min(base.min(), net.min()) * 0.95)

for ax in axes2[n_scenarios:]:
    ax.set_visible(False)

handles, labels = axes2[0].get_legend_handles_labels()

fig2.legend(
    handles,
    labels,
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.5, 0.0),
)

fig2.suptitle("Base vs Net Load by Scenario")

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.show()