#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:36:31 2026

@author: michael
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:49:23 2026

@author: michael
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


def pairwiseDistances(scenarios):
    diff = scenarios[:, None, :] - scenarios[None, :, :]
    C = np.sqrt(np.sum(diff ** 2, axis=-1))
    return C


def fastForwardSelection(p, scenarios, n_select):

    N = len(scenarios)
    
    S = len(scenarios)
    
    # Candidate scenarios (not selected yet)
    J = np.arange(S)

    # Pairwise distance matrix
    C_original = pairwiseDistances(scenarios)
    C = C_original.copy()    
    selected = []
    
    for i in range(n_select):
        z = {}
        
        for u in J:
            z[u] = sum(  p[k] * C[k, u]  for k in range(S) if k != u)
    
        # Select scenario with lowest cost 
        u_new = min(z, key=z.get)
    
        selected.append(u_new)
    
        # Remove from the set 
        J = J[J != u_new]

        for k in range(S):
            C[k, J] = np.minimum( C[k, J],C[k, u_new])
    
    selected = np.array(selected)
    
    deleted = np.array(
        [j for j in range(S) if j not in selected])

    
    # Reassign Probabillites 
    
    newProbs = np.zeros(len(selected))
    
    
    for j in deleted:
        # nearest selected scenario
        nearest = selected[
        np.argmin(C_original[j, selected])]
        
        newProbs[
            np.where(selected == nearest)[0][0]
        ] += p[j]
    
    
    # probabilities of selected scenarios
    for idx, s in enumerate(selected):
        newProbs[idx] += p[s]
        

    return selected, deleted, newProbs
    

def load_scenario_tree(n_noon_nodes, n_evening_per_node,
                        std_night,
                        t_noon=3600, n_initial=3000,
                        csv_path='Pricing data/hrl_load_metered.csv',
                        target_date='2025-12-10'):
    histLoads = pd.read_csv(csv_path)
    histLoads = histLoads[['datetime_beginning_ept', 'mw']]
    histLoads.columns = ['TimeStamp', 'mw']
    
    histLoads['TimeStamp'] = pd.to_datetime(histLoads['TimeStamp'])
    histLoads['Date'] = histLoads['TimeStamp'].dt.date
    histLoads['Hour'] = histLoads['TimeStamp'].dt.hour
    
    specificDay = histLoads[histLoads['Date'] == pd.Timestamp(target_date).date()]
    load = np.array(specificDay['mw'])
    load = load / 2
    
    n_hours = len(load)
    x_hour = np.arange(n_hours)
    x_5min = np.linspace(0, n_hours - 1, n_hours * 12)

    load_5min = np.interp(x_15min, x_hour, load)

    
    
    
    T_max = len(load_5min)
    
    raw_full = np.random.normal(
        loc=load_5min, scale=std_night * load, size=(n_initial, T_max)
    )
    raw_probs = np.full(n_initial, 1.0 / n_initial)
    
    prenoon_slice = raw_full[:, :t_noon ]
    afternoonSlice = raw_full[:, t_noon  :]

    
    
    
    preNoonSelected, deleted, preNoonNewProbs = fastForwardSelection(raw_probs, prenoon_slice, n_noon_nodes)
    
    preNoonPath = prenoon_slice[preNoonSelected]
    
    
    C = pairwiseDistances(prenoon_slice)
    
    distanceToNodes = C[:, preNoonSelected]
    cluster = np.argmin(distanceToNodes, axis=1)
    
    
    fullScenarios = []
    scenarioProbs = []
    scenarioToNode = {}
    
    s_idx = 0
    
    for node in range(n_noon_nodes):
    
        members = np.where(cluster == node)[0]
    
        afternoonScenarios = afternoonSlice[members]
    
        afternoonProbs = raw_probs[members]
        afternoonProbs = afternoonProbs / afternoonProbs.sum()
    
    
        afternoonSelected, deleted, afternoonNewProbs = fastForwardSelection(
            afternoonProbs,
            afternoonScenarios,
            n_evening_per_node
        )
    
        afternoonPath = afternoonScenarios[afternoonSelected]
    
        for j in range(n_evening_per_node):
            full_path = np.concatenate([preNoonPath[node],  afternoonPath[j]])
            fullScenarios.append(full_path)
            scenarioProbs.append(preNoonNewProbs[node] * afternoonNewProbs[j] )
            scenarioToNode[s_idx] = node
            s_idx += 1
            
    fullScenarios = np.array(fullScenarios)
    scenarioProbs = np.array(scenarioProbs)
    
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    colors = plt.cm.Blues(
        np.linspace(0.35, 0.9, n_noon_nodes)
    )
    
    for node in range(n_noon_nodes):
        node_color = colors[node]
        # Plot all final scenarios belonging to this noon node
        for s in range(len(fullScenarios)):
            if scenarioToNode[s] == node:
                ax.plot(
                    range(24),
                    fullScenarios[s, :],
                    color=node_color,
                    lw=1.2,
                    alpha=0.5)
        # Plot representative pre-noon path
        ax.plot(
            range(t_noon),
            preNoonPath[node],
            color=node_color,
            lw=2.8,
            label=f"Noon-node {node} (p={preNoonNewProbs[node]:.2f})")
    # Noon boundary
    ax.axvline( t_noon, color="grey", linestyle=":", lw=1.5)
    # Original forecast
    ax.plot( range(24),load, color="black",lw=3, linestyle="--",  label="Forecast")
    
    ax.set_title( "Two-Stage Scenario Tree (Fast Forward Selection)", fontsize=18, fontweight="bold", pad=15)
    
    ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold")
    ax.set_ylabel("Electrical Load (kW)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=True, fancybox=True, framealpha=0.95)

    plt.tight_layout()
    plt.show()

    return np.array(fullScenarios), np.array(scenarioProbs),scenarioToNode 