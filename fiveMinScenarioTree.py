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
from scipy.spatial.distance import cdist





def fastForwardSelection(p, scenarios, n_select):
    
    S = len(scenarios)

    # Pairwise distance matrix
    C_original = cdist(scenarios, scenarios)
    C = C_original.copy()
 
    # Candidate scenarios
    J = np.arange(S)
 
    # Selected scenarios
    selected = []
 
    # Fast Forward Selection
    for _ in range(n_select):
 
        # Compute weighted distance of every candidate
        z = np.sum(p[:, None] * C, axis=0)
 
        # Ignore scenarios already selected
        if selected:
            z[selected] = np.inf
 
        # Select scenario with minimum cost
        u_new = np.argmin(z)
        selected.append(u_new)
 
        # Remove from candidate set
        J = J[J != u_new]
 
        # Update minimum distances
        C[:, J] = np.minimum(C[:, J], C[:, [u_new]])
 
    selected = np.array(selected)
 
    # Deleted scenarios
    mask = np.ones(S, dtype=bool)
    mask[selected] = False
    deleted = np.where(mask)[0]
 
    # Reassign probabilities
    newProbs = np.zeros(len(selected))
 
    # Lookup table: original scenario index -> reduced scenario index
    selected_index = {s: i for i, s in enumerate(selected)}
 
    # Assign deleted scenario probabilities to nearest selected scenario
    for j in deleted:
        nearest = selected[np.argmin(C_original[j, selected])]
        newProbs[selected_index[nearest]] += p[j]
 
    # Add probabilities of selected scenarios themselves
    for i, s in enumerate(selected):
        newProbs[i] += p[s]

    return selected, deleted, newProbs




def load_scenario_tree(
    n_noon_nodes,
    n_evening_per_node,
    std_night,
    n_initial=3000,
    csv_path='Pricing data/hrl_load_metered.csv',
    granularity='15min',
    target_date='2025-12-10',
    showPlot = True):
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
    
    
    if granularity == '5min':
        x  = np.linspace(0, n_hours - 1, n_hours * 12)
        load_interpolated = np.interp(x, x_hour, load)

    elif granularity == '15min':
        x  = np.linspace(0, n_hours - 1, n_hours * 4)
        load_interpolated = np.interp(x, x_hour, load)
    elif granularity == 'hour':
      
        load_interpolated = load
    
    T_max = len(load_interpolated)
    
    # raw_full = np.random.normal(
    #     loc=load_interpolated, scale=std_night * load_interpolated, size=(n_initial, T_max)
    # )
    
    rho = 0.97
    error_std = std_night * np.mean(load_interpolated)
    
    errors = np.zeros((n_initial,T_max))
    
    errors[:,0] = np.random.normal(
        0,
        error_std,
        n_initial
    )
    
    for t in range(1,T_max):
        errors[:,t] = (
            rho * errors[:,t-1]
            +
            np.random.normal(
                0,
                error_std*np.sqrt(1-rho**2),
                n_initial
            )
        )
    
    raw_full = load_interpolated + errors
    raw_probs = np.full(n_initial, 1.0 / n_initial)
    
    t_noon = int(T_max /2) 
    
    
    prenoon_slice = raw_full[:, :t_noon ]
    afternoonSlice = raw_full[:, t_noon  :]

    
    
    
    preNoonSelected, deleted, preNoonNewProbs = fastForwardSelection(raw_probs, prenoon_slice, n_noon_nodes)
    
    preNoonPath = prenoon_slice[preNoonSelected]
    
    
    C = cdist(prenoon_slice, prenoon_slice)
    
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
    
    if showPlot == True:
        
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
                        range(T_max),
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
        #ax.plot( range(T_max),load_interpolated, color="black",lw=3, linestyle="--",  label="Forecast")
        
        ax.set_title( "Two-Stage Scenario Tree", fontsize=18, fontweight="bold", pad=15)
        
        ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold")
        ax.set_ylabel("Electrical Load (kW)", fontsize=14, fontweight="bold")
        #ax.set_xticks(range(0, T_max, 2))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=9, frameon=True, fancybox=True, framealpha=0.95)
    
        plt.tight_layout()
        plt.show()

    return np.array(fullScenarios), np.array(scenarioProbs),scenarioToNode 