#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:58:28 2026

@author: michael
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from gurobipy import *
from sklearn.cluster import KMeans
import matplotlib.cm as cm
np.random.seed(42)

from drMSPenumerateRT import * 
from plottingFuncsMSP import * 
from fiveMinScenarioTree import * 
from generateDAProfiles import * 
from GenerateRTProfiles import * 

import time

n_noon_nodes =2
n_evening_per_node = 2
std_night = 0.02

# Scenario Generation Params 


# Model Params
alpha = 0.5 # Trade off between Minimising Peak and Minimising ramping  1 is all peak 
max_rt_DR = 3 # Total number of RT DR resources 
max_da_DR = 5 # Total number of DA DR resources 
rampLimit = 60


# need

fullScenarios, scenarioProbs,scenarioToNode  = load_scenario_tree(n_noon_nodes, n_evening_per_node,
                        std_night,
                        n_initial=3000,
                        csv_path='Pricing data/hrl_load_metered.csv',
                        granularity = '5min',
                        target_date='2025-12-10')

### DA DR Profiles 

archetypes = {
    "1": {
        "hourly_raw": np.array([0, 0, -2, -4, -8, -12, -16, -12, -4, +15, +15, +12, +4, 0]),
        "tau_minutes": 25,
        "noise_std": 0.05,
        #"noise_std": 0,
        "noise_phi": 0.75,
        "settle_tol": 0.02,
    },
    "2": {
        "hourly_raw": np.array([0, 0, -10, -16, -20, -20, -16, -10, +12, +12, +6, 0]),
        "tau_minutes": 6,
        "noise_std": 0.08,
        #"noise_std": 0,
        "noise_phi": 0.6,
        "settle_tol": 0.02,
    },
    "3": {
        "hourly_raw": np.array([0, -6, -10, -10, -10, -10, -10, -6, +2, 0]),
        "tau_minutes": 4,
        #"noise_std": 0,
        "noise_std": 0.08,
        "noise_phi": 0.5,
        "settle_tol": 0.02,
    },
    "4": {
        "hourly_raw": np.array([0, 0, -4, -8, -12, -12, -8, -4, +20, +20, +20, +8, 0]),
        "tau_minutes": 30,
        "noise_std": 0.05,
        #"noise_std": 0,
        "noise_phi": 0.8,
        "settle_tol": 0.02,
    },
    "5": {
        "hourly_raw": np.array([0, 0, -16, -24, -24, -24, -24, -16, +25, +20, +15, +8, 0]),
        "tau_minutes": 8,
        "noise_std": 0.08,
        #"noise_std": 0,
        "noise_phi": 0.6,
        "settle_tol": 0.02,
    },
}

scheduled_profiles_DA, labels = generateDAProfiles(archetypes)


#### RT DR Profiles 
T = 288

delta_raw = {
    "1": np.array([-10, -10, +6, +0]),
    "2": np.array([-6, -6, +4, +3,0]),
    "3": np.array([+4, -10, -10, +10, +4, 0]),
    "4": np.array([+4, +4, -11, -11, +6,0]),
    # "5": np.array([-6, -6, -6, +6, +3, +2,0]),
    # "6": np.array([-4, -4, +3, +2,  0]),
    # "7": np.array([-8, -8, +6, +4, 0]),
    # "8": np.array([-12, +7,0]),
}

TAU_MINUTES = {
    "1":     12,
    "2":     12,
    "3":     12,
    "4":     12,
    # "5":     20,   
    # "6":     6,    
    # "7":     20,
    # "8":     6,
}

OP_START, OP_END = T // 2, T  

results = generateRTProfiles(
    delta_raw=delta_raw,
    tau_minutes=TAU_MINUTES,
    T=T,
    op_start=OP_START,
    op_end=OP_END,
    noise_std={"1": 0.1, "2": 0.08, "3": 0.08, "4": 0.08},
    #noise_std={"1": 0.1, "2": 0.08, "3": 0.08, "4": 0.08, "5": 0.08, "6":0.08, "7":0.08, "8": 0.08},
    noise_phi=0.1,
    settle_tol=0.02,
    noise_seed=42,
)

enumerated_profiles_RT = results["profiles"]
resource_names = results["resource_names"]
valid_i = results["valid_i"]
start_of = results["start_of"]
n_starts = results["n_starts"]

n_starts_max = max(n_starts.values())





alpha_list = [0.75, 0.5, 0.25]
rampValue = 50
node_configs = [
    (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (6, 5)
]

results_by_alpha = {}

for alpha in alpha_list:
    n_scenarios_list = []
    objVals = []
    solveTimes = []
    nodesExplored = []

    for n_noon_nodes, n_evening_per_node in node_configs:
        n_scenarios = n_noon_nodes * n_evening_per_node
        fullScenarios, scenarioProbs, scenarioToNode = load_scenario_tree(
            n_noon_nodes, n_evening_per_node,
            std_night,
            n_initial=3000,
            csv_path='Pricing data/hrl_load_metered.csv',
            granularity='5min',
            target_date='2025-12-10',
            showPlot=False
        )
        start = time.time()
        try:
            L_net_vals, delta_L_vals, delta_L_Fixed_vals, drModel, objval, r_vals, pi_vals, max_rolling_ramp, nodecount = drMSPModel_EnumerateRT(
                alpha, max_rt_DR, max_da_DR,
                scheduled_profiles_DA, enumerated_profiles_RT,
                n_noon_nodes, n_evening_per_node, scenarioProbs, fullScenarios, scenarioToNode,
                rampValue, n_starts_max, n_starts
            )
            elapsed = time.time() - start
            objVals.append(objval)
            nodesExplored.append(nodecount)
        except Exception as e:
            elapsed = time.time() - start
            objVals.append(0)
            nodesExplored.append(0)
            print(f"alpha={alpha}, {n_noon_nodes}x{n_evening_per_node} failed: {e}")

        solveTimes.append(elapsed)
        n_scenarios_list.append(n_scenarios)
        
        print(f"alpha={alpha}, {n_noon_nodes}x{n_evening_per_node} = {n_scenarios} scenarios -> "
              f"obj={objVals[-1]:.2f}, time={elapsed:.1f}s, nodes={nodesExplored[-1]}")

    results_by_alpha[alpha] = {
        'n_scenarios': np.array(n_scenarios_list),
        'objVals': np.array(objVals),
        'solveTimes': np.array(solveTimes),
        'nodesExplored': np.array(nodesExplored),
    }
    
    
import matplotlib.pyplot as plt

metrics = ['objVals', 'solveTimes', 'nodesExplored']
labels = ['Objective', 'Solve Time (s)', 'Nodes Explored']

fig, axes = plt.subplots(len(metrics), len(alpha_list), figsize=(5 * len(alpha_list), 10), sharex=True)

for col, alpha in enumerate(alpha_list):
    r = results_by_alpha[alpha]
    mask = r['objVals'] != 0

    for row, (key, label) in enumerate(zip(metrics, labels)):
        ax = axes[row, col]
        ax.plot(r['n_scenarios'][mask], r[key][mask], marker='o', color='C0')
        ax.grid(alpha=0.3)

        if row == 0:
            ax.set_title(f'α={alpha}')
        if col == 0:
            ax.set_ylabel(label)
        if row == len(metrics) - 1:
            ax.set_xlabel('# Scenarios')

plt.tight_layout()
plt.show()