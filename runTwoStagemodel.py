#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:36:37 2026

@author: michael
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:27:03 2026

@author: michael
"""

from drTwoStageEnumerate import * 
from plottingFuncsMSP import * 
from fiveMinScenarioTree import * 
from generateDAProfiles import * 
from GenerateRTProfiles import * 
import time

n_noon_nodes =4
n_evening_per_node = 1
std_night = 0.02

# Scenario Generation Params 


# Model Params
alpha = 0.5 # Trade off between Minimising Peak and Minimising ramping  1 is all peak 
max_rt_DR = 3 # Total number of RT DR resources 
max_da_DR = 5 # Total number of DA DR resources 
rampLimit = 50


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
    "5": np.array([-6, -6, -6, +6, +3, +2,0]),
    "6": np.array([-4, -4, +3, +2,  0]),
    "7": np.array([-8, -8, +6, +4, 0]),
    "8": np.array([-12, +7,0]),
}

TAU_MINUTES = {
    "1":     12,
    "2":     12,
    "3":     12,
    "4":     12,
    "5":     20,   
    "6":     6,    
    "7":     20,
    "8":     6,
}

OP_START, OP_END = T // 2, T  

results = generateRTProfiles(
    delta_raw=delta_raw,
    tau_minutes=TAU_MINUTES,
    T=T,
    op_start=OP_START,
    op_end=OP_END,
    # noise_std={"1": 0.1, "2": 0.08, "3": 0.08, "4": 0.08},
    noise_std={"1": 0.1, "2": 0.08, "3": 0.08, "4": 0.08, "5": 0.08, "6":0.08, "7":0.08, "8": 0.08},
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


### Call the model v


n_noon_nodes =4
n_evening_per_node = 1

L_net_vals, delta_L_vals, delta_L_Fixed_vals, drModel, objval, r_vals, pi_vals, max_rolling_ramp, nodecount= drTwoStage_EnumerateRT( alpha, max_rt_DR, max_da_DR,
             scheduled_profiles_DA, enumerated_profiles_RT,n_noon_nodes, n_evening_per_node, scenarioProbs, fullScenarios, scenarioToNode, rampLimit, n_starts_max, n_starts)

n_scenarios = n_noon_nodes * n_evening_per_node



n_noon_nodes =2
n_evening_per_node = 2

fullScenarios, scenarioProbs,scenarioToNode  = load_scenario_tree(n_noon_nodes, n_evening_per_node,
                        std_night,
                        n_initial=3000,
                        csv_path='Pricing data/hrl_load_metered.csv',
                        granularity = '5min',
                        target_date='2025-12-10')

L_net_vals, delta_L_vals, delta_L_Fixed_vals, drModel, objval, r_vals, pi_vals, max_rolling_ramp, nodecount = drMSPModel_EnumerateRT(
    alpha, max_rt_DR, max_da_DR,
    scheduled_profiles_DA, enumerated_profiles_RT,
    n_noon_nodes, n_evening_per_node, scenarioProbs, fullScenarios, scenarioToNode,
    rl, n_starts_max, n_starts)

plotLoadprofiles(L_net_vals, delta_L_vals, delta_L_Fixed_vals, scenarioProbs, fullScenarios, n_scenarios, objval, r_vals,pi_vals, scenarioToNode, max_rolling_ramp)

plotDRUsageForEnumerate(L_net_vals, delta_L_vals, delta_L_Fixed_vals, scenarioProbs, objval, r_vals,pi_vals, scenarioToNode, max_rolling_ramp)



