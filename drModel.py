#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:38:31 2026

@author: michael
"""


import numpy as np
from gurobipy import *

def DRModel(alpha, beta, max_rt_DR,max_da_DR,  H, scheduled_profiles, delta_flexible, rt_fixed_profiles, n_scenarios, p, load_scenarios):
    
    # Sets 
    
    J = range(len(scheduled_profiles))
    K = range(len(delta_flexible))
    T = range(24)
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
    drModel.addConstrs(r[t,s]  >= L_net[t,s] - L_net[t-1,s] for t in range(1,24) for s in S)
    drModel.addConstrs(r[t,s]  >= -(L_net[t,s] - L_net[t-1,s]) for t in range(1,24) for s in S)
    
    # Load 
    
    
    drModel.addConstrs(delta_L[j, t] == scheduled_profiles[j, t] * u[j] for j in J for t in T)
    
    
    
    drModel.addConstrs(delta_L_Fixed[l, t, s] == rt_fixed_profiles[l, t] * u_Fixed[l, s] for l in L for t in T for s in S)
    
    # Max number of DA resources 
    drModel.addConstr(quicksum(u[j] for j in J ) <= max_da_DR) 

     
    # Convolution constraints for Second Stage
    
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
    
    
    drModel.optimize()
    
    
    
    L_net_vals     = {(s, t): L_net[t, s].X     for t in T for s in S}
    delta_L_vals   = {(j, t): delta_L[j, t].X   for j in J for t in T}
    delta_L_Flex_vals = {(s, k, t): delta_L_Flex[k, t, s].X for k in K for t in T for s in S}
    delta_L_Fixed_vals = {(s, l, t): delta_L_Fixed[l, t, s].X for l in L for t in T for s in S}
    
    return L_net_vals, delta_L_vals, delta_L_Flex_vals, delta_L_Fixed_vals, drModel

