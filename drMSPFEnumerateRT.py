#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:04:34 2026

@author: michael
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyomo.environ as pyo

#from fastForwardSelection import *




def drMSPModel_EnumerateRT(alpha, max_rt_DR, max_da_DR, H, scheduled_profiles, delta_flexible,n_noon_nodes, n_evening_per_node, scenario_probs, full_scenarios, scenario_to_node):
    n_scenarios = n_noon_nodes * n_evening_per_node

    J = range(len(scheduled_profiles))
    K = range(len(delta_flexible))
    T = range(full_scenarios.shape[1])
    S = range(n_scenarios)
    N = range(n_noon_nodes)
    
    m = pyo.ConcreteModel()
    

    ## Sets 
    m.J = pyo.Set(initialize=J)
    m.K = pyo.Set(initialize=K)
    m.T = pyo.Set(initialize=T)
    m.S = pyo.Set(initialize=S)
    m.N = pyo.Set(initialize = N)


    ## Vars 
    
    m.delta_L = pyo.Var(m.J, m.T, domain=pyo.Reals)
    
    m.delta_L_Flex = pyo.Var(m.K, m.T, m.N, domain=pyo.Reals)
    #m.delta_L_Fixed = pyo.Var(m.L, m.T, m.S, domain=pyo.Reals)
    
    m.L_net = pyo.Var(m.T, m.S, domain=pyo.Reals)
    
    m.u = pyo.Var(m.J, domain=pyo.Binary)
    
    m.u_Fixed = pyo.Var(m.L, m.S, domain=pyo.Binary)
    
    m.pi = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.r = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
    
    
    # Objective Functions
    
    def obj_rule(m):
        return (
            alpha * sum(scenario_probs[s] * m.pi[s] for s in m.S)
            +
            (1 - alpha) * sum(
                scenario_probs[s] * sum(m.r[t, s] for t in m.T)
                for s in m.S
            )
        )
    
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints 
    def net_load_rule(m, t, s):
        return (
            m.L_net[t, s]
            ==
            full_scenarios[s][t]
            + sum(m.delta_L[j, t] for j in m.J)
            + sum(m.delta_L_Flex[k, t, scenario_to_node[s]] for k in m.K)
        )
    
    m.net_load = pyo.Constraint(m.T, m.S, rule=net_load_rule)
    
    
    
    def peak_rule(m, t, s):
        return m.pi[s] >= m.L_net[t, s]
    
    m.peak = pyo.Constraint(m.T, m.S, rule=peak_rule)
    
    
    def ramp_up_rule(m, t, s):
        if t == 0:
            return pyo.Constraint.Skip
        return m.r[t, s] >= m.L_net[t, s] - m.L_net[t - 1, s]
    
    
    def ramp_down_rule(m, t, s):
        if t == 0:
            return pyo.Constraint.Skip
        return m.r[t, s] >= -(m.L_net[t, s] - m.L_net[t - 1, s])
    
    m.ramp_up = pyo.Constraint(m.T, m.S, rule=ramp_up_rule)
    m.ramp_down = pyo.Constraint(m.T, m.S, rule=ramp_down_rule)
    
    
    # da Constraints
    def da_rule(m, j, t):
        return m.delta_L[j, t] == scheduled_profiles[j][t] * m.u[j]
    
    m.da = pyo.Constraint(m.J, m.T, rule=da_rule)
    
    m.da_limit = pyo.Constraint(
        expr=sum(m.u[j] for j in m.J) <= max_da_DR
    )
    
    
    # Realtime Constraints
    
    def fixed_rule(m, l, t, s):
        return (
            m.delta_L_Fixed[l, t, s]
            == rt_fixed_profiles[l][t] * m.u_Fixed[l, s]
        )

    m.fixed = pyo.Constraint(m.L, m.T, m.S, rule=fixed_rule)

    
    def rt_fixed_rule(m, s):
        return sum(m.u_Fixed[l, s] for l in m.L) <= max_rt_DR

    m.rt_fixed = pyo.Constraint(m.S, rule=rt_fixed_rule)

    
    
    solver = pyo.SolverFactory("highs")
    solver.solve(m)
    
    #
    L_net_vals = {(s, t): pyo.value(m.L_net[t, s]) for t in T for s in S}
    
    delta_L_vals = {(j, t): pyo.value(m.delta_L[j, t]) for j in J for t in T}
    
    delta_L_Fixed_vals = {
        (s, l, t): pyo.value(m.delta_L_Fixed[l, t, s])
        for l in L for t in T for s in S
    }
     
    
    r_vals = {(s, t): pyo.value(m.r[t, s] ) for t in T for s in S}
    
    pi_vals = {(s): pyo.value(m.pi[s]) for s in S}
    
    objval = pyo.value(m.obj)
    
    return  (
        L_net_vals,
        delta_L_vals,
        delta_L_Fixed_vals,
        m,
        objval,r_vals,pi_vals
    )


