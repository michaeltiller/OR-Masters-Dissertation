#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:05:25 2026

@author: michael
"""

import pyomo.environ as pyo


def DRModel(alpha, beta, max_rt_DR, max_da_DR,
            H, scheduled_profiles, delta_flexible,
            rt_fixed_profiles, n_scenarios, p, load_scenarios):

    J = range(len(scheduled_profiles))
    K = range(len(delta_flexible))
    L = range(len(rt_fixed_profiles))
    T = range(24)
    S = range(n_scenarios)

    m = pyo.ConcreteModel()

    # -----------------------
    # Sets
    # -----------------------
    m.J = pyo.Set(initialize=J)
    m.K = pyo.Set(initialize=K)
    m.L = pyo.Set(initialize=L)
    m.T = pyo.Set(initialize=T)
    m.S = pyo.Set(initialize=S)

    # -----------------------
    # Variables
    # -----------------------

    m.delta_L = pyo.Var(m.J, m.T, domain=pyo.Reals)

    m.delta_L_Flex = pyo.Var(m.K, m.T, m.S, domain=pyo.Reals)
    m.delta_L_Fixed = pyo.Var(m.L, m.T, m.S, domain=pyo.Reals)

    m.L_net = pyo.Var(m.T, m.S, domain=pyo.Reals)

    m.u = pyo.Var(m.J, domain=pyo.Binary)

    m.u_Flex = pyo.Var(m.K, m.T, m.S, domain=pyo.Binary)
    m.u_Fixed = pyo.Var(m.L, m.S, domain=pyo.Binary)

    m.pi = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.r = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)



    def obj_rule(m):
        return (
            alpha * sum(p[s] * m.pi[s] for s in m.S)
            +
            (1 - alpha) * sum(
                p[s] * sum(m.r[t, s] for t in m.T)
                for s in m.S
            )
        )

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


    def net_load_rule(m, t, s):
        return (
            m.L_net[t, s]
            ==
            load_scenarios[s][t]
            + sum(m.delta_L[j, t] for j in m.J)
            + sum(m.delta_L_Flex[k, t, s] for k in m.K)
            + sum(m.delta_L_Fixed[l, t, s] for l in m.L)
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



    def da_rule(m, j, t):
        return m.delta_L[j, t] == scheduled_profiles[j][t] * m.u[j]

    m.da = pyo.Constraint(m.J, m.T, rule=da_rule)

    m.da_limit = pyo.Constraint(
        expr=sum(m.u[j] for j in m.J) <= max_da_DR
    )



    def fixed_rule(m, l, t, s):
        return (
            m.delta_L_Fixed[l, t, s]
            == rt_fixed_profiles[l][t] * m.u_Fixed[l, s]
        )

    m.fixed = pyo.Constraint(m.L, m.T, m.S, rule=fixed_rule)

    
    # Flex convolution 
    
    def flex_rule(m, k, t, s):

        Hk = H[k]
        start = max(0, t - Hk + 1)

        return (
            m.delta_L_Flex[k, t, s]
            ==
            sum(
                delta_flexible[k][t - tp] * m.u_Flex[k, tp, s]
                for tp in range(start, t + 1)
            )
        )

    m.flex = pyo.Constraint(m.K, m.T, m.S, rule=flex_rule)

    # At most one activation per window
    def flex_window_rule(m, k, t, s):
        Hk = H[k]
        start = max(0, t - Hk + 1)

        return sum(m.u_Flex[k, tp, s] for tp in range(start, t + 1)) <= 1

    m.flex_window = pyo.Constraint(m.K, m.T, m.S, rule=flex_window_rule)

    # Only one activation per k,s
    def flex_total_rule(m, k, s):
        return sum(m.u_Flex[k, t, s] for t in m.T) <= 1

    m.flex_total = pyo.Constraint(m.K, m.S, rule=flex_total_rule)

    # Prevent infeasible end-starts
    def flex_end_rule(m, k, t, s):
        if t + H[k] - 1 > 23:
            return m.u_Flex[k, t, s] == 0
        return pyo.Constraint.Skip

    m.flex_end = pyo.Constraint(m.K, m.T, m.S, rule=flex_end_rule)



    def rt_flex_rule(m, s):
        return sum(m.u_Flex[k, t, s] for k in m.K for t in m.T) <= (1 - beta) * max_rt_DR

    def rt_fixed_rule(m, s):
        return sum(m.u_Fixed[l, s] for l in m.L) <= beta * max_rt_DR

    m.rt_flex = pyo.Constraint(m.S, rule=rt_flex_rule)
    m.rt_fixed = pyo.Constraint(m.S, rule=rt_fixed_rule)



    solver = pyo.SolverFactory("highs")
    solver.solve(m)

    #
    L_net_vals = {(s, t): pyo.value(m.L_net[t, s]) for t in T for s in S}
    delta_L_vals = {(j, t): pyo.value(m.delta_L[j, t]) for j in J for t in T}

    delta_L_Flex_vals = {
        (s, k, t): pyo.value(m.delta_L_Flex[k, t, s])
        for k in K for t in T for s in S
    }

    delta_L_Fixed_vals = {
        (s, l, t): pyo.value(m.delta_L_Fixed[l, t, s])
        for l in L for t in T for s in S
    }
    
    r_vals = {(s, t): pyo.value(m.r[t, s] ) for t in T for s in S}
    
    pi_vals = {(s): pyo.value(m.pi[s]) for s in S}

    objval = pyo.value(m.obj)

    return (
        L_net_vals,
        delta_L_vals,
        delta_L_Flex_vals,
        delta_L_Fixed_vals,
        m,
        objval,r_vals,pi_vals
    )