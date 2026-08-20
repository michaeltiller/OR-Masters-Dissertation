#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:14:23 2026

@author: michael
"""

import pyomo.environ as pyo



def drMSPModel_Penalty(lambda_ramp, max_rt_DR, max_da_DR, scheduled_profiles, rt_fixed_profiles,
                        n_noon_nodes, n_evening_per_node, scenario_probs, full_scenarios,
                        scenario_to_node, rampLimit, n_starts_max, n_starts,
                        n_starts_da):
    n_scenarios = n_noon_nodes * n_evening_per_node

    J = range(len(scheduled_profiles))
    L = range(len(rt_fixed_profiles))
    T = range(full_scenarios.shape[1])
    S = range(n_scenarios)
    N = range(n_noon_nodes)

    m = pyo.ConcreteModel()

    ## Sets
    m.J = pyo.Set(initialize=J)
    m.L = pyo.Set(initialize=L)
    m.T = pyo.Set(initialize=T)
    m.S = pyo.Set(initialize=S)
    m.N = pyo.Set(initialize=N)

    midday = int(len(T) / 2)

    ## Vars

    m.delta_L = pyo.Var(m.J, m.T, domain=pyo.Reals)

    m.delta_L_Fixed = pyo.Var(m.L, m.T, m.S, domain=pyo.Reals)

    m.L_net = pyo.Var(m.T, m.S, domain=pyo.Reals)

    def u_index_rule(m):
        return (
            (j, k)
            for j in m.J
            for k in range(n_starts_da[j])   # only valid starts for THIS resource
        )

    def u_fixed_index_rule(m):
        return (
            (l, i, n)
            for l in m.L
            for n in m.N
            for i in range(n_starts[l])   # only valid starts for THIS resource
        )

    m.JK = pyo.Set(dimen=2, initialize=u_index_rule)
    m.u = pyo.Var(m.JK, domain=pyo.Binary)

    m.LIN = pyo.Set(dimen=3, initialize=u_fixed_index_rule)
    m.u_Fixed = pyo.Var(m.LIN, domain=pyo.Binary)

    m.pi = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.r = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)

    m.v_ramp = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)

    # Objective Functions

    def obj_rule(m):
        return (
             sum(scenario_probs[s] * m.pi[s] for s in m.S)
            +
            sum(
                scenario_probs[s] * sum(m.r[t, s] for t in m.T)
                for s in m.S
             ) + lambda_ramp * sum(
                    scenario_probs[s] * m.v_ramp[t, s]
                    for t in m.T
                    for s in m.S
                    if t >= midday + 3
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

    # DA Constraints 
    def da_rule(m, j, t):
        return m.delta_L[j, t] == sum(
            scheduled_profiles[j][k][t] * m.u[j, k]
            for k in range(n_starts_da[j])
        )

    m.da = pyo.Constraint(m.J, m.T, rule=da_rule)

    m.da_limit = pyo.Constraint(
        expr=sum(m.u[j, k] for j in m.J for k in range(n_starts_da[j])) <= max_da_DR
    )

    def one_start_per_da_resource_rule(m, j):
        return sum(m.u[j, k] for k in range(n_starts_da[j])) <= 1

    m.one_start_per_resource = pyo.Constraint(m.J, rule=one_start_per_da_resource_rule)

    # Realtime Constraints

    def fixed_rule(m, l, t, s):
        n = scenario_to_node[s]
        return m.delta_L_Fixed[l, t, s] == sum(
            rt_fixed_profiles[l][i][t] * m.u_Fixed[l, i, n]
            for i in range(n_starts[l])     
        )

    m.fixed = pyo.Constraint(m.L, m.T, m.S, rule=fixed_rule)

    def rt_fixed_rule(m, n):
        return sum(
            m.u_Fixed[l, i, n]
            for l in m.L for i in range(n_starts[l])   
        ) <= max_rt_DR

    m.rt_fixed = pyo.Constraint(m.N, rule=rt_fixed_rule)

    def rolling_ramp_rule(m, t, s):
        if t < midday + 3:
            return pyo.Constraint.Skip
        return (
            m.r[t, s]
            + m.r[t - 1, s]
            + m.r[t - 2, s]
            <= rampLimit + m.v_ramp[t, s]
        )

    m.rolling_ramp = pyo.Constraint(m.T, m.S, rule=rolling_ramp_rule)

    def one_start_per_rt_resource_rule(m, l, n):
        return sum(m.u_Fixed[l, i, n] for i in range(n_starts[l])) <= 1

    m.one_start_per_rt_resource = pyo.Constraint(m.L, m.N, rule=one_start_per_rt_resource_rule)

    solver = pyo.SolverFactory("highs")
    solver.options['mip_rel_gap'] = 0.01


    nodecount = solver._solver_model.getInfo().mip_node_count
    L_net_vals = {(s, t): pyo.value(m.L_net[t, s]) for t in T for s in S}

    delta_L_vals = {(j, t): pyo.value(m.delta_L[j, t]) for j in J for t in T}

    delta_L_Fixed_vals = {
        (s, l, t): pyo.value(m.delta_L_Fixed[l, t, s])
        for l in L for t in T for s in S}

    u_vals = {
        (j, k): pyo.value(m.u[j, k])
        for j in J for k in range(n_starts_da[j])
    }

    u_Fixed_vals = {
        (l, i, n): pyo.value(m.u_Fixed[l, i, n])
        for l in L for n in N for i in range(n_starts[l])
    }

    r_vals = {(s, t): pyo.value(m.r[t, s]) for t in T for s in S}

    pi_vals = {(s): pyo.value(m.pi[s]) for s in S}

    objval = pyo.value(m.obj)

    # Output the max rolling ramp

    rolling_ramp_vals = {}

    max_rolling_ramp = {}

    for s in S:
        rolling_ramp_vals[s] = {}

        for t in range(midday, len(T)):

            if t < 2:
                rolling_ramp_vals[s][t] = None
                continue

            rolling_ramp_vals[s][t] = (
                abs(L_net_vals[(s, t)] - L_net_vals[(s, t-1)])
                +
                abs(L_net_vals[(s, t-1)] - L_net_vals[(s, t-2)])
                +
                abs(L_net_vals[(s, t-2)] - L_net_vals[(s, t-3)])
                if t >= midday + 3 else None
            )

        # maximum rolling ramp for scenario
        max_rolling_ramp[s] = max(
            v for v in rolling_ramp_vals[s].values()
            if v is not None
        )

    return (
        L_net_vals,
        delta_L_vals,
        delta_L_Fixed_vals,
        m,
        objval, r_vals, pi_vals, max_rolling_ramp, nodecount,
        u_vals, u_Fixed_vals
    )


def drMSPModel_Penalty_FixedDA(
    lambda_ramp, max_rt_DR, max_da_DR, scheduled_profiles, rt_fixed_profiles,
    n_noon_nodes, n_evening_per_node, scenario_probs, full_scenarios,
    scenario_to_node, rampLimit, n_starts_max, n_starts,
    n_starts_da, u_star
):
 
    n_scenarios = n_noon_nodes * n_evening_per_node
 
    J = range(len(scheduled_profiles))
    L = range(len(rt_fixed_profiles))
    T = range(full_scenarios.shape[1])
    S = range(n_scenarios)
    N = range(n_noon_nodes)
 
    m = pyo.ConcreteModel()
 
    ## Sets
    m.J = pyo.Set(initialize=J)
    m.L = pyo.Set(initialize=L)
    m.T = pyo.Set(initialize=T)
    m.S = pyo.Set(initialize=S)
    m.N = pyo.Set(initialize=N)
 
    midday = int(len(T) / 2)
 
    ## Vars
    m.delta_L = pyo.Var(m.J, m.T, domain=pyo.Reals)
 
    m.delta_L_Fixed = pyo.Var(m.L, m.T, m.S, domain=pyo.Reals)
 
    m.L_net = pyo.Var(m.T, m.S, domain=pyo.Reals)
 
    def u_index_rule(m):
        return (
            (j, k)
            for j in m.J
            for k in range(n_starts_da[j])
        )
 
    def u_fixed_index_rule(m):
        return (
            (l, i, n)
            for l in m.L
            for n in m.N
            for i in range(n_starts[l])
        )
 
    m.JK = pyo.Set(dimen=2, initialize=u_index_rule)
    m.u = pyo.Var(m.JK, domain=pyo.Binary)
 
   # Fix first stage Decisions 
    for j in J:
        for k in range(n_starts_da[j]):
            m.u[j, k].fix(round(u_star[(j, k)]))
 
    m.LIN = pyo.Set(dimen=3, initialize=u_fixed_index_rule)
    m.u_Fixed = pyo.Var(m.LIN, domain=pyo.Binary)
 
    m.pi = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.r = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
 
    m.v_ramp = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
 
    ## Objective Function
    def obj_rule(m):
        return (
            sum(scenario_probs[s] * m.pi[s] for s in m.S)
            +
             sum(
                scenario_probs[s] * sum(m.r[t, s] for t in m.T)
                for s in m.S
            ) + lambda_ramp * sum(
                    scenario_probs[s] * m.v_ramp[t, s]
                    for t in m.T
                    for s in m.S
                    if t >= midday + 3
                )
        )
 
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
 
    ## Constraints
 
    def net_load_rule(m, t, s):
        return (
            m.L_net[t, s]
            ==
            full_scenarios[s][t]
            + sum(m.delta_L[j, t] for j in m.J)
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
 
    ## Day-Ahead Constraints
 
    def da_rule(m, j, t):
        return m.delta_L[j, t] == sum(
            scheduled_profiles[j][k][t] * m.u[j, k]
            for k in range(n_starts_da[j])
        )
 
    m.da = pyo.Constraint(m.J, m.T, rule=da_rule)
 
    m.da_limit = pyo.Constraint(
        expr=sum(
            m.u[j, k]
            for j in m.J
            for k in range(n_starts_da[j])
        ) <= max_da_DR
    )
 
    def one_start_per_da_resource_rule(m, j):
        return sum(
            m.u[j, k]
            for k in range(n_starts_da[j])
        ) <= 1
 
    m.one_start_per_resource = pyo.Constraint(
        m.J,
        rule=one_start_per_da_resource_rule
    )
 
    ## Real-Time Constraints
 
    def fixed_rule(m, l, t, s):
        n = scenario_to_node[s]
        return m.delta_L_Fixed[l, t, s] == sum(
            rt_fixed_profiles[l][i][t] * m.u_Fixed[l, i, n]
            for i in range(n_starts[l])
        )
 
    m.fixed = pyo.Constraint(m.L, m.T, m.S, rule=fixed_rule)
 
    def rt_fixed_rule(m, n):
        return sum(
            m.u_Fixed[l, i, n]
            for l in m.L
            for i in range(n_starts[l])
        ) <= max_rt_DR
 
    m.rt_fixed = pyo.Constraint(m.N, rule=rt_fixed_rule)
 
    def one_start_per_rt_resource_rule(m, l, n):
        return sum(
            m.u_Fixed[l, i, n]
            for i in range(n_starts[l])
        ) <= 1
 
    m.one_start_per_rt_resource = pyo.Constraint(
        m.L,
        m.N,
        rule=one_start_per_rt_resource_rule
    )
 
    ## Rolling ramp constraint (soft, via v_ramp penalty)
 
    def rolling_ramp_rule(m, t, s):
        if t < midday + 3:
            return pyo.Constraint.Skip
        return (
            m.r[t, s]
            + m.r[t - 1, s]
            + m.r[t - 2, s]
            <= rampLimit + m.v_ramp[t, s]
        )
 
    m.rolling_ramp = pyo.Constraint(
        m.T,
        m.S,
        rule=rolling_ramp_rule
    )
 
    ## Solve
 
    solver = pyo.SolverFactory("highs")
    solver.options['mip_rel_gap'] = 0.01
    results = solver.solve(m)
  

    nodecount = solver._solver_model.getInfo().mip_node_count

    ## Outputs
 
    L_net_vals = {
        (s, t): pyo.value(m.L_net[t, s])
        for t in T
        for s in S
    }
 
    delta_L_vals = {
        (j, t): pyo.value(m.delta_L[j, t])
        for j in J
        for t in T
    }
 
    delta_L_Fixed_vals = {
        (s, l, t): pyo.value(m.delta_L_Fixed[l, t, s])
        for l in L
        for t in T
        for s in S
    }
 
    u_vals = {
        (j, k): pyo.value(m.u[j, k])
        for j in J
        for k in range(n_starts_da[j])
    }
 
    u_Fixed_vals = {
        (l, i, n): pyo.value(m.u_Fixed[l, i, n])
        for l in L
        for n in N
        for i in range(n_starts[l])
    }
 
    r_vals = {
        (s, t): pyo.value(m.r[t, s])
        for t in T
        for s in S
    }
 
    pi_vals = {
        s: pyo.value(m.pi[s])
        for s in S
    }
 
    objval = pyo.value(m.obj)
 
    ## Maximum rolling ramp
 
    rolling_ramp_vals = {}
    max_rolling_ramp = {}
 
    for s in S:
        rolling_ramp_vals[s] = {}
 
        for t in range(midday, len(T)):
 
            if t < 2:
                rolling_ramp_vals[s][t] = None
                continue
 
            rolling_ramp_vals[s][t] = (
                abs(L_net_vals[(s, t)] - L_net_vals[(s, t - 1)])
                + abs(L_net_vals[(s, t - 1)] - L_net_vals[(s, t - 2)])
                + abs(L_net_vals[(s, t - 2)] - L_net_vals[(s, t - 3)])
                if t >= midday + 3 else None
            )
 
        max_rolling_ramp[s] = max(
            v for v in rolling_ramp_vals[s].values()
            if v is not None
        )
 
    return (
        L_net_vals,
        delta_L_vals,
        delta_L_Fixed_vals,
        m,
        objval,
        r_vals,
        pi_vals,
        max_rolling_ramp,
        nodecount,
        u_vals,
        u_Fixed_vals,
    )