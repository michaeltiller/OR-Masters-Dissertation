#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:21:45 2026

@author: michael
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from collections import OrderedDict

def get_subplot_grid(n_plots, max_cols=3):
    n_cols = min(max_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols

def plotDRUsage(L_net_vals, delta_L_vals, delta_L_Flex_vals, delta_L_Fixed_vals, p, objval):
    
    S = sorted(set(s for s, t in L_net_vals.keys()))
    T = sorted(set(t for s, t in L_net_vals.keys()))
    J = sorted(set(j for j, t in delta_L_vals.keys()))
    K = sorted(set(k for s, k, t in delta_L_Flex_vals.keys()))
    L=  sorted(set(l for s, l, t in delta_L_Fixed_vals.keys()))
    
    n_scenarios = len(S)
    x     = np.arange(1, 25)
    width = 0.25                       
    dr_colors = ['#2196F3',   '#4CAF50','#FF9800', '#E91E63',  '#9C27B0',  '#F44336',  '#009688',  '#3F51B5',  '#795548','#607D8B',  ]  

    n_rows, n_cols = get_subplot_grid(n_scenarios)

    fig1, axes1 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    
    axes1 = axes1.flatten()

    for idx, s in enumerate(S):
        ax = axes1[idx]

        da_pos = np.zeros(24)
        da_neg = np.zeros(24)

        for j in J:
            contrib = np.array([delta_L_vals[(j, t)] for t in T])

            if np.all(np.abs(contrib) < 1e-6):
                continue

            pos = np.clip(contrib, 0, None)
            neg = np.clip(contrib, None, 0)

            ax.bar(x - width, pos, width,
                   bottom=da_pos,
                   color=dr_colors[j],
                   edgecolor='black',
                   linewidth=0.5,
                   hatch=None,
                   alpha=0.9,
                   label=f"DA DR {j}")

            ax.bar(x - width, neg, width,
                   bottom=da_neg,
                   color=dr_colors[j],
                   edgecolor='black',
                   linewidth=0.5,
                   hatch=None,
                   alpha=0.9)
            
            da_pos += pos
            da_neg += neg

        flex_pos = np.zeros(24)
        flex_neg = np.zeros(24)

        for k in K:
            contrib = np.array([delta_L_Flex_vals[(s, k, t)] for t in T])

            if np.all(np.abs(contrib) < 1e-6):
                continue

            pos = np.clip(contrib, 0, None)
            neg = np.clip(contrib, None, 0)

            ax.bar(x, pos, width,
                   bottom=flex_pos,
                   color=dr_colors[k],
                   edgecolor='black',
                   linewidth=0.5,
                   hatch='///',
                   alpha=0.9,
                   label=f"Flex DR {k}")

            ax.bar(x, neg, width,
                   bottom=flex_neg,
                   color=dr_colors[k],
                   edgecolor='black',
                   linewidth=0.5,
                   hatch='///',
                   alpha=0.9)

            flex_pos += pos
            flex_neg += neg

        fixed_pos = np.zeros(24)
        fixed_neg = np.zeros(24)

        for l in L:
            contrib = np.array([delta_L_Fixed_vals[(s, l, t)] for t in T])

            if np.all(np.abs(contrib) < 1e-6):
                continue

            pos = np.clip(contrib, 0, None)
            neg = np.clip(contrib, None, 0)

            ax.bar(x + width, pos, width,
                   bottom=fixed_pos,
                   color=dr_colors[l],
                   edgecolor='black',
                   linewidth=0.5,
                   hatch='xx',
                   alpha=0.9,
                   label=f"Fixed DR {l}")

            ax.bar(x + width, neg, width,
                   bottom=fixed_neg,
                   color=dr_colors[l],
                   edgecolor='black',
                   linewidth=0.5,
                   hatch='xx',
                   alpha=0.9)

            fixed_pos += pos
            fixed_neg += neg

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"Scenario {s} (p={p[s]:.2f})", fontsize=10)
        ax.set_xlabel("Hour")
        ax.set_ylabel("DR Contribution (kW)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in range(24)])
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    for idx in range(n_scenarios, len(axes1)):
        axes1[idx].set_visible(False)


    handles = []
    labels = []
    
    for ax in axes1:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    by_label = OrderedDict(zip(labels, handles))

    fig1.legend(by_label.values(), by_label.keys(),
                loc='lower center',
                ncol=6,
                bbox_to_anchor=(0.5, 0.0),
                fontsize=9)




    fig1.suptitle(f"DR Load Contributions by Scenario, Objval:{round(objval, 2)}", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.show()


    
    
def plotLoadprofiles(L_net_vals, delta_L_vals, delta_L_Flex_vals, delta_L_Fixed_vals, p, load_scenarios, n_scenarios, objval):
    x     = np.arange(1, 25)
    width = 0.35
  
  
    
    S = sorted(set(s for s, t in L_net_vals.keys()))
    T = sorted(set(t for s, t in L_net_vals.keys()))
    J = sorted(set(j for j, t in delta_L_vals.keys()))
    K = sorted(set(k for s, k, t in delta_L_Flex_vals.keys()))
    
    n_scenarios = len(S)

    n_rows, n_cols = get_subplot_grid(n_scenarios)
    
    fig2, axes2 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        sharex=True,
        squeeze=False
    )
    
    axes2 = axes2.flatten()
    
    
    for idx, s in enumerate(S):
        ax = axes2[idx]
        base    = load_scenarios[s, :]
        L_net_s = np.array([L_net_vals[(s, t)] for t in T])
        ax.plot(x, base,    color='#E53935', marker='o', markersize=3, lw=1.5, linestyle='--', label="Base load")
        ax.plot(x, L_net_s, color='#1565C0', marker='o', markersize=3, lw=2,   linestyle='-',  label="Net load")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"Scenario {s}  (p={p[s]:.2f})", fontsize=10)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Load (kW)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in range(24)])
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(bottom=min(base.min(), L_net_s.min()) * 0.95)
    
    
    for idx in range(n_scenarios, len(axes2)):
        axes2[idx].set_visible(False)
    
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=2,
                bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig2.suptitle(f"Base vs Net Load by Scenario, Objval:{round(objval, 2)}", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.show()
    
    
        