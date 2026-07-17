#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:00:28 2026

@author: michael
"""

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


def format_time_axis(ax, T):

    if T == 24:
        interval_minutes = 60
    elif T == 96:
        interval_minutes = 15
    elif T == 288:
        interval_minutes = 5
    else:
        interval_minutes = 1440 / T

    total_hours = T * interval_minutes / 60

    tick_hours = np.arange(
        0,
        total_hours + 1,
        2
    )

    tick_positions = tick_hours * 60 / interval_minutes

    ax.set_xticks(tick_positions)

    ax.set_xticklabels(
        [f"{int(h):02d}:00" for h in tick_hours],
        rotation=45,
        ha="right"
    )

    ax.set_xlim(0, T-1)

    return interval_minutes


def plotDRUsage(L_net_vals, delta_L_vals, delta_L_Flex_vals, 
                scenario_probs, objval, r_vals, pi_vals, scenarioToNode, max_rolling_ramp):
    
    S = sorted(set(s for s, t in L_net_vals.keys()))
    T = sorted(set(t for s, t in L_net_vals.keys()))
    J = sorted(set(j for j, t in delta_L_vals.keys()))
    K = sorted(set(k for n, k, t in delta_L_Flex_vals.keys()))
    
    ramping = sum(
        scenario_probs[s] * sum(r_vals[s, t] for t in T)
        for s in S
    )

    peak = sum(
        scenario_probs[s] * pi_vals[s]
        for s in S
    )
    
    
    n_scenarios = len(S)

    x = np.arange(len(T))

    interval_minutes = format_time_axis


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


    n_rows, n_cols = get_subplot_grid(n_scenarios)

    fig1, axes1 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6*n_cols, 4*n_rows),
        sharex=True,
        squeeze=False
    )


    axes1 = axes1.flatten()


    # Use bars for lower resolution
    use_bars = len(T) <= 96


    for idx, s in enumerate(S):

        ax = axes1[idx]


        da_total = np.zeros(len(T))


        for j in J:

            contrib = np.array(
                [
                    delta_L_vals[(j,t)]
                    for t in T
                ]
            )


            if np.all(np.abs(contrib)<1e-6):
                continue


            if use_bars:

                pos = np.clip(contrib,0,None)
                neg = np.clip(contrib,None,0)


                ax.bar(
                    x,
                    pos,
                    width=0.8,
                    color=dr_colors[j],
                    alpha=0.8,
                    label=f"DA DR {j}"
                )

            else:

                ax.plot(
                    x,
                    contrib,
                    color=dr_colors[j],
                    lw=1.5,
                    label=f"DA DR {j}"
                )


            da_total += contrib



        for k in K:

            contrib = np.array(
                [
                    delta_L_Flex_vals[
                        (scenarioToNode[s],k,t)
                    ]
                    for t in T
                ]
            )


            if np.all(np.abs(contrib)<1e-6):
                continue


            if use_bars:

                ax.bar(
                    x,
                    contrib,
                    width=0.4,
                    hatch='///',
                    alpha=0.8,
                    color=dr_colors[k],
                    label=f"Flex DR {k}"
                )


            else:

                ax.plot(
                    x,
                    contrib,
                    linestyle="--",
                    lw=1.5,
                    color=dr_colors[k],
                    label=f"Flex DR {k}"
                )



        ax.axhline(
            0,
            color='black',
            linewidth=0.8,
            linestyle='--'
        )


        ax.set_title(
            f"Node {scenarioToNode[s]}, Scenario {s} "
            f"(p={scenario_probs[s]:.2f})",
            fontsize=10
        )


        ax.set_xlabel("Time")
        ax.set_ylabel("DR Contribution (kW)")


        format_time_axis(
            ax,
            len(T)
        )


        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.4
        )



    for idx in range(n_scenarios,len(axes1)):
        axes1[idx].set_visible(False)



    handles=[]
    labels=[]


    for ax in axes1:

        h,l=ax.get_legend_handles_labels()

        handles.extend(h)
        labels.extend(l)



    by_label=OrderedDict(
        zip(labels,handles)
    )


    fig1.legend(
        by_label.values(),
        by_label.keys(),
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5,0),
        fontsize=9
    )


    fig1.suptitle(
        f"DR Load Contributions by Scenario, "
        f"Objval:{round(objval,2)}, "
        f"Ramping:{round(ramping,2)}, "
        f"Peak:{round(peak,2)}",
        fontsize=13
    )


    plt.tight_layout(
        rect=[0,0.06,1,0.97]
    )

    plt.show()
    
    
def plotLoadprofiles(L_net_vals, delta_L_vals, delta_L_Flex_vals,
                     scenario_probs, full_scenarios,
                     n_scenarios, objval,
                     r_vals, pi_vals, scenarioToNode, max_rolling_ramp):


    S = sorted(set(s for s,t in L_net_vals.keys()))
    T = sorted(set(t for s,t in L_net_vals.keys()))


    ramping = sum(
        scenario_probs[s]*sum(r_vals[s,t] for t in T)
        for s in S
    )


    peak = sum(
        scenario_probs[s]*pi_vals[s]
        for s in S
    )


    x=np.arange(len(T))


    n_scenarios=len(S)


    n_rows,n_cols=get_subplot_grid(n_scenarios)


    fig2,axes2=plt.subplots(
        n_rows,
        n_cols,
        figsize=(6*n_cols,4*n_rows),
        sharex=True,
        squeeze=False
    )


    axes2=axes2.flatten()



    for idx,s in enumerate(S):

        ax=axes2[idx]


        base=full_scenarios[s,:]


        L_net_s=np.array(
            [
                L_net_vals[(s,t)]
                for t in T
            ]
        )


        ax.plot(
            x,
            base,
            color='#E53935',
            lw=2,
            linestyle='--',
            label="Base load"
        )


        ax.plot(
            x,
            L_net_s,
            color='#1565C0',
            lw=2,
            label="Net load"
        )


        ax.axhline(
            0,
            color='black',
            linewidth=0.8,
            linestyle='--'
        )


        ax.set_title(
            f"Node {scenarioToNode[s]}, Scenario {s} "
            f"(p={scenario_probs[s]:.2f}), Max 3 Period Ramping: {max_rolling_ramp[s]:.2f}",
            fontsize=10
        )


        ax.set_xlabel("Time")
        ax.set_ylabel("Load (kW)")


        format_time_axis(
            ax,
            len(T)
        )


        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.4
        )


        ax.set_ylim(
            bottom=min(
                base.min(),
                L_net_s.min()
            )*0.95
        )


    for idx in range(n_scenarios,len(axes2)):

        axes2[idx].set_visible(False)



    handles,labels=axes2[0].get_legend_handles_labels()


    fig2.legend(
        handles,
        labels,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5,0),
        fontsize=9
    )


    fig2.suptitle(
        f"Base vs Net Load by Scenario, "
        f"Objval:{round(objval,2)}, "
        f"Ramping:{round(ramping,2)}, "
        f"Peak:{round(peak,2)}",
        fontsize=13
    )


    plt.tight_layout(
        rect=[0,0.06,1,0.97]
    )


    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict


def plotDRUsageForEnumerate(L_net_vals, delta_L_vals, delta_L_Fixed_vals,
                             scenario_probs, objval, r_vals, pi_vals,
                             scenarioToNode, max_rolling_ramp, flex_shape_names=None):
    """
    flex_shape_names: optional dict/list mapping k -> archetype name
        (e.g. meta[k][0] from your enumeration script). If provided,
        all k's belonging to the same archetype share one color, and
        the legend shows one entry per archetype instead of one per
        (shape, start-time) combination -- essential once K gets into
        the hundreds. If omitted, falls back to a colormap that scales
        to any number of k's without ever index-erroring.
    """

    S = sorted(set(s for s, t in L_net_vals.keys()))
    T = sorted(set(t for s, t in L_net_vals.keys()))
    J = sorted(set(j for j, t in delta_L_vals.keys()))
    K = sorted(set(k for n, k, t in delta_L_Fixed_vals.keys()))

    ramping = sum(scenario_probs[s] * sum(r_vals[s, t] for t in T) for s in S)
    peak = sum(scenario_probs[s] * pi_vals[s] for s in S)

    n_scenarios = len(S)
    x = np.arange(len(T))

    # ---- DA colors: small, fixed set is fine ----
    da_cmap = cm.get_cmap('tab10', max(len(J), 1))
    da_colors = {j: da_cmap(i) for i, j in enumerate(J)}

    # ---- Flex (RT-fixed) colors: scale to however many K there are ----
    if flex_shape_names is not None:
        # group by archetype name -> one color per archetype, not per k
        shape_names_sorted = sorted(set(
            flex_shape_names[k] if not isinstance(flex_shape_names, dict)
            else flex_shape_names[k]
            for k in K
        ))
        n_shapes = len(shape_names_sorted)
        shape_cmap = cm.get_cmap('tab10', max(n_shapes, 1))
        shape_color = {name: shape_cmap(i) for i, name in enumerate(shape_names_sorted)}
        flex_color_for_k = {k: shape_color[flex_shape_names[k]] for k in K}
        flex_label_for_k = {k: flex_shape_names[k] for k in K}
    else:
        # no grouping info given -- safe fallback, never index-errors
        # regardless of how large K is (continuous colormap sample per k)
        flex_cmap = cm.get_cmap('nipy_spectral')
        flex_color_for_k = {
            k: flex_cmap(i / max(len(K) - 1, 1)) for i, k in enumerate(K)
        }
        flex_label_for_k = {k: f"Flex DR {k}" for k in K}

    n_rows, n_cols = get_subplot_grid(n_scenarios)
    fig1, axes1 = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
        sharex=True, squeeze=False
    )
    axes1 = axes1.flatten()

    use_bars = len(T) <= 96

    for idx, s in enumerate(S):
        ax = axes1[idx]
        da_total = np.zeros(len(T))

        for j in J:
            contrib = np.array([delta_L_vals[(j, t)] for t in T])
            if np.all(np.abs(contrib) < 1e-6):
                continue

            if use_bars:
                pos = np.clip(contrib, 0, None)
                ax.bar(x, pos, width=0.8, color=da_colors[j], alpha=0.8,
                       label=f"DA DR {j}")
            else:
                ax.plot(x, contrib, color=da_colors[j], lw=1.5, label=f"DA DR {j}")

            da_total += contrib

        for k in K:
            contrib = np.array([delta_L_Fixed_vals[(s, k, t)] for t in T])
            if np.all(np.abs(contrib) < 1e-6):
                continue

            c = flex_color_for_k[k]
            label = flex_label_for_k[k]

            if use_bars:
                ax.bar(x, contrib, width=0.4, hatch='///', alpha=0.8,
                       color=c, label=label)
            else:
                ax.plot(x, contrib, linestyle="--", lw=1.5, color=c, label=label)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(
            f"Node {scenarioToNode[s]}, Scenario {s} (p={scenario_probs[s]:.2f}), Max 3 Period Ramping: {max_rolling_ramp[s]:.2f}",
            fontsize=10
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("DR Contribution (kW)")
        format_time_axis(ax, len(T))
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    for idx in range(n_scenarios, len(axes1)):
        axes1[idx].set_visible(False)

    handles, labels = [], []
    for ax in axes1:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    by_label = OrderedDict(zip(labels, handles))
    fig1.legend(
        by_label.values(), by_label.keys(),
        loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), fontsize=9
    )

    fig1.suptitle(
        f"DR Load Contributions by Scenario, "
        f"Objval:{round(objval, 2)}, "
        f"Ramping:{round(ramping, 2)}, "
        f"Peak:{round(peak, 2)}",
        fontsize=13
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.show()