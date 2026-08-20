import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_resource_series(J, L, T, S, delta_L_vals, delta_L_Fixed_vals):
    
    series = {s: {} for s in S}
    for s in S:
        for j in J:
            series[s][('DA', j)] = np.array([delta_L_vals[(j, t)] for t in T])
        for l in L:
            series[s][('RT', l)] = np.array([delta_L_Fixed_vals[(s, l, t)] for t in T])
    return series


def rebound_overlap_index(series_s, tol=1e-6):
    
    resources = list(series_s.keys())
    stacking, offset = 0.0, 0.0
    pairwise_stack, pairwise_offset = {}, {}

    for a_idx in range(len(resources)):
        for b_idx in range(a_idx + 1, len(resources)):
            a, b = resources[a_idx], resources[b_idx]
            va, vb = series_s[a], series_s[b]

            reb_a, reb_b = np.clip(va, 0, None), np.clip(vb, 0, None)
            act_a, act_b = np.clip(-va, 0, None), np.clip(-vb, 0, None)

            stack_ab = np.minimum(reb_a, reb_b).sum()
            offset_ab = (np.minimum(reb_a, act_b) + np.minimum(reb_b, act_a)).sum()

            stacking += stack_ab
            offset += offset_ab
            if stack_ab > tol:
                pairwise_stack[(a, b)] = stack_ab
            if offset_ab > tol:
                pairwise_offset[(a, b)] = offset_ab

    return stacking, offset, pairwise_stack, pairwise_offset


def rebound_overlap_summary(series, scenario_probs, S):
   
    rows = []
    pair_stack_totals, pair_offset_totals = {}, {}

    for s in S:
        stacking, offset, pw_stack, pw_offset = rebound_overlap_index(series[s])
        p = scenario_probs[s]
        rows.append({'scenario': s, 'prob': p, 'stacking': stacking, 'offset': offset})

        for pair, val in pw_stack.items():
            pair_stack_totals[pair] = pair_stack_totals.get(pair, 0.0) + p * val
        for pair, val in pw_offset.items():
            pair_offset_totals[pair] = pair_offset_totals.get(pair, 0.0) + p * val

    df = pd.DataFrame(rows)
    df['exp_stacking'] = df['prob'] * df['stacking']
    df['exp_offset'] = df['prob'] * df['offset']

    top_stack = sorted(pair_stack_totals.items(), key=lambda kv: -kv[1])
    top_offset = sorted(pair_offset_totals.items(), key=lambda kv: -kv[1])

    return df, top_stack, top_offset


def per_step_ramp(load_series):
   
    r = np.full(len(load_series), np.nan)
    r[1:] = np.abs(np.diff(load_series))
    return r


def rolling_sum_ramp(r_series, T, midday):
    
    T = list(T)
    out = np.full(len(T), np.nan)
    for idx, t in enumerate(T):
        if t >= midday + 3:
            out[idx] = r_series[idx] + r_series[idx - 1] + r_series[idx - 2]
    return out


def raw_rolling_ramp_series(s, T, full_scenarios, midday):
    
    r = per_step_ramp(full_scenarios[s])
    return rolling_sum_ramp(r, T, midday)


def solved_rolling_ramp_series(s, T, r_vals, midday):
   
    T = list(T)
    r = np.array([r_vals[(s, t)] for t in T])
    return rolling_sum_ramp(r, T, midday)

def add_broken_zero_axis(fig, ax0, position, base, L_net_s, T, break_ratio=0.15):
    
    import matplotlib.gridspec as gridspec

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=position, height_ratios=[1 - break_ratio, break_ratio], hspace=0.05
    )
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

    data_min = min(base.min(), L_net_s.min())
    data_max = max(base.max(), L_net_s.max())
    pad = (data_max - data_min) * 0.08

    ax_top.set_ylim(data_min - pad, data_max + pad)
    ax_bot.set_ylim(0, data_min * 0.15)  # small sliver anchored at 0

    for ax in (ax_top, ax_bot):
        ax.plot(T, base, color='#E53935', lw=2, linestyle='--')
        ax.plot(T, L_net_s, color='#1565C0', lw=2)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False)
    ax_bot.set_yticks([0])

    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0], [0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0], [1], transform=ax_bot.transAxes, **kwargs)

    return ax_top, ax_bot

def _step_to_clock(step, steps_per_day):
    minutes_per_step = 24 * 60 / steps_per_day
    total_minutes = (step * minutes_per_step) % (24 * 60)
    return f"{int(total_minutes // 60):02d}:{int(total_minutes % 60):02d}"

def plot_scenario_dispatch(s, T, series_s, r_vals, full_scenarios, rampLimit, midday,
                            L_net_vals, scenario_to_node, scenario_probs,
                            title=None, palette=None, format_time_axis=None):
    T_full = list(T)
    resources = [r for r in series_s if np.any(np.abs(series_s[r]) > 1e-9)]
    if not resources:
        print(f"scenario {s}: no DR resources dispatched, nothing to plot")
        return
    if palette is None:
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(resources), 1)))
        palette = {r: cmap[i] for i, r in enumerate(resources)}

    rr_raw_full = raw_rolling_ramp_series(s, T_full, full_scenarios, midday)
    rr_solved_full = solved_rolling_ramp_series(s, T_full, r_vals, midday)

    t_start = midday + 3
    idx = [i for i, t in enumerate(T_full) if t >= t_start]
    T = [T_full[i] for i in idx]
    base = full_scenarios[s, idx]
    L_net_s = np.array([L_net_vals[(s, t)] for t in T])
    pos_parts = [np.clip(series_s[r][idx], 0, None) for r in resources]
    neg_parts = [np.clip(series_s[r][idx], None, 0) for r in resources]
    rr_raw = rr_raw_full[idx] if isinstance(rr_raw_full, np.ndarray) else [rr_raw_full[i] for i in idx]
    rr_solved = rr_solved_full[idx] if isinstance(rr_solved_full, np.ndarray) else [rr_solved_full[i] for i in idx]
    colors = [palette[r] for r in resources]
    labels = [f"{r[0]} {r[1]}" for r in resources]

    fig = plt.figure(figsize=(7, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.5, 1], hspace=0.1)

    ax0_top, ax0_bot = add_broken_zero_axis(fig, None, gs[0], base, L_net_s, T)
    ax1 = fig.add_subplot(gs[1], sharex=ax0_top)
    ax2 = fig.add_subplot(gs[2], sharex=ax0_top)

    ax0_top.set_title(
        f"Afternoon Peak: Node {scenario_to_node[s]}, Scenario {s} (p={scenario_probs[s]:.2f})",
        fontsize=10
    )
    ax0_top.set_ylabel("Load (MW)")
    ax0_top.legend(
        handles=[
            plt.Line2D([], [], color='#E53935', lw=2, linestyle='--', label="Base load"),
            plt.Line2D([], [], color='#1565C0', lw=2, label="Base load + DR Contribution"),
        ],
        loc="upper left", fontsize=7, frameon=True, framealpha=0.9
    )

    ax1.stackplot(T, *pos_parts, colors=colors, alpha=0.85, labels=labels)
    ax1.stackplot(T, *neg_parts, colors=colors, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Δ Load Contribution" )
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.legend(loc="lower left", fontsize=7, frameon=True,
               framealpha=0.9, ncol=2)
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    ax2.plot(T, rr_raw, color="grey", lw=1.6, linestyle="--",
              label="Rolling Ramp, before DR")
    ax2.plot(T, rr_solved, color="black", lw=1.6,
              label="Rolling Ramp, after DR")
    ax2.axhline(rampLimit, color="firebrick", linestyle="--", lw=1.2,
                label="rampLimit")
    ax2.set_xlabel("Time step", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Rolling 3-period Ramp")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9)
    steps_per_day = len(T_full)
    steps_per_hour = steps_per_day / 24
    first_hour_step = int(np.ceil(t_start / steps_per_hour) * steps_per_hour)
    tick_positions = np.arange(first_hour_step, T_full[-1] + 1, steps_per_hour)
    tick_labels = [_step_to_clock(int(round(t)), steps_per_day) for t in tick_positions]

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.set_xlabel("Time of Day", fontsize=13, fontweight="bold")

    for ax in (ax0_top, ax0_bot, ax1):
        ax.tick_params(labelbottom=False)

    plt.tight_layout()
    plt.show()