#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:45:48 2026
@author: michael
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def generateRTProfiles(
    delta_raw,
    tau_minutes,
    T,
    op_start,
    op_end,
    intervals_per_hour=12,
    dt_minutes=5,
    start_step=3,
    hourly_interrupt_only=False,
    showPlot=True,
    noise_std=None,        # dict {resource: std fraction of peak}, or None for no noise
    noise_phi=0.7,         # AR(1) autocorrelation, 0=white noise, closer to 1=smoother wander
    settle_tol=0.02,
    noise_seed=None
):
    INTERVALS_PER_HOUR = 12
    HOURS_PER_DAY = 24
    T = HOURS_PER_DAY * INTERVALS_PER_HOUR
    DT_MINUTES = 5

    rng = np.random.default_rng(noise_seed)
    noise_std = noise_std or {}

    def ar1_noise(n, std, phi):
        """Autocorrelated noise: each step drifts from the last rather than
        jumping independently, giving a 'wandering/jagged' look instead of
        speckly white noise."""
        eps_std = std * np.sqrt(1 - phi**2)
        eps = rng.normal(0, eps_std, size=n)
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = phi * noise[t-1] + eps[t]
        return noise

    def upsample_physical(profile, tau):
        """First-order lag response with a lead-in period prepended, using
        the profile's own first value as the lead-in target. The ramp
        settles onto that first target DURING the lead-in, so declared
        plateaus (even consecutive identical entries like -15,-15) come
        out fully flat for their whole reported duration instead of the
        ramp eating into the first hour.

        Also returns a noise mask: True everywhere except the settled
        negative (reduction) plateau, which stays a clean flat line."""
        alpha = dt_minutes / tau
        # settling time (samples) for the lag to get within settle_tol of
        # any step, amplitude-independent for a first-order system
        settle_steps = int(np.ceil(np.log(settle_tol) / np.log(1 - alpha)))
        lead_steps = min(settle_steps, intervals_per_hour)  # cap at 1hr worth

        profile_ext = np.append(profile, 0)
        setpoint_core = np.repeat(profile_ext, intervals_per_hour)
        lead_setpoint = np.full(lead_steps, profile_ext[0])
        setpoint = np.concatenate([lead_setpoint, setpoint_core])

        y = np.zeros(len(setpoint))
        for t in range(1, len(setpoint)):
            y[t] = y[t-1] + alpha * (setpoint[t] - y[t-1])

        out_len = lead_steps + len(profile) * intervals_per_hour
        y = y[:out_len]
        setpoint = setpoint[:out_len]

        ref = np.max(np.abs(setpoint)) if np.max(np.abs(setpoint)) > 0 else 1.0
        settled = np.abs(setpoint - y) <= settle_tol * ref
        no_noise = settled & (setpoint < 0)  # flat only on the negative plateau
        noise_mask = ~no_noise
        return y, noise_mask

    delta_5min = {}
    for r in delta_raw:
        y, noise_mask = upsample_physical(delta_raw[r], tau_minutes[r])
        std = noise_std.get(r, 0.0) * np.max(np.abs(delta_raw[r]))
        if std > 0:
            noise = ar1_noise(len(y), std, noise_phi)
            y = y + noise * noise_mask
        delta_5min[r] = y

    resource_names = list(delta_5min.keys())
    L = range(len(resource_names))
    raw_profiles = {l: [] for l in L}
    raw_starts = {l: [] for l in L}

    if showPlot:
        # fig, ax = plt.subplots(figsize=(10, 5))
        # for resource, profile in delta_5min.items():
        #     t = np.arange(len(profile)) / INTERVALS_PER_HOUR
        #     ax.plot(t, profile, linewidth=2, label=resource)
        # ax.set_xlabel("Time (hours)")
        # ax.set_ylabel("Δ Load (MW) ")
        # ax.set_title("Real-Time Demand Response Profiles")
        # ax.grid(True)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
    
        
        
        COLORS = ["#2E4057", "#D64933", "#4C9A2A", "#4F86C6", "#E8A33D", "#8E44AD"]

        n = len(delta_5min)
        ncols = 2
        nrows = -(-n // ncols)  # ceil division
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharey=True, sharex = True, dpi=150)
        axes = np.array(axes).reshape(-1)  # flatten for uniform indexing
        
        for i, (resource, profile) in enumerate(delta_5min.items()):
            ax = axes[i]
            t = np.arange(len(profile)) / INTERVALS_PER_HOUR
            ax.plot(t, profile, linewidth=1.8, color=COLORS[i % len(COLORS)])
            ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title(f"RT DR Resource: {resource}", fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
            if i % ncols == 0:
                ax.set_ylabel(r"$\Delta$ Load (MW)")
            if i >= n - ncols:
                ax.set_xlabel("Time (hours)")
        
        # hide unused axes if n is odd
        for j in range(n, len(axes)):
            axes[j].axis("off")
        
        fig.suptitle("Real-Time Demand Response Profiles", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()
                
        
        
        
    for l, resource in enumerate(resource_names):
        profile = delta_5min[resource]
        duration = len(profile)
        last_start = min(op_end, T) - duration
        neg = np.where(profile < 0)[0]
        first_negative = neg[0] if len(neg) else 0
        if hourly_interrupt_only:
            target_residue = (-first_negative) % intervals_per_hour
            first_valid = op_start + ((target_residue - op_start) % intervals_per_hour)
            starts = range(first_valid, last_start + 1, intervals_per_hour)
        else:
            starts = range(op_start, last_start + 1, start_step)
        for start in starts:
            full = np.zeros(T)
            full[start:start + duration] = profile
            raw_profiles[l].append(full)
            raw_starts[l].append(start)

    n_starts_max = max(len(v) for v in raw_profiles.values())
    delta_ilt = {}
    valid_i = {}
    start_of = {}
    for l in L:
        n_real = len(raw_profiles[l])
        pad = n_starts_max - n_real
        delta_ilt[l] = raw_profiles[l] + [np.zeros(T) for _ in range(pad)]
        valid_i[l] = [True] * n_real + [False] * pad
        start_of[l] = raw_starts[l] + [None] * pad

    enumerated_profiles = np.array([delta_ilt[l] for l in L])

    return {
        "profiles": enumerated_profiles,
        "resource_names": resource_names,
        "valid_i": valid_i,
        "start_of": start_of,
        "n_starts": {l: n_starts_max for l in L},
    }