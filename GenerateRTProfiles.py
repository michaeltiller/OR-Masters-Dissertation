#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:45:48 2026
@author: michael
"""
import numpy as np
import matplotlib.pyplot as plt


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
    showPlot=True
):

    INTERVALS_PER_HOUR = 12
    HOURS_PER_DAY = 24
    T = HOURS_PER_DAY * INTERVALS_PER_HOUR
    DT_MINUTES = 5

    def upsample_physical(profile, tau):
        profile_ext = np.append(profile, 0)
        setpoint = np.repeat(profile_ext, intervals_per_hour)
        alpha = dt_minutes / tau
        y = np.zeros(len(setpoint))
        for t in range(1, len(setpoint)):
            y[t] = y[t-1] + alpha * (setpoint[t] - y[t-1])
        return y[:len(profile) * intervals_per_hour]

    delta_5min = {
        r: upsample_physical(delta_raw[r], tau_minutes[r])
        for r in delta_raw
    }

    resource_names = list(delta_5min.keys())
    L = range(len(resource_names))
    raw_profiles = {l: [] for l in L}
    raw_starts = {l: [] for l in L}

    if showPlot:
        fig, ax = plt.subplots(figsize=(10, 5))
        for resource, profile in delta_5min.items():
            t = np.arange(len(profile)) / INTERVALS_PER_HOUR  # hours
            ax.plot(t, profile, linewidth=2, label=resource)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Δ Load ")
        ax.set_title("Upsampled Physical Demand Response Profiles")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    for l, resource in enumerate(resource_names):
        profile = delta_5min[resource]
        duration = len(profile)
        last_start = min(op_end, T) - duration

        neg = np.where(profile < 0)[0]
        first_negative = neg[0] if len(neg) else 0

        if hourly_interrupt_only:
            # We need (start + first_negative) % intervals_per_hour == 0,
            # i.e. start % intervals_per_hour == (-first_negative) % intervals_per_hour.
            # Build the start grid directly on that residue instead of
            # filtering a step=start_step loop (which only visits a subset
            # of residues mod intervals_per_hour and can skip the target
            # residue entirely, yielding an all-empty profile set).
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