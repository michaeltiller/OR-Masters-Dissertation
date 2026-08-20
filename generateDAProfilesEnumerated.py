#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:47:43 2026

@author: michael
"""

import numpy as np
import matplotlib.pyplot as plt

def generateDAProfiles_Enumerated(
    archetypes,
    intervals_per_hour=12,
    hours_per_day=24,
    dt_minutes=5,
    start_step=6,      
    op_start_hour=6,
    op_end_hour=24,
    hourly_interrupt_only=False,
    showPlot=True,
    noise_seed=None,
    interrupt_lengths=None ):
    INTERVALS_PER_HOUR = intervals_per_hour
    T = hours_per_day * INTERVALS_PER_HOUR

    op_start = op_start_hour * INTERVALS_PER_HOUR
    op_end = op_end_hour * INTERVALS_PER_HOUR

    rng = np.random.default_rng(noise_seed)
    interrupt_lengths = interrupt_lengths or {}

    def ar1_noise(n, std, phi):
        eps_std = std * np.sqrt(1 - phi**2)
        eps = rng.normal(0, eps_std, size=n)
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + eps[t]
        return noise

    def upsample_physical(profile, tau_minutes, settle_tol=0.02, duration_hours=None):
        alpha = dt_minutes / tau_minutes
        decay = abs(1 - alpha)
        if decay == 0:
            settle_steps = 1
        elif decay >= 1:
            settle_steps = INTERVALS_PER_HOUR
        else:
            settle_steps = int(np.ceil(np.log(settle_tol) / np.log(decay)))
        lead_steps = min(settle_steps, INTERVALS_PER_HOUR)

        profile_ext = np.append(profile, 0)
        setpoint_core = np.repeat(profile_ext, INTERVALS_PER_HOUR)

        if duration_hours is not None:
            neg = np.where(setpoint_core < 0)[0]
            if len(neg):
                first = neg[0]
                cut = first + int(round(duration_hours * INTERVALS_PER_HOUR))
                cut = min(cut, len(setpoint_core))
                setpoint_core = setpoint_core.copy()
                setpoint_core[cut:] = 0.0

        lead_setpoint = np.full(lead_steps, profile_ext[0])
        setpoint = np.concatenate([lead_setpoint, setpoint_core])

        y = np.zeros(len(setpoint))
        for t in range(1, len(setpoint)):
            y[t] = y[t - 1] + alpha * (setpoint[t] - y[t - 1])

        out_len = lead_steps + len(profile) * INTERVALS_PER_HOUR
        y = y[:out_len]
        setpoint = setpoint[:out_len]

        ref = np.max(np.abs(setpoint)) if np.max(np.abs(setpoint)) > 0 else 1.0
        settled = np.abs(setpoint - y) <= settle_tol * ref
        no_noise = settled & (setpoint < 0)
        noise_mask = ~no_noise
        return y, noise_mask

    raw_by_resource_duration = {}
    for name in archetypes:
        lengths = interrupt_lengths.get(name)
        raw_by_resource_duration[name] = list(lengths) if lengths else ["full"]

    fine_profiles = {}
    for name, cfg in archetypes.items():
        hourly = cfg["hourly_raw"]
        for dlabel in raw_by_resource_duration[name]:
            duration_hours = None if dlabel == "full" else dlabel
            fine, noise_mask = upsample_physical(
                hourly, tau_minutes=cfg["tau_minutes"],
                settle_tol=cfg.get("settle_tol", 0.02),
                duration_hours=duration_hours,
            )
            std = cfg.get("noise_std", 0.0) * np.max(np.abs(hourly))
            if std > 0:
                phi = cfg.get("noise_phi", 0.7)
                noise = ar1_noise(len(fine), std, phi)
                fine = fine + noise * noise_mask
            fine_profiles[(name, dlabel)] = fine

    resource_names = list(archetypes.keys())
    J = range(len(resource_names))
    raw_profiles = {j: [] for j in J}
    raw_starts = {j: [] for j in J}
    raw_durations = {j: [] for j in J}  # interrupt-length label per i, parallel to raw_starts

    for j, name in enumerate(resource_names):
        for dlabel in raw_by_resource_duration[name]:
            profile = fine_profiles[(name, dlabel)]
            duration = len(profile)
            last_start = min(op_end, T) - duration
            neg = np.where(profile < 0)[0]
            first_negative = neg[0] if len(neg) else 0
            if hourly_interrupt_only:
                target_residue = (-first_negative) % INTERVALS_PER_HOUR
                first_valid = op_start + ((target_residue - op_start) % INTERVALS_PER_HOUR)
                starts = range(first_valid, last_start + 1, INTERVALS_PER_HOUR)
            else:
                starts = range(op_start, last_start + 1, start_step)
            for start in starts:
                full = np.zeros(T)
                full[start:start + duration] = profile
                raw_profiles[j].append(full)
                raw_starts[j].append(start)
                raw_durations[j].append(dlabel)

    n_starts_max = max(len(v) for v in raw_profiles.values())
    delta_jit = {}
    valid_i = {}
    start_of = {}
    duration_of = {}
    for j in J:
        n_real = len(raw_profiles[j])
        pad = n_starts_max - n_real
        delta_jit[j] = raw_profiles[j] + [np.zeros(T) for _ in range(pad)]
        valid_i[j] = [True] * n_real + [False] * pad
        start_of[j] = raw_starts[j] + [None] * pad
        duration_of[j] = raw_durations[j] + [None] * pad

    enumerated_profiles = np.array([delta_jit[j] for j in J])

    if showPlot:
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57", "#6A4C93"]
        hours_axis = np.arange(T) / INTERVALS_PER_HOUR
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for j, name in enumerate(resource_names):
            variants = raw_profiles[j]
            representative = variants[len(variants) // 2]
            ax.plot(hours_axis, representative, color=colors[j % len(colors)],
                    linewidth=1.5, label=name)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xlim(0, 24)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("$\\Delta$Load (MW)")
        ax.set_title("Day-Ahead DR Profiles", fontweight="bold")
        ax.legend(title="Resource", frameon=False)
        plt.tight_layout()
        plt.show()

    return {
        "profiles": enumerated_profiles,
        "resource_names": resource_names,
        "valid_i": valid_i,
        "start_of": start_of,
        "duration_of": duration_of,   # interrupt length (hours, fractional OK) per (j,i)
        "n_starts": {j: n_starts_max for j in J},
    }