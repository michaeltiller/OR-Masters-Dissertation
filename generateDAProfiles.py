import numpy as np
import matplotlib.pyplot as plt

def generateDAProfiles(archetypes, showPlot=True, noise_seed=None):
    INTERVALS_PER_HOUR = 12
    HOURS_PER_DAY = 24
    T = HOURS_PER_DAY * INTERVALS_PER_HOUR
    DT_MINUTES = 5
    SYSTEM_PEAK_HOUR = 17

    rng = np.random.default_rng(noise_seed)

    def ar1_noise(n, std, phi):
        eps_std = std * np.sqrt(1 - phi**2)  # keeps stationary variance ~= std
        eps = rng.normal(0, eps_std, size=n)
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = phi * noise[t-1] + eps[t]
        return noise

    for name, cfg in archetypes.items():
        peak_offset_hours = int(np.argmin(cfg["hourly_raw"]))
        #peak_offset_hours = int(np.argmax(cfg["hourly_raw"]))
        cfg["start_hour"] = SYSTEM_PEAK_HOUR - peak_offset_hours
        cfg["hourly"] = cfg["hourly_raw"]
        #cfg["hourly"] = -cfg["hourly_raw"]


    def upsample_physical(profile, factor=INTERVALS_PER_HOUR,
                       tau_minutes=12, dt_minutes=DT_MINUTES,
                       settle_tol=0.02):
        alpha = dt_minutes / tau_minutes
        decay = abs(1 - alpha)  # magnitude of per-step error decay, handles alpha >= 1
        if decay == 0:
            settle_steps = 1
        elif decay >= 1:
            settle_steps = factor  # doesn't converge in finite closed form; cap at 1hr
        else:
            settle_steps = int(np.ceil(np.log(settle_tol) / np.log(decay)))
        lead_steps = min(settle_steps, factor)  # cap at 1hr worth
    
        profile_ext = np.append(profile, 0)
        setpoint_core = np.repeat(profile_ext, factor)
        lead_setpoint = np.full(lead_steps, profile_ext[0])
        setpoint = np.concatenate([lead_setpoint, setpoint_core])
    
        y = np.zeros(len(setpoint))
        for t in range(1, len(setpoint)):
            y[t] = y[t - 1] + alpha * (setpoint[t] - y[t - 1])
    
        out_len = lead_steps + len(profile) * factor
        y = y[:out_len]
        setpoint = setpoint[:out_len]
    
        ref = np.max(np.abs(setpoint)) if np.max(np.abs(setpoint)) > 0 else 1.0
        settled = np.abs(setpoint - y) <= settle_tol * ref
        no_noise = settled & (setpoint < 0)
        noise_mask = ~no_noise
        return y, noise_mask

    da_profiles = {}
    for name, cfg in archetypes.items():
        hourly = cfg["hourly"]
        fine, noise_mask = upsample_physical(
            hourly,
            tau_minutes=cfg["tau_minutes"],
            settle_tol=cfg.get("settle_tol", 0.02),
        )

        std = cfg.get("noise_std", 0.0) * np.max(np.abs(hourly))
        if std > 0:
            phi = cfg.get("noise_phi", 0.7)
            noise = ar1_noise(len(fine), std, phi)
            fine = fine + noise * noise_mask

        start_slot = cfg["start_hour"] * INTERVALS_PER_HOUR
        full = np.zeros(T)
        end_slot = min(T, start_slot + len(fine))
        full[start_slot:end_slot] = fine[:end_slot - start_slot]
        da_profiles[name] = full

    scheduled_profiles_DA = np.array(list(da_profiles.values()))  # (n_archetypes, 288)
    labels = list(da_profiles.keys())

    if showPlot:
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57", "#6A4C93"]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        hours_axis = np.arange(T) / INTERVALS_PER_HOUR
        for name, c in zip(labels, colors):
            ax.plot(hours_axis, da_profiles[name], color=c, linewidth=2, label=name)

        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("\u0394Load (MW)")
        ax.set_title("Day-Ahead DR Profiles")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, frameon=False, loc="lower left")

        plt.tight_layout()
        plt.show()
    return scheduled_profiles_DA, labels