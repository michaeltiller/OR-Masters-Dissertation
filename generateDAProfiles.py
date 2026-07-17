import numpy as np
import matplotlib.pyplot as plt


    
def generateDAProfiles(archetypes, showPlot = True):
    INTERVALS_PER_HOUR = 12
    HOURS_PER_DAY = 24
    T = HOURS_PER_DAY * INTERVALS_PER_HOUR   #
    DT_MINUTES = 5



    SYSTEM_PEAK_HOUR = 17 

    #offset so peak reduction is at 6  
    
    for name, cfg in archetypes.items():
        peak_offset_hours = int(np.argmax(cfg["hourly_raw"]))   # deepest reduction, since sign flips later
        cfg["start_hour"] = SYSTEM_PEAK_HOUR - peak_offset_hours
        cfg["hourly"] = -cfg["hourly_raw"]
    
    
    def upsample_physical(profile, factor=INTERVALS_PER_HOUR,
                           tau_minutes=12, dt_minutes=DT_MINUTES):
        """First-order lag: approaches each hourly setpoint exponentially
        instead of jumping to it (thermal/electrical inertia)."""
        profile_ext = np.append(profile, 0)
        setpoint = np.repeat(profile_ext, factor)
        alpha = dt_minutes / tau_minutes
        y = np.zeros(len(setpoint))
        for t in range(1, len(setpoint)):
            y[t] = y[t - 1] + alpha * (setpoint[t] - y[t - 1])
        return y[:len(profile) * factor]
    
    
    
    da_profiles = {}
    for name, cfg in archetypes.items():
        hourly = cfg["hourly"]
        fine = upsample_physical(hourly, tau_minutes=cfg["tau_minutes"])
        start_slot = cfg["start_hour"] * INTERVALS_PER_HOUR
        full = np.zeros(T)
        end_slot = min(T, start_slot + len(fine))
        full[start_slot:end_slot] = fine[:end_slot - start_slot]
        da_profiles[name] = full
    
    scheduled_profiles_DA = np.array(list(da_profiles.values()))  # (n_archetypes, 288)
    labels = list(da_profiles.keys())
    
    if showPlot == True:
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57", "#6A4C93"]
        
        fig, ax = plt.subplots(figsize=(11, 5.5))
        hours_axis = np.arange(T) / INTERVALS_PER_HOUR
        for name, c in zip(labels, colors):
            ax.plot(hours_axis, da_profiles[name], color=c, linewidth=2, label=name)
        
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("\u0394Load (kW)")
        ax.set_title("Realistic Day-Ahead DR Profiles by Archetype")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, frameon=False, loc="lower left")
        
        plt.tight_layout()
        plt.show()

    return scheduled_profiles_DA, labels
