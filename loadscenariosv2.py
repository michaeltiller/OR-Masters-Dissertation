#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:23:19 2026

@author: michael
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def loadscenarios(n_scenarios, std_load):

    histLoads = pd.read_csv('Pricing data/hrl_load_metered.csv')
    histLoads = histLoads[['datetime_beginning_ept', 'mw']]
    histLoads.columns = ['TimeStamp', 'mw'] 
    
     
    histLoads['TimeStamp'] = pd.to_datetime(histLoads['TimeStamp'])
    histLoads['Date'] = histLoads['TimeStamp'].dt.date
    histLoads['Hour'] = histLoads['TimeStamp'].dt.hour
    
    
    specificDay = histLoads[histLoads['Date']== pd.Timestamp('2025-12-10').date()]
    
    
    load = np.array(specificDay['mw'])
    load = load/2
        
    
    n_initial = 3000
    
    scenarios_raw = np.random.normal(
        loc=load,                        
        scale=std_load * load,            
        size=(n_initial, 24)
    )
    
    
    km = KMeans(n_clusters=n_scenarios, random_state=42, n_init=10)
    
    km.fit(scenarios_raw)
    
    load_scenarios = km.cluster_centers_                   
    labels = km.labels_
    p = np.array([(labels == k).sum() / n_initial for k in range(n_scenarios)])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_scenarios))
    
    for color, s in zip(colors, range(n_scenarios)):
        ax.plot(
            range(24),
            load_scenarios[s],
            color=color,
            lw=2,
            alpha=0.75,
            label=f"Scenario {s} (p={p[s]:.2f})" )
    
    ax.plot(  range(24), load, color="black",  lw=3, linestyle="--",label="Forecast" )
    
    ax.set_title( "Generated Daily Load Scenarios", fontsize=18,  fontweight="bold",  pad=15, )
    
    ax.set_xlabel( "Hour of Day", fontsize=14,fontweight="bold" )
    
    ax.set_ylabel( "Electrical Load (kW)", fontsize=14, fontweight="bold", )
    
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(axis="both", labelsize=11)
    
    ax.grid(True, linestyle="--", alpha=0.3)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.legend(
        loc="upper left",
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )
    
    plt.tight_layout()
    plt.show()
        
    return p, load_scenarios