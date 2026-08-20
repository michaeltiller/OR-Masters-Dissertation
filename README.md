# Scheduling Demand Response Resources Under Net-load Uncertainty Using Multi-stage Stochastic Programming

This repository contains the main code for my MSc dissertation in Operational Research with Data Science at the University of Edinburgh. It schedules demand response (DR) resources under net-load uncertainty using a multi-stage stochastic programme, representing DR as a fixed set of enumerated start-time and duration profiles activated through binary decision variables.

## Notebook

- **DR scheduling.ipynb** — main notebook that the model is run in, builds the DR resource profiles and scenario tree, solves the model, and produces the some of the main results.

## Core Modules

- **scenarioswithWind.py** — builds the multi-stage scenario tree from historical load data, including AR(1) forecast-error noise, scenario-dependent peak timing, AI-driven ramping events, and net load constructed from historical wind generation.
- **drMSPModel_penalty_enumeratedDART.py** — the multi-stage stochastic programming model.
- **GenerateRTProfilesFlex.py** — generates the enumerated real-time DR resource profiles.
- **generateDAProfilesEnumerated - generates the enumerated day-ahead DR resource profiles.
- **plottingFuncsMSP.py** — plotting functions used to visualise model solutions and dispatch schedules.
- **reboundanalysis.py** — computes rebound stacking and offsetting between DR resources, used to diagnose how resource rebounds interact across the scheduling horizon.

## Requirements

- Pyomo
- HiGHS solver
- numpy
- pandas
- matplotlib


