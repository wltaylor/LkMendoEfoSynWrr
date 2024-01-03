# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:09:08 2020

@author: cd
"""

import matplotlib.pyplot as plt
import scenario_builder as scenario
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

# Scenario name
name = 'pfo'
# Define simulation begin and end dates
dt_bgn = '1948-10-01 12:00'
dt_end = '2010-09-30 12:00'

# Load input files
res_params = scenario.get_res_params(dt_bgn, dt_end)
# Create model simulation
sim_pfo = scenario.build_scenario_efo(
    name=name,
    t_bgn=dt_bgn,
    t_end=dt_end,
    t_unit='D',
    stor_init=68400,
    res_params=res_params,
    hcst_lm=res_params['inp_q']['qLkMendocino'].to_numpy(),
    hcst_wf=res_params['inp_q']['qWestFork'].to_numpy(),
    hcst_hop=res_params['inp_q']['qHopland'].to_numpy(),
    fcst_horiz=336)
# Run Model
sim_pfo = sim_pfo.run_simulation()
# Model Results
results_df = scenario.create_results_df(
    simulation=sim_pfo, name=sim_pfo.name)
results_df.xs(name)['stor_mendo'].plot()
results_df.xs(name)['spill'].plot()
# Save Results
results_df.to_csv('results/testrun_results_pfo.csv')
