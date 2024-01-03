# -*- coding: utf-8 -*-

#############################################################
# Author: Chris Delaney <chris.delaney@scwa.ca.gov>         #
#                                                           #
#         Sonoma County Water Agency in collaboration with  #
#         Center for Western Weather and Water Extremes     #
#         Scripps Institution of Oceanography               #
#         UC San Diego                                      #
#                                                           #
#         https://github.com/hydrophile                     #
#         https://www.sonomawater.org/                      #
#         https://cw3e.ucsd.edu/                            #
#                                                           #
# Date of Creation: 2020/09/28                              #
#############################################################


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scenario_builder as scenario
from efo.simulation import Simulation
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)


name = 'sumrsamp30'
n_syn = 10
ver = 'v10'
# Simulation date range
dt_bgn = '1985-10-16 12:00'
dt_end = '2010-09-30 12:00'

# ----------------LOAD INPUT FILES-------------------
res_params = scenario.get_res_params(dt_bgn, dt_end)
results_list = []
for j in range(1, n_syn + 1):
    syn_ens_file = f'inp/syn_hefs/LkMendoSynHcst_corr_{ver}_{name}_samp{j}.npz'
    npzfile_hcst = np.load(syn_ens_file)
    hcst_lm = npzfile_hcst['syn_hcstLm']
    hcst_wf = npzfile_hcst['syn_hcstWf']
    hcst_hop = npzfile_hcst['syn_hcstHop']
    sim_syn = scenario.build_scenario_efo(
        name=f'syn_{ver}_{name}_{str(j)}',
        t_bgn=dt_bgn,
        t_end=dt_end,
        t_unit='D',
        stor_init=68400,
        res_params=res_params,
        hcst_lm=hcst_lm,
        hcst_wf=hcst_wf,
        hcst_hop=hcst_hop,
        fcst_horiz=336)
    # Run Model
    sim_syn = sim_syn.run_simulation()
    # Model Results
    results_df = scenario.create_results_df(
        simulation=sim_syn, name=sim_syn.name)
    # results_df.to_csv(f'results/testrun_results_efo_cornell_syn_hefs_{j}.csv')
    results_list.append(results_df)
results_all = pd.concat(results_list)
results_all.to_csv('results/testrun_results_efo_cornell_syn_hefs_all.csv')

