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
# Date of Creation: 2022/10/30                              #
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
bgn_idx = 1

dt_bgn = '1948-10-02 12:00'
dt_end = '1985-09-30 12:00'

# ----------------LOAD INPUT FILES-------------------
res_params = scenario.get_res_params(dt_bgn, dt_end)

results_df = pd.DataFrame()
for j in range(bgn_idx, bgn_idx + n_syn):
    syn_ens_file = f'inp/syn_hefs_prehcst/LkMendoSynHcst_corr_{ver}_{name}_samp{j}.npz'
    npzfile_hcst = np.load(syn_ens_file)
    hcst_lm = npzfile_hcst['syn_hcstLm'][:-1, :, :]
    hcst_wf = npzfile_hcst['syn_hcstWf'][:-1, :, :]
    hcst_hop = npzfile_hcst['syn_hcstHop'][:-1, :, :]
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
    cur_results = scenario.create_results_df(
        simulation=sim_syn, name=sim_syn.name)
    # curResults.to_csv(f'results/EfoResults_CornellSyn_corr_{simSyn.name}_{bgnIdx+nSyn}.csv')
    results_df = pd.concat((results_df, cur_results))

results_df.to_csv(f'results/testrun_results_cornell_syn_corr_hefs_prehind_all.csv')
