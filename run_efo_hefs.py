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
# Date of Creation: 2020/09/15                              #
#############################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scenario_builder as scenario
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

import gc

name = 'efo_cnrfc'

dt_bgn = '1985-10-02 12:00'
dt_end = '2010-09-30 12:00'

# ----------------LOAD INPUT FILES-------------------
res_params = scenario.get_res_params(dt_bgn, dt_end)

# Load Hindcasts
npzfile_hcst = np.load('inp/inp_Hcst_Cnrfc61_Daily.npz')
hcst_lm = npzfile_hcst['hcstLm']
hcst_wf = npzfile_hcst['hcstWf']
hcst_hop = npzfile_hcst['hcstHop']

sim_efo = scenario.build_scenario_efo(
    name=name,
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
sim_efo = sim_efo.run_simulation()
# Model Results
results_df = scenario.create_results_df(
    simulation=sim_efo, name=sim_efo.name)
results_df.xs(name)['stor_mendo'].plot()
results_df.xs(name)['spill'].plot()
# Save Results
results_df.to_csv('results/testrun_results_efo_hefs.csv')
