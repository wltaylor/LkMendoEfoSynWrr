# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:16:07 2021

@author: cd
"""

import pandas as pd
import figs_syn as figs
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

dtBgn = '1985-10-16 12:00'
dtEnd = '2010-09-30 12:00'

# Load results
results_hefs = pd.read_csv(
    'results/results_efo_hefs.csv', index_col=[0, 1], parse_dates=True).loc[pd.IndexSlice[:, dtBgn:dtEnd], :]
results_pfo = pd.read_csv(
    'results/results_pfo.csv', index_col=[0, 1], parse_dates=True).loc[pd.IndexSlice[:, dtBgn:dtEnd], :]

# Load synthetic HEFS results
name_syn_hefs = 'sumrsamp30'
n_syn_files_hefs = 100
ver_syn_hefs = 'v10'
scen_key_hefs = f'syn_{ver_syn_hefs}_{name_syn_hefs}_'
results_syn_hefs = [pd.read_csv(
    f'results/results_{ver_syn_hefs}_{name_syn_hefs}_all.csv',
    index_col=[0, 1], parse_dates=True)]

# Load synthetic GEFS results
scen_key_gefs = 'efo_syn_gefs_v4_perturbed_'
n_syn_gefs = 10
path_gefs_results = 'results/efoResults_syn_gefs_v4_perturbed_'
results_syn_gefs = [None] * n_syn_gefs
for i in range(n_syn_gefs):
    results_syn_gefs[i] = pd.read_csv(path_gefs_results + str(i + 1) + '.csv', index_col=[0, 1], parse_dates=True)

# Compile into a single dataframe
results_comb = pd.concat((results_hefs, results_pfo))
# Index keys and legend
scenarios_comb = ['pfo', 'efo_cnrfc_new_topo']
legend_comb = ['PFO', 'HEFS']

fontSz = 8

fig = plt.figure(figsize=(7, 6), dpi=200)
gs = fig.add_gridspec(ncols=2, nrows=2)

# SYN-HEFS 
ax1a = fig.add_subplot(2, 2, 1)
ax2a = fig.add_subplot(2, 2, 3)
# Storage Exceedance
figs.plot_exceedance_stor(
    ax_stor_exc=ax1a,
    results_comb=results_comb,
    scen_comb=scenarios_comb,
    legend_comb=legend_comb,
    results_syn=results_syn_hefs,
    scen_syn=scen_key_hefs,
    syn_legend='syn-HEFS',
    fontsize=fontSz,
    lambda_exp=figs.CONV_TAF,
    units='TAF',
    show_xlabels=False,
    ymin=40000.,
)
# Plot release exceedance
figs.plot_exceedance_hop(
    ax_hop_exc=ax2a,
    results_comb=results_comb,
    scen_comb=scenarios_comb,
    legend_comb=legend_comb,
    results_syn=results_syn_hefs,
    scen_syn=scen_key_hefs,
    syn_legend='syn-HEFS',
    fontsize=fontSz,
    lambda_exp=figs.CONV_CFS2TAF,
    units='TAF/day',
)

# SYN-GEFS 
ax1b = fig.add_subplot(2, 2, 2)
ax2b = fig.add_subplot(2, 2, 4)
# Storage Exceedance
figs.plot_exceedance_stor(
    ax_stor_exc=ax1b,
    results_comb=results_comb,
    scen_comb=scenarios_comb,
    legend_comb=legend_comb,
    results_syn=results_syn_gefs,
    scen_syn=scen_key_gefs,
    syn_legend='syn-GEFS',
    fontsize=fontSz,
    lambda_exp=figs.CONV_TAF,
    units='TAF',
    show_xlabels=False,
    show_ylabels=False,
    fnc_syn_color=figs.line_color_syn_purple,
    ymin=40000.,
)

# Plot release exceedance
figs.plot_exceedance_hop(
    ax_hop_exc=ax2b,
    results_comb=results_comb,
    scen_comb=scenarios_comb,
    legend_comb=legend_comb,
    results_syn=results_syn_gefs,
    scen_syn=scen_key_gefs,
    syn_legend='syn-GEFS',
    fontsize=fontSz,
    lambda_exp=figs.CONV_CFS2TAF,
    units='TAF/day',
    show_ylabels=False,
    fnc_syn_color=figs.line_color_syn_purple,
)
fig.tight_layout()
plt.show()
fig.savefig(f'results/figs/SynManuscript_EFO_Exc_Fig08.png', dpi=600)
