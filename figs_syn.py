# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 06:27:06 2021

@author: cd
"""
import pandas as pd
import numpy as np
from efo.time import TimeCont
from efo.hypso import Hypso
import efo.figures as efo_fig
import efo.figures_fcst as fcst_fig
import matplotlib.pyplot as plt

CONV_CFS2TAF = lambda y, pos: (f'{y * 1.983471 / 1000.0: .0f}')
CONV_TAF = lambda y, pos: (f'{y / 1000.0: .0f}')

fontSz = 4.7
# Index keys and legend
# scenariosPfoCnrfc = ['pfo', 'efo_cnrfc_new_topo']
# legendPfoCnrfc = ['PFO', 'CNRFC']
# Create hypsometry object
inp_lm_hypso = pd.read_csv('inp/inp_LkMendo_Hypso.csv')
hypso_lm = Hypso('hypsoLm', inp_lm_hypso['Elev'].to_numpy(),
                 inp_lm_hypso['Storage'].to_numpy(), inp_lm_hypso['Area'].to_numpy())


def line_color_syn_orange(shade):
    color_ramp = plt.cm.Oranges(np.linspace(0, 1, 11))
    return color_ramp[shade]


def line_color_syn_purple(shade):
    colorRamp = plt.cm.PuRd(np.linspace(0, 1, 11))
    return colorRamp[shade]


# def plotColorsSyn(idx, nLines):
#     return stdColors[1]

# synAlpha = lambda nLines: 3./nLines
# synAlpha = lambda nLines: 1.
std_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = [std_colors[2], std_colors[0]]
std_shade = 5
std_alpha = 0.4


def plot_storage_TS(ax_stor, results_comb, scen_comb, legend_comb, results_syn,
                    scen_syn, hypso, fontsize, shade=std_shade, alpha=std_alpha,
                    ymin=0., show_elev=True, show_title=True, lambda_exp=None,
                    parameter='storage (ac-ft)', plot_title='Lake Mendocino Storage',
                    syn_legend='Synthetic',
                    fnc_syn_color=line_color_syn_orange):
    plot_title = plot_title if show_title else None
    # spill_crest_stor = np.full(resultsComb.xs(scenComb[0]).shape[0], 116500)
    # spill_crest_df = pd.DataFrame(
    #     spill_crest_stor, columns=['spill_crest'], index=resultsComb.xs(scenComb[0]).index)
    # spill_crest_df.index.name = 'date_time'
    # efoFig.plot_time_series(ax=axStor, df=spill_crest_df,
    #                             col='spill_crest',
    #                             legend=['Spillway Crest'],
    #                             lineStyle=':',
    #                             lineWidth=1,
    #                             lineColor=['k'],
    #                             fontSz=fontSz,
    #                             yMin = yMin,
    #                             lambdaExp = lambdaExp)
    # Synthetic results
    n_line = 0
    for curDf in results_syn:
        for ii, cur_scen in enumerate(curDf.index.get_level_values(0).unique()):
            if n_line == 0:
                cur_legend = [syn_legend]
            else:
                cur_legend = None
            n_line += 1
            efo_fig.plot_time_series(ax=ax_stor, df=curDf,
                                     col='stor_mendo',
                                     parameter=parameter,
                                     scenarios=[cur_scen],
                                     legend=cur_legend,
                                     fontsize=fontsize,
                                     linewidth=1.,
                                     ymax=140000,
                                     linecolor=[fnc_syn_color(shade)],
                                     alpha=alpha,
                                     ymin=ymin,
                                     y_axis_fnc=lambda_exp,
                                     legend_loc='lower left', )
    # Plot CNRFC Results
    if show_elev:
        efo_fig.plot_hydrograph_stor_elev(ax=ax_stor, df=results_comb,
                                          col='stor_mendo',
                                          scenarios=scen_comb,
                                          title=plot_title,
                                          legend=legend_comb,
                                          hypso=hypso,
                                          linewidth=1.2,
                                          fontsize=fontsize,
                                          linecolor=color_cycle,
                                          ymin=ymin,
                                          y_axis_fnc=lambda_exp,
                                          legend_loc='lower left')
    else:
        efo_fig.plot_time_series(ax=ax_stor, df=results_comb,
                                 col='stor_mendo',
                                 scenarios=scen_comb,
                                 title=plot_title,
                                 legend=legend_comb,
                                 legend_loc='lower left',
                                 linewidth=1.2,
                                 fontsize=fontsize,
                                 linecolor=color_cycle,
                                 ymin=ymin,
                                 y_axis_fnc=lambda_exp,
                                 )
    # Plot spill crest
    spill_crest_stor = np.full(results_comb.xs(scen_comb[0]).shape[0], 116500)
    spill_crest_df = pd.DataFrame(
        spill_crest_stor, columns=['spill_crest'], index=results_comb.xs(scen_comb[0]).index)
    spill_crest_df.index.name = 'date_time'
    efo_fig.plot_time_series(ax=ax_stor, df=spill_crest_df,
                             col='spill_crest',
                             legend=['Spillway: 116.5 TAF'],
                             linestyle='--',
                             linewidth=1.2,
                             linecolor=['k'],
                             fontsize=fontsize,
                             ymin=ymin,
                             ymax=150000.,
                             y_axis_fnc=lambda_exp,
                             legend_loc='lower left')
    return


def plot_exceedance(ax_exc, results_comb, scen_comb, legend_comb, results_syn, scenSyn, fontsize,
                    col, param, title, ymin=None, ymax=None, x_axis_scale='linear', syn_legend='Synthetic',
                    shade=std_shade, alpha=std_alpha, lambda_exp=None, fnc_syn_color=line_color_syn_orange,
                    show_xlabels=True, show_ylabels=True):
    # Synthetic results
    n_line = 0
    for curDf in results_syn:
        for i, cur_scen in enumerate(curDf.index.get_level_values(0).unique()):
            # if i == len(resultsSyn)-1:
            if n_line == 0:
                curLegend = [syn_legend]
            else:
                curLegend = None
            n_line += 1
            efo_fig.plot_exceedance(ax=ax_exc, df=curDf,
                                    col=col,
                                    parameter=param,
                                    scenarios=[cur_scen],
                                    legend=curLegend,
                                    fontsize=fontsize,
                                    linewidth=1.,
                                    linecolor=[fnc_syn_color(shade)],
                                    alpha=alpha,
                                    y_axis_fnc=lambda_exp,
                                    show_xlabels=show_xlabels,
                                    show_ylabels=show_ylabels,
                                    )
    efo_fig.plot_exceedance(ax=ax_exc, df=results_comb,
                            col=col,
                            parameter=param,
                            scenarios=scen_comb,
                            title=title,
                            legend=legend_comb,
                            fontsize=fontsize,
                            x_axis_scale=x_axis_scale,
                            linecolor=color_cycle,
                            linewidth=1.6,
                            ymin=ymin,
                            ymax=ymax,
                            y_axis_fnc=lambda_exp,
                            show_xlabels=show_xlabels,
                            show_ylabels=show_ylabels,
                            )
    return


# Storage Exceedance figure
def plot_exceedance_stor(ax_stor_exc, results_comb, scen_comb, legend_comb, results_syn,
                         scen_syn, fontsize, ymin=None, syn_legend='Synthetic', lambda_exp=None,
                         fnc_syn_color=line_color_syn_orange, units='ac-ft',
                         show_xlabels=True, show_ylabels=True, show_title=True,
                         title='Lake Mendocino Storage'):
    if show_title is None: title = None
    spill_crest_stor = np.full(results_comb.xs(scen_comb[0]).shape[0], 116500)
    spill_crest_df = pd.DataFrame(
        spill_crest_stor, columns=['spill_crest'], index=results_comb.xs(scen_comb[0]).index)
    spill_crest_df.index.name = 'date_time'
    efo_fig.plot_exceedance(ax=ax_stor_exc, df=spill_crest_df,
                            col='spill_crest',
                            legend=['Spillway: 116.5 TAF'],
                            linestyle='--',
                            linewidth=1.3,
                            linecolor=['k'],
                            fontsize=fontsize,
                            y_axis_fnc=lambda_exp,
                            show_xlabels=show_xlabels,
                            show_ylabels=show_ylabels,
                            )
    plot_exceedance(ax_stor_exc, results_comb, scen_comb, legend_comb, results_syn, scen_syn, fontsize,
                    col='stor_mendo',
                    param=f'storage ({units})',
                    title=title,
                    ymax=120000,
                    syn_legend=syn_legend,
                    fnc_syn_color=fnc_syn_color,
                    lambda_exp=lambda_exp,
                    # showXlabels=showXlabels,
                    show_ylabels=show_ylabels,
                    ymin=ymin,
                    )
    if not (show_xlabels):
        ax_stor_exc.set_xlabel(None)
    ax_stor_exc.xaxis.grid(True, which='major', linestyle=':')
    ax_stor_exc.yaxis.grid(True, which='major', linestyle=':')
    return


# Hopland flow Exceedance figure
def plot_exceedance_hop(ax_hop_exc, results_comb, scen_comb, legend_comb, results_syn,
                        scen_syn, fontsize, syn_legend='Synthetic', lambda_exp=None,
                        fnc_syn_color=line_color_syn_orange, units='cfs',
                        show_xlabels=True, show_ylabels=True, show_title=True):
    title = 'Hopland Flow' if show_title else None
    spill_crest_stor = np.full(results_comb.xs(scen_comb[0]).shape[0], 8000.)
    spill_crest_df = pd.DataFrame(
        spill_crest_stor, columns=['flood_stage'], index=results_comb.xs(scen_comb[0]).index)
    spill_crest_df.index.name = 'date_time'
    efo_fig.plot_exceedance(ax=ax_hop_exc, df=spill_crest_df,
                            col='flood_stage',
                            legend=['NWS Flood Stage'],
                            linestyle='--',
                            linewidth=1.3,
                            linecolor=['k'],
                            fontsize=fontsize,
                            x_axis_scale='log',
                            y_axis_fnc=lambda_exp,
                            show_xlabels=show_xlabels,
                            show_ylabels=show_ylabels,
                            )
    plot_exceedance(ax_hop_exc, results_comb, scen_comb, legend_comb, results_syn, scen_syn, fontsize,
                    col='q_hop',
                    param=f'flow ({units})',
                    title=title,
                    x_axis_scale='log',
                    syn_legend=syn_legend,
                    fnc_syn_color=fnc_syn_color,
                    lambda_exp=lambda_exp,
                    # showXlabels=showXlabels,
                    show_ylabels=show_ylabels,
                    )
    if not (show_xlabels):
        ax_hop_exc.set_xlabel(None)
    ax_hop_exc.xaxis.grid(True, which='major', linestyle=':')
    ax_hop_exc.yaxis.grid(True, which='major', linestyle=':')
    return


def plot_syn_flow_batch(ax_flow, results_syn, scen_syn, fontsize,
                        col, param=None, title=None, ymax=None, linestyle='-',
                        linewidth=1.2, syn_legend='Synthetic', shade=std_shade, alpha=std_alpha,
                        fnc_syn_color=line_color_syn_orange, xTickRotation=0):
    # if synLegend is None: synLegend = 'Synthetic'
    # Synthetic results
    n_line = 0
    for cur_df in results_syn:
        for i, curScen in enumerate(cur_df.index.get_level_values(0).unique()):
            if n_line == 0:
                curLegend = [syn_legend]
            else:
                curLegend = None
            n_line += 1
            ax_flow = efo_fig.plot_time_series(
                ax=ax_flow, df=cur_df,
                col=col,
                parameter=param,
                scenarios=[curScen],
                title=title,
                legend=curLegend,
                fontsize=fontsize,
                linewidth=linewidth,
                linecolor=[fnc_syn_color(shade)],
                linestyle=linestyle,
                ymax=ymax,
                alpha=alpha,
                # xAxDateFormat='%y-%m-%d',
                x_axis_date_format='%d-%b\n%Y',
                xtick_rotation=xTickRotation,
            )
    return ax_flow


# Flow
def plot_flow(ax_flow, results_comb, scen_comb, legend_comb, results_syn, scen_syn,
              col, param=None, title=None, ymax=None, linestyle='-', syn_legend='Synthetic',
              shade=4, alpha=0.3, lambda_exp=None, fnc_syn_color=line_color_syn_orange, xtick_rotation=0,
              fontsize=fontSz):
    # Synthetic results
    ax_flow = plot_syn_flow_batch(
        ax_flow=ax_flow,
        results_syn=results_syn,
        scen_syn=scen_syn,
        fontsize=fontsize,
        col=col,
        param=param,
        syn_legend=syn_legend,
        linestyle=linestyle,
        shade=shade,
        alpha=alpha,
        fnc_syn_color=fnc_syn_color,
        xTickRotation=xtick_rotation,
    )
    ax_flow = efo_fig.plot_time_series(ax=ax_flow, df=results_comb,
                                       col=col,
                                       parameter=param,
                                       scenarios=scen_comb,
                                       title=title,
                                       legend=legend_comb,
                                       legend_loc='upper right',
                                       fontsize=fontsize,
                                       linestyle=linestyle,
                                       linecolor=color_cycle,
                                       y_axis_fnc=lambda_exp,
                                       linewidth=1.2,
                                       xtick_rotation=xtick_rotation,
                                       )
    return ax_flow


# Hopland Flow
def plot_hop_flow(ax_flow, results_comb, scen_comb, legend_comb, results_syn, scen_syn, fontsize):
    # plot_color_syn_orange = np.flip(plt.cm.summer(np.linspace(0,1,len(resultsSyn)+2)[1:len(resultsSyn)+1]), axis=0)
    # Flood Stage
    q_flood = np.full(results_comb.xs(scen_comb[0]).shape[0], 8000)
    q_flood_df = pd.DataFrame(q_flood, columns=['flood_stage'], index=results_comb.xs(scen_comb[0]).index)
    q_flood_df.index.name = 'date_time'
    efo_fig.plot_time_series(ax=ax_flow, df=q_flood_df,
                             col='flood_stage',
                             legend=['NWS Flood Stage'],
                             linestyle=':',
                             linewidth=1,
                             linecolor=['k'],
                             )
    plot_flow(ax_flow, results_comb, scen_comb, legend_comb, results_syn, scen_syn, fontsize,
              col='q_hop',
              param='flow (cfs)',
              title='Russian River at Hopland Flow',
              fontsize=fontsize,
              )
    return


def create_detail_hydrographs(dt_bgn, dt_end, results_comb, scen_comb, legend_comb,
                              results_syn, scen_syn, qin_df, hypso_lm, fontsize, month, year):
    # Filter data for date period
    dt_results = results_comb.index.get_level_values(1)
    results_comb = results_comb.loc[(dt_results >= dt_bgn) & (dt_results <= dt_end)]
    # resultsComb = resultsComb.loc[dtBgn:dtEnd]
    qin_df = qin_df.loc[(qin_df.index >= dt_bgn) & (qin_df.index <= dt_end)]
    df_syn = []
    for cur_syn in results_syn:
        dt_syn = cur_syn.index.get_level_values(1)
        i_syn_per = (dt_syn >= dt_bgn) & (dt_syn <= dt_end)
        df_syn.append(cur_syn[i_syn_per])
    fig = plt.figure(figsize=(6.5, 8), dpi=200, constrained_layout=True)
    # fig = plt.figure(figsize=(6.5, 8), dpi=200)
    gs = fig.add_gridspec(ncols=1, nrows=3)
    # Plot storage
    ax_stor = fig.add_subplot(gs[0, 0])
    plot_storage_TS(ax_stor, results_comb, scen_comb, legend_comb, df_syn, scen_syn, hypso_lm, fontsize)
    ax_stor.set_xticklabels([])
    ax_stor.set_xticklabels([], minor=True)
    # Controlled Release
    legend_ctrl_rls = [curLegend + ' Control Release' for curLegend in legend_comb]
    ax_rls = fig.add_subplot(gs[1, 0])
    plot_syn_flow_batch(
        ax_flow=ax_rls,
        results_syn=results_syn,
        scen_syn=scen_syn,
        fontsize=fontsize,
        col='rls_ctrl',
        syn_legend=' Control Release')
    # plot_flow(ax_rls, resultsComb, scenComb, legend_ctrl_rls, df_syn, scenSyn, fontSz,
    #           col = 'rls_ctrl', 
    #           synLegend = ' Control Release'
    #           )
    legend_spill = [curLegend + ' Spill' for curLegend in legend_comb]
    plot_flow(ax_rls, results_comb, scen_comb, legend_spill, df_syn, scen_syn, fontsize,
              col='spill',
              syn_legend=' Spill',
              linestyle='--',
              shade=std_shade + 2,
              alpha=1.
              )
    efo_fig.plot_time_series(
        ax=ax_rls, df=results_comb,
        col='rls_ctrl',
        scenarios=scen_comb,
        legend=legend_ctrl_rls,
        legend_loc='upper right',
        fontsize=fontsize,
        linestyle='-',
        linecolor=color_cycle)
    # Inflow
    efo_fig.plot_time_series(ax=ax_rls, df=qin_df,
                             col='qLkMendocino',
                             parameter='flow (cfs)',
                             title='Lake Mendocino Release and Inflow',
                             legend=['Observed Inflow'],
                             linestyle=':',
                             linewidth=1.3,
                             linecolor=['m'],
                             legend_loc='upper left',
                             fontsize=fontsize)
    ax_rls.set_xticklabels([])
    ax_rls.set_xticklabels([], minor=True)
    # Hopland Flow
    ax_hop = fig.add_subplot(gs[2, 0])
    plot_hop_flow(ax_hop, results_comb, scen_comb, legend_comb, df_syn, scen_syn, fontsize)
    title = month + ' ' + year + ' Flood Event\n'
    fig.suptitle(title, fontsize=fontsize + 1)
    # fig.tight_layout()
    # plt.show
    return fig


def create_detail_manuscript(
        ax_stor, ax_rls, dt_bgn, dt_end, results_comb, scen_comb, legend_comb,
        results_syn, scen_syn, qin_df, syn_legend, hypso_lm, fontsize,
        show_stor=True, show_elev=False, show_flow=True,
        fnc_syn_color=line_color_syn_orange):
    # Filter data for date period
    dtResults = results_comb.index.get_level_values(1)
    dfPerComb = results_comb.loc[(dtResults >= dt_bgn) & (dtResults <= dt_end)]
    qin_df = qin_df.loc[(qin_df.index >= dt_bgn) & (qin_df.index <= dt_end)]
    dfSyn = []
    for curSyn in results_syn:
        dtSyn = curSyn.index.get_level_values(1)
        iSynPer = (dtSyn >= dt_bgn) & (dtSyn <= dt_end)
        dfSyn.append(curSyn[iSynPer])
    # Plot storage
    plot_storage_TS(
        ax_stor, dfPerComb, scen_comb, legend_comb, dfSyn, scen_syn, hypso_lm, fontsize,
        show_elev=show_elev,
        show_title=False,
        lambda_exp=CONV_TAF,
        parameter='storage (TAF)',
        fnc_syn_color=fnc_syn_color,
        syn_legend=syn_legend,
    )
    ax_stor.set_xticklabels([])
    ax_stor.set_xticklabels([], minor=True)
    if not (show_stor):
        ax_stor.set_yticklabels([])
        ax_stor.set_yticklabels([], minor=True)
        ax_stor.set_ylabel(None)
    # Controlled Release
    legendCtrlRls = [curLegend + ' Release' for curLegend in legend_comb]
    plot_flow(ax_rls, dfPerComb, scen_comb, legendCtrlRls, dfSyn, scen_syn,
              fontsize=fontsize,
              col='rls_ctrl',
              syn_legend=syn_legend + ' Release',
              linestyle='-',
              fnc_syn_color=fnc_syn_color,
              xtick_rotation=0,
              )
    # Inflow
    efo_fig.plot_time_series(ax=ax_rls, df=qin_df,
                             col='qLkMendocino',
                             # parameter='flow (cfs)',
                             # title='Lake Mendocino Release and Inflow',
                             legend=['Observed Inflow'],
                             linestyle=efo_fig.DENSE_DOT,
                             linewidth=1.2,
                             linecolor=['k'],
                             legend_loc='upper left',
                             fontsize=fontsize,
                             y_axis_fnc=CONV_CFS2TAF,
                             parameter='flow (TAF/day)')
    if not (show_flow):
        ax_rls.set_yticklabels([])
        ax_rls.set_yticklabels([], minor=True)
        ax_rls.set_ylabel(None)
    # axRls.xaxis.set_tick_params(rotation = 45)
    # for label in axRls.xaxis.get_majorticklabels():
    #     label.set_horizontalalignment('right')
    # axRls.set_xticklabels([])
    # axRls.set_xticklabels([], minor=True)
    return ax_stor, ax_rls


def create_lead_marker(ax, leadDay, dtStr, fontSz):
    ax.vlines(x=np.datetime64(dtStr),
              ymin=0., ymax=4050., color='k', linestyle='--', linewidth=1.)
    ax.text(np.datetime64(dtStr), 4400., str(leadDay), horizontalalignment='center',
            verticalalignment='center', fontsize=fontSz)


def create_hydro_panels(
        ax1, ax2, ax3, plot_title, dt_bgn, dt_end, dt_peak, dt_leads,
        scen_key, scenarios_comb, legend_comb, results_comb,
        results_syn, qin_df, syn_legend, show_stor=True, show_flow=True, annotate_ha='left',
        fnc_syn_color=line_color_syn_orange, showTitle=False):
    # Storage time series
    plot_storage_TS(
        ax_stor=ax1,
        results_comb=results_comb,
        scen_comb=scenarios_comb,
        legend_comb=legend_comb,
        results_syn=results_syn,
        scen_syn=scen_key,
        hypso=hypso_lm,
        fontsize=fontSz,
        ymin=0,
        show_title=showTitle,
        show_elev=False,
        lambda_exp=CONV_TAF,
        plot_title=plot_title,
        parameter='storage (TAF)',
        fnc_syn_color=fnc_syn_color,
        syn_legend=syn_legend,
    )
    # storPeak = resultsComb.xs(scenariosComb[1])['stor_mendo'][dtPeak:dtPeak]
    x_anotate = np.datetime64(dt_peak)
    arrowprops = {
        'arrowstyle': '->',
        'color': 'r'}
    t_delta = np.timedelta64(1000, 'D') if annotate_ha == 'left' else -np.timedelta64(1000, 'D')
    ax1.annotate(
        text=pd.Timestamp(dt_peak).strftime('%Y-%m-%d'),
        xy=(x_anotate, 117500.),
        xytext=(x_anotate + t_delta, 135000.),
        arrowprops=arrowprops, fontsize=fontSz + 1.5,
        horizontalalignment=annotate_ha,
        verticalalignment='center',
        color='r')
    if not (show_stor):
        ax1.set_yticklabels([])
        ax1.set_yticklabels([], minor=True)
        ax1.set_ylabel(None)
    # Detail event
    ts = pd.Timestamp(dt_peak)
    create_detail_manuscript(
        ax_stor=ax2,
        ax_rls=ax3,
        dt_bgn=dt_bgn,
        dt_end=dt_end,
        results_comb=results_comb,
        scen_comb=scenarios_comb,
        legend_comb=legend_comb,
        results_syn=results_syn,
        scen_syn=scen_key,
        qin_df=qin_df,
        hypso_lm=hypso_lm,
        fontsize=fontSz,
        show_stor=show_stor,
        show_flow=show_flow,
        fnc_syn_color=fnc_syn_color,
        syn_legend=syn_legend,
    )
    for cur_key in dt_leads.keys():
        create_lead_marker(ax3, cur_key, dt_leads[cur_key], fontSz + 1)


def plot_ensemble_mean(ax, dt_syn, n_syn_hcst, path_syn_hcst, key, dt_init,
                       ymax, linecolor, linelabel,
                       key_dt=None, show_legend=False):
    # dtSyn = pd.date_range(start='1985-10-16 12:00', end='2010-09-30 12:00', freq='D')
    if dt_syn is not None: rowSyn = np.where(dt_syn == dt_init)[0][0]
    list_syn = []
    for i in range(1, n_syn_hcst + 1):
        # synEnsFile = f'inp/syn_ens/{name}/LkMendoSynHcst_{ver}_{name}_samp{i}.npz'
        npzfileHcst = np.load(path_syn_hcst + str(i) + '.npz')
        scstLm = npzfileHcst[key][:-1, :, :]
        if key_dt:
            dt_syn = npzfileHcst[key_dt][:-1]
            rowSyn = np.where(dt_syn == dt_init)[0][0]
        npzfileHcst.close()
        list_syn.append(scstLm[rowSyn, :, :])
    mean_syn = np.mean(np.concatenate(list_syn, axis=1), axis=1)
    fcst_fig.plot_fcst_hydro(
        ax=ax,
        t_fcst=pd.date_range(start=dt_init, end=dt_init + pd.Timedelta(days=14), freq='D'),
        fcst_vals=mean_syn,
        linelabel=linelabel,
        linecolor=linecolor,
        linestyle='--',
        legend_loc='upper right',
        fontsize=fontSz,
        ymax=ymax,
        y_axis_fnc=CONV_CFS2TAF,
        ylabel='flow (TAF/day)',
        time_interval=48,
        linewidth=1.2,
        x_axis_date_format='%m-%d',
        show_legend=show_legend)


def plot_ensemble_fcst(
        ax, dt_init, dt_peak, days_lead, n_syn_hefs, path_syn_hefs, dt_syn_hefs, n_syn_gefs,
        path_syn_gefs, dt_syn_gefs, hcst_lm, dt_hcst, pcst_lm, dt_pcst, ymax,
        show_flow=True, key_dt=None, plot_mean_hefs=True, hefs_prefix='', show_legend=False):
    dt_init = pd.Timestamp(dt_init)
    # Get current hindcasts
    cur_hcst = hcst_lm[np.where(dt_hcst == dt_init)[0][0], :, :]
    # Get current pcst
    curPcst = pcst_lm[np.where(dt_pcst == dt_init)[0][0], :15]
    # Set day 0 of current HCST to PCST
    cur_hcst[0, :] = curPcst[0]
    # Plot CNRFC Ensemble
    fcst_fig.plot_fcst_ens(
        ax=ax,
        t_fcst=pd.date_range(start=dt_init, end=dt_init + pd.Timedelta(days=14), freq='D'),
        # fcstResults = hcstLm[np.where(dtHcst==dtInit)[0][0], :, :],
        fcst_results=cur_hcst,
        color_ovrd='silver',
        ymax=ymax,
        linewidth=0.7,
        time_interval=48,
        fontsize=fontSz,
        x_axis_date_format='%m-%d',
        y_axis_fnc=CONV_CFS2TAF)
    # Lead day vline
    ax.vlines(x=np.datetime64(dt_peak),
              ymin=0., ymax=ymax, color='r', linestyle=fcst_fig.DENSE_DOT, linewidth=1.)
    plt.annotate(
        text='', xy=(np.datetime64(dt_peak), ymax - ymax * 0.13),
        xytext=(dt_init, ymax - ymax * 0.13), arrowprops=dict(arrowstyle='<->'), fontsize=fontSz)
    ax.text(
        np.datetime64(dt_init) + (np.datetime64(dt_peak) - np.datetime64(dt_init)) / 2, ymax - ymax * 0.1,
        f'{str(days_lead)} days', horizontalalignment='center',
        verticalalignment='center', fontsize=fontSz)
    # Plot observed
    fcst_fig.plot_fcst_hydro(
        ax=ax,
        t_fcst=pd.date_range(start=dt_init, end=dt_init + pd.Timedelta(days=14), freq='D'),
        # fcstVals = pcstLm[np.where(dtPcst==dtInit)[0][0], :15], 
        fcst_vals=curPcst,
        linelabel='observed',
        linecolor='k',
        ymax=ymax,
        linewidth=1.2,
        time_interval=48,
        fontsize=fontSz,
        legend_loc='upper right',
        x_axis_date_format='%m-%d',
        y_axis_fnc=CONV_CFS2TAF,
        show_legend=show_legend)
    # Plot HEFS Mean
    if plot_mean_hefs:
        fcst_fig.plot_fcst_hydro(
            ax=ax,
            t_fcst=pd.date_range(start=dt_init, end=dt_init + pd.Timedelta(days=14), freq='D'),
            fcst_vals=np.mean(hcst_lm[np.where(dt_hcst == dt_init)[0][0], :, :], axis=1),
            linelabel=hefs_prefix + 'HEFS mean',
            linecolor='b',
            linestyle='--',
            ymax=ymax,
            linewidth=1.2,
            time_interval=48,
            fontsize=fontSz,
            legend_loc='upper right',
            x_axis_date_format='%m-%d',
            y_axis_fnc=CONV_CFS2TAF,
            show_legend=show_legend
        )
    plot_ensemble_mean(
        ax, dt_syn_hefs, n_syn_hefs, path_syn_hefs, 'syn_hcstLm', dt_init, ymax,
        'coral', 'syn-HEFS mean', show_legend=show_legend)
    if path_syn_gefs is not None:
        plot_ensemble_mean(
            ax, dt_syn_gefs, n_syn_gefs, path_syn_gefs, 'hcstLm', dt_init, ymax, 'm',
            'syn-GEFS mean', key_dt=key_dt, show_legend=show_legend)
    if not (show_flow):
        ax.set_yticklabels([])
        ax.set_yticklabels([], minor=True)
        ax.set_ylabel(None)
    idxTick = int(days_lead / 2)
    if days_lead % 2 != 0:
        xTicks = ax.get_xticks()
        newTicks = np.insert(xTicks, int(idxTick + 1), xTicks[idxTick] + 1)
        idxTick += 1
        ax.set_xticks(newTicks)
        plt.setp(ax.get_xticklabels(), fontsize=fontSz)
    ax.get_xticklabels()[idxTick].set_color('r')
    # ax.get_xticklines()[2*idxTick+1].set_color('red') 
    marker_color = ['k'] * len(ax.get_xticklines())
    marker_color[idxTick * 2] = 'r'
    for ii, cur_tick in enumerate(ax.get_xticklines()):
        cur_tick.set_marker('|')
        cur_tick.set_markersize(6)
        cur_tick.set_markeredgecolor(marker_color[ii])


def fig_manuscript_hcst(
        dt_bgn, dt_end, dt_peak, dt_leads, ymax_ens,
        annotate_ha='left',
        syn_hefs_name='v10_sumrsamp30',
        syn_gefs_ver='v4',
        path_results_hefs='results/results_efo_hefs.csv',
        path_results_pfo='results/results_pfo.csv',
        path_results_syn_hefs='results/results_v10_sumrsamp30_all.csv',
        path_results_syn_gefs='results/results_efo_syn_gefs_all.csv',):
    hcst_dt_bgn = '1985-10-16 12:00'
    hcst_dt_end = '2010-09-30 12:00'
    # Load observed flows
    qin_df = pd.read_csv('inp/inp_Q_1948-2010.csv', index_col=0, parse_dates=True)
    # Load Hindcasts
    npzfile_hcst = np.load('inp/inp_Hcst_Cnrfc61_Daily.npz')
    hcst_lm = npzfile_hcst['hcstLm']
    dt_hcst = pd.date_range(start='1985-10-01 12:00', end='2010-09-30 12:00', freq='D')
    # Load perfect forecasts
    npzfile_pcst = np.load('inp/inp_PerfectForecast_1948-2010.npz')
    pcst_lm = npzfile_pcst['pcstLm'][:-1, :]
    dt_pcst = pd.date_range(start='1948-09-30 12:00', end='2010-09-29 12:00', freq='D')
    # Load results
    results_hefs = pd.read_csv(
        path_results_hefs, index_col=[0, 1],
        parse_dates=True).loc[pd.IndexSlice[:, hcst_dt_bgn:hcst_dt_end], :]
    results_pfo = pd.read_csv(
        path_results_pfo, index_col=[0, 1],
        parse_dates=True).loc[pd.IndexSlice[:, hcst_dt_bgn:hcst_dt_end], :]
    # Load synthetic HEFS results
    scen_key_hefs = f'syn_{syn_hefs_name}_'
    results_syn_hefs = [pd.read_csv(
        path_results_syn_hefs,
        index_col=[0, 1], parse_dates=True)]
    # Load synthetic GEFS results
    scen_key_gefs = f'efo_syn_gefs_{syn_gefs_ver}_perturbed_'
    n_syn_gefs = 10
    results_syn_gefs = [pd.read_csv(
        path_results_syn_gefs,
        index_col=[0, 1], parse_dates=True)]
    # Compile into a single dataframe
    results_comb = pd.concat((results_hefs, results_pfo))
    # Index keys and legend
    scenarios_comb = ['pfo', 'efo_cnrfc']
    legend_comb = ['PFO', 'HEFS']
    fig = plt.figure(figsize=(7, 9.5))
    # First Column - synHEFS
    ax1a = fig.add_subplot(4, 2, 1)
    ax2a = fig.add_subplot(4, 2, 3)
    ax3a = fig.add_subplot(4, 2, 5)
    create_hydro_panels(
        ax1a, ax2a, ax3a,
        'syn-HEFS', dt_bgn, dt_end,
        dt_peak, dt_leads, scen_key_hefs,
        scenarios_comb, legend_comb, results_comb, results_syn_hefs, qin_df, 'syn-HEFS',
        annotate_ha=annotate_ha)
    # Second Column - synGEFS
    ax1b = fig.add_subplot(4, 2, 2)
    ax2b = fig.add_subplot(4, 2, 4)
    ax3b = fig.add_subplot(4, 2, 6)
    create_hydro_panels(
        ax1b, ax2b, ax3b, 'syn-GEFS',
        dt_bgn, dt_end, dt_peak, dt_leads, scen_key_gefs,
        scenarios_comb, legend_comb, results_comb, results_syn_gefs, qin_df, 'syn-GEFS',
        show_stor=False, show_flow=False, fnc_syn_color=line_color_syn_purple,
        annotate_ha=annotate_ha)
    # Ensemble forecast panels
    for ii, cur_lead in enumerate(dt_leads.keys()):
        cur_ax = fig.add_subplot(4, 3, 10 + ii)
        show_flow = True if ii == 0 else False
        show_legend = False if ii < len(dt_leads) - 1 else True
        plot_ensemble_fcst(
            ax=cur_ax,
            dt_init=dt_leads[cur_lead],
            dt_peak=dt_peak,
            days_lead=cur_lead,
            n_syn_hefs=len(results_syn_hefs),
            path_syn_hefs=f'inp/syn_hefs/LkMendoSynHcst_corr_{syn_hefs_name}_samp',
            dt_syn_hefs=pd.date_range(start='1985-10-15 12:00', end='2010-09-30 12:00', freq='D'),
            n_syn_gefs=n_syn_gefs,
            path_syn_gefs=f'inp/syn_gefs/inp_hcst_syn_gefs_{syn_gefs_ver}_daily_',
            dt_syn_gefs=None,
            hcst_lm=hcst_lm,
            dt_hcst=dt_hcst,
            pcst_lm=pcst_lm,
            dt_pcst=dt_pcst,
            show_flow=show_flow,
            ymax=ymax_ens,
            key_dt='fcstDates',
            show_legend=show_legend,
        )
    # fig.tight_layout()
    plt.show()
    return fig


def fig_manuscript_prehcst(
        dt_bgn, dt_end, dt_peak, dt_leads, ymax_ens,
        syn_hefs_name='v10_sumrsamp30',
        n_syn=100,
        path_results_pfo='results/results_pfo.csv',
        path_results_syn_hefs='results/results_v10_sumrsamp30_prehcst_all.csv',
        ):
    hcst_dt_bgn = '1948-10-01 12:00'
    hcst_dt_end = '1985-09-30 12:00'
    # Load observed flows
    qin_df = pd.read_csv('inp/inp_Q_1948-2010.csv', index_col=0, parse_dates=True)
    # Load perfect forecasts
    npzfile_pcst = np.load('inp/inp_PerfectForecast_1948-2010.npz')
    pcst_lm = npzfile_pcst['pcstLm'][:-1, :]
    dt_pcst = pd.date_range(start='1948-09-30 12:00', end='2010-09-29 12:00', freq='D')
    # Load results
    results_pfo = pd.read_csv(
        path_results_pfo, index_col=[0, 1],
        parse_dates=True).loc[pd.IndexSlice[:, hcst_dt_bgn:hcst_dt_end], :]
    # Load synthetic HEFS results
    scen_key = f'syn_{syn_hefs_name}_'
    results_syn = [pd.read_csv(
            path_results_syn_hefs,
            index_col=[0, 1], parse_dates=True)]
    # Load hindcasts for plotting
    path_syn_hcst = f'inp/syn_hefs_prehcst/LkMendoSynHcst_corr_{syn_hefs_name}_samp'
    dt_syn = pd.date_range(start='1948-10-01 12:00', end='2010-09-30 12:00', freq='D')
    rand_samp = int(np.random.uniform(low=1, high=100))
    npz_file = np.load(path_syn_hcst + f'{rand_samp}.npz')
    syn_hcst = npz_file['syn_hcstLm'][:, :, :]
    # Index keys and legend
    scenarios = ['pfo']
    legend = ['PFO']
    # Create figure
    fig = plt.figure(figsize=(7, 7))
    # Create axes
    n_rows = 3
    n_cols = 2
    ax1a = fig.add_subplot(n_rows, n_cols, 1)
    # Storage Exceedance
    plot_exceedance_stor(
        ax_stor_exc=ax1a,
        results_comb=results_pfo,
        scen_comb=scenarios,
        legend_comb=legend,
        results_syn=results_syn,
        scen_syn=scen_key,
        syn_legend='syn-HEFS',
        fontsize=4.5,
        lambda_exp=lambda y, pos: (f'{y / 1000.0: .0f}'),
        units='TAF',
        show_xlabels=True,
        title=None
    )
    # Hydrographs
    ax1b = fig.add_subplot(n_rows, n_cols, 3)
    ax2a = fig.add_subplot(n_rows, n_cols, 2)
    ax2b = fig.add_subplot(n_rows, n_cols, 4)
    create_hydro_panels(
        ax1b, ax2a, ax2b,
        plot_title=None,
        dt_bgn=dt_bgn,
        dt_end=dt_end,
        dt_peak=dt_peak,
        dt_leads=dt_leads,
        scen_key=scen_key,
        scenarios_comb=scenarios,
        legend_comb=legend,
        results_comb=results_pfo,
        results_syn=results_syn,
        qin_df=qin_df,
        syn_legend='syn-HEFS',
        showTitle=False)
    ax2a.legend(prop=dict(size=4.7), loc='lower right')
    # Ensemble forecast panels
    for ii, cur_lead in enumerate(dt_leads.keys()):
        cur_ax = fig.add_subplot(n_rows, 3, (n_rows - 1) * 3 + 1 + ii)
        show_flow = True if ii == 0 else False
        show_legend = False if ii < len(dt_leads) - 1 else True
        plot_ensemble_fcst(
            ax=cur_ax,
            dt_init=dt_leads[cur_lead],
            dt_peak=dt_peak,
            days_lead=cur_lead,
            n_syn_hefs=n_syn,
            path_syn_hefs=path_syn_hcst,
            dt_syn_hefs=dt_syn,
            n_syn_gefs=None,
            path_syn_gefs=None,
            dt_syn_gefs=None,
            hcst_lm=syn_hcst,
            dt_hcst=dt_syn,
            pcst_lm=pcst_lm,
            dt_pcst=dt_pcst,
            show_flow=show_flow,
            ymax=ymax_ens,
            key_dt='fcstDates',
            plot_mean_hefs=False,
            hefs_prefix='syn-',
            show_legend=show_legend
        )
    # fig.suptitle('Pre-Hindcast', fontsize=8, fontweight='bold')
    fig.text(0.135, 0.83, 'a)', fontsize=8, fontweight='bold')
    fig.text(0.135, 0.58, 'c)', fontsize=8, fontweight='bold')
    fig.text(0.87, 0.85, 'b)', fontsize=8, fontweight='bold')
    fig.text(0.87, 0.58, 'd)', fontsize=8, fontweight='bold')
    fig.text(0.325, 0.31, 'e)', fontsize=8, fontweight='bold')
    fig.text(0.605, 0.31, 'f)', fontsize=8, fontweight='bold')
    fig.text(0.87, 0.28, 'g)', fontsize=8, fontweight='bold')
    # fig.tight_layout()
    plt.show()
    return fig
