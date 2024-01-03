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
# Date of Creation: 2021/05/04                              #
#############################################################

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import probscale
# from matplotlib.ticker import FormatStrFormatter
import efo.functions as fnc
import matplotlib.ticker as tkr  
import efo.functions as fnc  

DENSE_DOT = (0,(1,1))
DENSE_DASH = (0,(5,1))
DENSE_DASH_DOT = (0,(3,1,1,1))
DENSE_DASH_DOT_DOT = (0,(3,1,1,1,1))
    
# TODO: Check if df is multi index and if scenarios is not None then cycle through unique index values
# TODO: Shouldn't scenarios be changed to simulations? What's the difference?
def plot_time_series(*, ax, df, col, parameter=None, legend=None, title=None, scenarios=[],
                     fontsize=9, ymin=0, ymax=None, xmin=None, xmax=None, linewidth=1.5,
                     legend_loc='best', linestyle='-', alpha=1., y_axis_fnc=None,
                     linecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'],
                     x_axis_date_format=None, show_xticks=True, xtick_rotation=0,
                     legend_fontsize=None, datetime_index='date_time'):
    if x_axis_date_format: ax.xaxis.set_major_formatter(mdates.DateFormatter(x_axis_date_format))
    if legend:
        is_legend=True
    else: 
        is_legend=False
        legend=['_nolegend_']*len(scenarios)
    max_y = ax.get_ylim()[1]
    i = 0
    if not scenarios:
        # try:
        #     date_times = df.index.get_level_values(datetime_index)
        # except:
        date_times = df.index
        ax.plot(date_times,
                df[col], label=legend[0],
                color=linecolor[0], linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        max_y = max(max_y, df[col].max()*1.05)
    else:
        for i, curScen in enumerate(scenarios):
            date_times = df.xs(scenarios[0]).index.get_level_values(datetime_index)
            # ax.plot(df.xs(scenarios[0]).index.get_level_values(date_time_index),
            #         df.xs(curScen)[col], label=legend[i],
            #         color=lineColor[i], linewidth=lineWidth, linestyle=lineStyle, alpha=alpha)
            ax.plot(date_times,
                    df.xs(curScen)[col], label=legend[i],
                    color=linecolor[i], linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            cur_max_y = df.xs(curScen)[col].max()
            max_y = max(cur_max_y+cur_max_y*0.05, max_y)
    if legend and legend[i] != '_nolegend_': ax.legend(prop=dict(size=fontsize), loc=legend_loc)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    else:
        ax.set_ylim(top=max_y)
    # Set y-axis to a lambda expression if one is provided
    # TODO: lambdaExp could be a function so change to function argument to something like fnc_yaxis
    if y_axis_fnc is not None:
        yfmt = tkr.FuncFormatter(y_axis_fnc)
        ax.yaxis.set_major_formatter(yfmt)
    else:
        ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    ax.set_ylim(bottom=ymin)
    if xmin is None:
        # xMin = df.xs(scenarios[0]).index[0] if scenarios else df.index[0]
        xmin = date_times[0]
    if xmax is None:
        # xMax = df.xs(scenarios[0]).index[-1] if scenarios else df.index[-1]
        xmax = date_times[-1]
    ax.set_xlim(xmin, xmax)
    if show_xticks:
        # ax.xaxis.set_major_formatter(mdates.DateFormatter(xAxDateFormat))
        if xtick_rotation != 0:
            ax.xaxis.set_tick_params(rotation = xtick_rotation)
            for label in ax.xaxis.get_majorticklabels():
                label.set_horizontalalignment('right')
    else:
        ax.set_xticklabels([])
    if title is not None: ax.set_title(title, fontsize=fontsize + 1, pad=-1)
    if parameter is not None: ax.set_ylabel(parameter, fontsize=fontsize)
    if is_legend:
        if legend_fontsize is None: legend_fontsize = fontsize
        ax.legend(prop=dict(size=legend_fontsize), loc=legend_loc)
    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    # plt.xticks(fontsize=fontSz)
    # plt.yticks(fontsize=fontSz)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    # ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    # ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.5)
    return ax
    
    
def plot_hydrograph_stor_elev(
        *, ax, df, col, hypso, parameter=None, legend=None, title=None, scenarios=[],
        fontsize=9, ymin=0, ymax=None, xmin=None, xmax=None, linewidth=1.5,
        legend_loc='best', linestyle='-', show_xticks=True,
        linecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'],
        alpha=1., y_axis_fnc=None, is_primary_stor=True, legend_fontsize=None):
    ax = plot_time_series(ax=ax, df=df,
                          col=col,
                          parameter=parameter,
                          scenarios=scenarios,
                          title=title,
                          legend=legend,
                          fontsize=fontsize,
                          ymin=ymin,
                          ymax=ymax,
                          xmin=xmin,
                          xmax=xmax,
                          linestyle=linestyle,
                          linewidth=linewidth,
                          legend_loc=legend_loc,
                          linecolor=linecolor,
                          show_xticks=show_xticks,
                          alpha=alpha,
                          y_axis_fnc=y_axis_fnc,
                          legend_fontsize=legend_fontsize)
    # ax.set_ylim(bottom=yMin)
    def stor2elev(x): return hypso.stor2elev(stor=x)
    def elev2stor(x): return hypso.elev2stor(elev=x)
    y_ticks1 = ax.get_yticks()
    if is_primary_stor:
        ax_elev = ax.secondary_yaxis('right', functions=(stor2elev, elev2stor))
        parameter2 = 'Elevation(ft.)'
        y_ticks2 = hypso.stor2elev(y_ticks1)
    else:
        ax_elev = ax.secondary_yaxis('right', functions=(elev2stor, stor2elev))
        parameter2 = 'Storage (ac-ft)'
        y_ticks2 = hypso.elev2stor(y_ticks1)
    # yTicksStor = ax.get_yticks()
    # elevTicks = hypso.stor2elev(yTicksStor)
    ax_elev.set_yticks(y_ticks2)
    ax_elev.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.1f'))
    plt.setp(ax_elev.get_yticklabels(), fontsize=fontsize)
    ax_elev.set_ylabel(parameter2, fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    # plt.show()
    return ax
    
    
# TODO: This should return a pivot table of exceedance results by alternative
def plot_exceedance(*, ax, df, col, legend=None, scenarios=[None], parameter=None, title=None,
                    fontsize=9, ymin=0, ymax=None, linewidth=1.5, legend_loc='best', linestyle='-',
                    linecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'],
                    x_axis_scale='linear', y_axis_scale='linear', alpha=1., y_axis_fnc=None,
                    show_ylabels=True, show_xlabels=True, marker=None, markersize=None,
                    ytick_precision=3):
    if legend:
        isLegend=True
    else: 
        isLegend=False
        legend=['_nolegend_']*len(scenarios)
    # if isinstance(legend, list):
    #     isLegend =[False]*len(legend)
    #     for i, curLegend in enumerate(legend):
    #         if legend[i] and legend[i] != '_nolegend_':
    #             isLegend[i] = True  
    #         else:
    #             isLegend[i] = False
    #             legend[i]='_nolegend_'
    # elif legend is None: 
    #     legend=['_nolegend_']*len(scenarios)
    maxY = 0
    for i, cur_scen in enumerate(scenarios):
        if cur_scen is not None:
            cur_arr = df.xs(cur_scen)[col].to_numpy() if isinstance(df, pd.DataFrame) else df.xs(cur_scen).to_numpy()
        else:
            cur_arr = df[col].to_numpy() if isinstance(df, pd.DataFrame) else df.xs(cur_scen).to_numpy()
        maxY = max(np.nanmax(cur_arr), maxY)
        exc, arr_sort = fnc.exc_prob(arr=cur_arr)
        exc *= 100
        ax.plot(
            exc, arr_sort,
            label=legend[i], 
            color=linecolor[i],
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            alpha=alpha)
    ax.set_xscale(x_axis_scale)
    ax.set_yscale(y_axis_scale)
    if x_axis_scale == 'log':
        ax.set_xlim(min(exc), 100)
    # elif xAxisScale == 'prob':
    #     ax.set_xlim(min(exc), max(exc))
    elif x_axis_scale == 'linear':
        ax.set_xlim(0, 100)
    ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    else:
        ax.set_ylim(top=maxY+maxY*0.05)
    # Set the y-axis to a lambda expression if one is supplied
    if y_axis_fnc is not None:
        yfmt = tkr.FuncFormatter(y_axis_fnc)
        ax.yaxis.set_major_formatter(yfmt)
    else:
        yTicks = [np.format_float_positional(
            tic, precision=ytick_precision, unique=False, fractional=False, trim='k')
            for tic in ax.get_yticks()]
        ax.set_yticks(ax.get_yticks(), labels=yTicks)
        # ax.set_yticklabels(yTicks)
        # ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    if title is not None: ax.set_title(title, fontsize=fontsize + 1)
    if parameter is not None: ax.set_ylabel(parameter, fontsize=fontsize)
    ax.set_xlabel('Percent Exceedance (%)', fontsize=fontsize)
    if any([l != '_nolegend_' for l in legend]): ax.legend(prop=dict(size=fontsize), loc=legend_loc)
    # TODO: This could be a function in the module where you pass the axis
    if not(show_ylabels):
        ax.set_yticklabels([])
        ax.set_yticklabels([], minor=True)
        ax.set_ylabel(None)
    if not(show_xlabels):
        ax.set_xticklabels([])
        ax.set_xticklabels([], minor=True)
        ax.set_xlabel(None)
    # TODO: grid does not work with log axis
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)


def plot_box(
        *, ax, df, col, scenarios, legend,
        title=None, parameter=None, fontsize=9, ymin=0, ymax=None, linewidth=1.5,
        y_axis_scale='linear', box_colors=None):
    n_scenarios = len(scenarios)
    if box_colors:
        if type(box_colors) is not list and type(box_colors) is not tuple:
            box_colors = [box_colors] * n_scenarios
    else:
        box_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # data_arr = np.zeros(shape=([df.xs(scenarios[0]).count(), n_scenarios]))
    data_arr = np.zeros(shape=([len(df.xs(scenarios[0])), n_scenarios]))
    for i, cur_scen in enumerate(scenarios):
        data_arr[:, i] = df.xs(cur_scen)[col].to_numpy().flatten()\
            if isinstance(df, pd.DataFrame) else df.xs(cur_scen).to_numpy()
    box_plot = ax.boxplot(data_arr, labels=legend, whis=[0, 100], showmeans=True)
    # boxColors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0, n_scenarios):
        plt.setp(box_plot['boxes'][i], color=box_colors[i], linewidth=linewidth)
        plt.setp(box_plot['caps'][2*i], color=box_colors[i], linewidth=linewidth)
        plt.setp(box_plot['caps'][2*i+1], color=box_colors[i], linewidth=linewidth)
        plt.setp(box_plot['fliers'][i], color=box_colors[i], linewidth=linewidth)
        plt.setp(box_plot['means'][i], color=box_colors[i],
                 marker='D', markerfacecolor=box_colors[i], markeredgecolor=box_colors[i])
        plt.setp(box_plot['medians'][i], color=box_colors[i], linewidth=linewidth)
        plt.setp(box_plot['whiskers'][2*i], color=box_colors[i], linewidth=linewidth)
        plt.setp(box_plot['whiskers'][2*i+1], color=box_colors[i], linewidth=linewidth)
    max_y = np.max(data_arr)
    ax.set_yscale(y_axis_scale)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    else:
        ax.set_ylim(top=max_y+max_y*0.05)
    ax.set_ylim(bottom=ymin)
    if title is not None: ax.set_title(title, fontsize=fontsize + 1)
    if parameter is not None: ax.set_ylabel(parameter, fontsize=fontsize)
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    

# TODO: Should be able to pass a figure if you want to use an existing figure object
def fig_exc_box(*, df, col, parameter, scenarios, legend,
                title=None, fontsize=9, ymin=0, ymax=None, linewidth=1.5, legend_loc='best',
                x_axis_scale='linear', y_axis_scale='linear',
                linecolors=plt.rcParams['axes.prop_cycle'].by_key()['color']):
    fig = plt.figure(figsize=(6.5, 4), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(ncols=8, nrows=1)
    # Subplot 1
    ax1 = fig.add_subplot(gs[0, :5])
    plot_exceedance(ax=ax1, df=df,
                    col=col,
                    parameter=parameter,
                    scenarios=scenarios,
                    legend=legend,
                    ymin=ymin,
                    ymax=ymax,
                    fontsize=fontsize,
                    linewidth=linewidth,
                    legend_loc=legend_loc,
                    x_axis_scale=x_axis_scale,
                    y_axis_scale=y_axis_scale,
                    linecolor=linecolors)
    # Subplot 2
    ax2 = fig.add_subplot(gs[0, 5:9])
    plot_box(ax=ax2, df=df,
             col=col,
             scenarios=scenarios,
             legend=legend,
             ymin=ymin,
             ymax=ymax,
             fontsize=fontsize,
             linewidth=linewidth,
             y_axis_scale=y_axis_scale,
             box_colors=linecolors)
    ax2.set_yticklabels([])
    if title is not None: fig.suptitle(title, fontsize=fontsize + 1)
    return fig


def plot_obj_fnc(
        ax, J, dec_vars, dec_vars_labels, prct_exc,
        is_sorted=False, markersize=2, fontsize=7, markertype='o', legend_loc='best'):
    J = J.copy()
    n_curves = J.shape[0]
    curve_num = np.arange(0, n_curves)
    J[np.isnan(J)] = -1.
    if is_sorted:
        sort_rows = np.argsort(J, kind='stable')
        J = J[sort_rows]
    else:
        sort_rows = np.arange(0, n_curves)
    J[J<0] = np.nan
    ax.plot(curve_num, J, label='J')
    for i, cur_dv in enumerate(dec_vars):
        ax.plot(curve_num, cur_dv[sort_rows], linewidth=1, label=dec_vars_labels[i])
    i_prct_grtr = J >= np.nanpercentile(J, 100 - prct_exc)
    marker_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    curve_nums = np.where(i_prct_grtr)[0]
    ax.plot(curve_nums[0], J[i_prct_grtr][0],
            markertype, fillstyle='none',
            label=r'$\leq$' + f'{prct_exc}% Exc J', color=marker_colors[0],
            markersize=markersize)
    # curve_nums = np.where(i_prct_grtr)[0]
    for i in range(1, curve_nums.shape[0]):
        ax.plot(curve_nums[i], J[curve_nums][i],
                markertype, fillstyle='none',
                color=marker_colors[i - int(np.floor(i/len(marker_colors))*len(marker_colors))],
                markersize=markersize)
    ax.set_ylabel('Result', fontsize=fontsize)
    ax.set_xlabel('Curve Alternative', fontsize=fontsize)
    ax.set_xlim(0, n_curves)
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    ax.legend(prop=dict(size=fontsize), loc='best')
    return i_prct_grtr
    
    
# def plot_fitted_tol_curve(
#         ax, curves, yLabel, nDeg=3, plotTitle=None, fontSz=7, showXTicks=True, tStep='D'):
#     nSteps = 1 if tStep == 'D' else 24
#     # Plot Candidate Risk Curves
#     daysHoriz = np.arange(0, (curves.shape[0]-1)*nSteps+1, nSteps)
#     ax.plot(daysHoriz, curves, linewidth=1)
#     fitCurve = fnc.get_polyfit_curve(curves[1:].T, nSteps, nDeg)
#     # fitCurve = np.min(np.vstack((fitCurve, curveMaxRange)), axis=0)
#     # Values should be increasing
#     # for i, val in enumerate(fitCurve):
#     #     if i == 0:
#     #         if val > 0: fitCurve[i] = 0
#     #     else:
#     #         if val < fitCurve[i-1]: fitCurve[i] = 0
#     chngCurve = fitCurve[1:] - fitCurve[:-1]
#     iNegChng = chngCurve < 0
#     fitCurve[:-1][iNegChng] = 0.
#     fitCurve = np.hstack((np.nan, fitCurve))
#     ax.plot(np.arange(0, (curves.shape[0]-1)*nSteps+1), fitCurve, 'k--',linewidth=3)
#     plt.xticks(daysHoriz)
#     ax.set_ylabel(yLabel, fontsize=fontSz)
#     if showXTicks:
#         plt.setp(ax.get_xticklabels(), fontsize=fontSz)
#         ax.set_xlabel('Forecast Lead Time (days)', fontsize=fontSz)
#     else:
#         ax.set_xticklabels([])
#     plt.setp(ax.get_yticklabels(), fontsize=fontSz)
#     ax.set_xlim(0, daysHoriz[-1])
#     ax.set_ylim(bottom=0.)
#     if plotTitle:
#         ax.set_title(plotTitle, size=fontSz)
#     ax.xaxis.grid(True, which='major', linestyle=':')
#     ax.yaxis.grid(True, which='major', linestyle=':')
#     return fitCurve


def plot_fitted_tol_curve(
        ax, curves, ylabel, n_deg=3, plot_title=None, fontsize=7, show_xticks=True, time_step='D'):
    # Find beginning lead days with the same value (usually 0)
    if curves.shape[1] > 1:
        for ii in range(curves.shape[0] - 1):
            if np.any(curves[ii+1, 1:] != curves[ii+1, 0]):
                break
        idx_bgn_interp = ii
    else:
        idx_bgn_interp = 1
    n_steps = 1 if time_step == 'D' else 24
    # Plot Candidate Risk Curves
    days_horiz = np.arange(0, (curves.shape[0]-1)*n_steps+1, n_steps)
    ax.plot(days_horiz, curves, linewidth=1)
    y_intercept = curves[idx_bgn_interp, 0]
    curves_0intercept = curves - y_intercept
    fit_curve_lead = fnc.get_polyfit_curve(curves_0intercept[idx_bgn_interp:].t, n_steps, n_deg) + y_intercept
    fit_curve_fnc = interp1d(
        days_horiz[: idx_bgn_interp+2],
        np.hstack((curves[: idx_bgn_interp+1, 0].flatten(), fit_curve_lead[n_steps])), kind='linear')
    fit_curve_prefix = fit_curve_fnc(np.arange(0, days_horiz[idx_bgn_interp]+1))
    fit_curve = np.hstack((fit_curve_prefix, fit_curve_lead[n_steps:]))
    fit_curve[fit_curve<0.] = 0.
    # fit_curve = np.min(np.vstack((fit_curve, curveMaxRange)), axis=0)
    # Values should be increasing
    # for i, val in enumerate(fit_curve):
    #     if i == 0:
    #         if val > 0: fit_curve[i] = 0
    #     else:
    #         if val < fit_curve[i-1]: fit_curve[i] = 0
    chng_curve = fit_curve[1:] - fit_curve[:-1]
    i_neg_chng = chng_curve < 0
    fit_curve[:-1][i_neg_chng] = 0.
    # fit_curve = np.hstack((np.nan, fit_curve))
    ax.plot(np.arange(0, (curves.shape[0]-1)*n_steps+1), fit_curve, 'k--',linewidth=3)
    plt.xticks(days_horiz)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if show_xticks:
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_xlabel('Forecast Lead Time (days)', fontsize=fontsize)
    else:
        ax.set_xticklabels([])
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_xlim(0, days_horiz[-1])
    ax.set_ylim(bottom=0.)
    if plot_title:
        ax.set_title(plot_title, size=fontsize)
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    return fit_curve


def scatter_decision_variables(
        ax, J, x_dv, y_dv, prct_exc, xlabel, ylabel,
        markersize=2.5, fontsize=7, markertype='o', legend_loc='lower left'):
    i_prct_grtr = J >= np.nanpercentile(J, 100 - prct_exc)
    plt.scatter(x_dv[np.invert(i_prct_grtr)], y_dv[np.invert(i_prct_grtr)],
                s=1.5, label=None, color='grey')
    curve_nums = np.where(i_prct_grtr)[0]
    marker_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.plot(x_dv[curve_nums][0], y_dv[curve_nums][0], 'o',
            label=r'$\leq$' + f'{prct_exc}% Exc J',
            color=marker_colors[0],
            markersize=markersize)
    for i in range(1, curve_nums.shape[0]):
        ax.plot(x_dv[curve_nums][i], y_dv[curve_nums][i], 'o',
                color=marker_colors[i - int(np.floor(i/len(marker_colors))*len(marker_colors))],
                markersize=markersize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    ax.legend(prop=dict(size=fontsize), loc=legend_loc)
    return i_prct_grtr




