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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
from matplotlib.axis import Axis
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

DENSE_DOT = (0,(1,1))
DENSE_DASH = (0,(5,1))
DENSE_DASH_DOT = (0,(3,1,1,1))
DENSE_DASH_DOT_DOT = (0,(3,1,1,1,1))


def plot_fcst_ens(ax, t_fcst, fcst_results, parameter=None, units=None,
                  fontsize=9, time_interval=24, ymin=0, ymax=None, plot_title=None, linewidth=0.75,
                  show_xticks=True, show_title=True, color_ovrd=None, linestyle='-', is_zulu=True,
                  x_axis_date_format=None, y_axis_fnc=None):
    z = 'z' if is_zulu else ''
    if x_axis_date_format is None: x_axis_date_format = f'%m-%d-%H{z}'
    if color_ovrd:
        ax.plot(t_fcst, fcst_results, linewidth=linewidth, linestyle=linestyle, color=color_ovrd)
    else:
        ax.plot(t_fcst, fcst_results, linewidth=linewidth, linestyle=linestyle)
    ax.relim()
    ax.autoscale()
    ax.set_xlim(t_fcst[0], t_fcst[-1])
    ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    if y_axis_fnc is not None:
        yfmt = tkr.FuncFormatter(y_axis_fnc)
        ax.yaxis.set_major_formatter(yfmt)
    else:
        ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    # Set the date tick interval
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=time_interval))
    if show_xticks:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(x_axis_date_format))
        ax.xaxis.set_tick_params(rotation = 45)
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment('right')
    else:
        ax.set_xticklabels([])
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    if parameter is not None: ax.set_ylabel(parameter +' (' + units +')', fontsize=fontsize)
    if show_title:
        if plot_title:
            ax.set_title(plot_title, fontsize=fontsize)
        else:
            ax.set_title(parameter, fontsize=fontsize)
    return ax
    
    
def plot_fcst_risk(*, ax, t_fcst, risk_tol, risk_fcst,
                   name_threshold='Storage Threshold', plot_title=None, fontsize=9,
                   time_interval=24, ymin=0, ymax=None, show_xticks=True,
                   show_title=True, color=None, legend='', is_zulu=True, xlabel=None,
                   legend_loc='upper right'):
    z = 'z' if is_zulu else ''
    color_tol = color if color else 'b'
    color_risk = color if color else 'r'
    ax.plot(t_fcst, risk_tol * 100, color=color_tol, linestyle='--', linewidth=1, label=f'{legend} - Risk Tolerance')
    ax.plot(t_fcst, risk_fcst * 100, color=color_risk, linestyle='-', linewidth=1.5, label=f'{legend} - Forecasted Risk')
    ax.set_xlim(t_fcst[0], t_fcst[-1])
    ax.set_ylim(bottom=0, top=ymax)
    # Set the date tick interval
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=time_interval))
    if show_xticks:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(f'%m-%d-%H{z}'))
        ax.xaxis.set_tick_params(rotation = 45)
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment('right')
        if xlabel is None:
            if is_zulu:
                ax.set_xlabel('Month-Day-Hour (UTC)', fontsize=fontsize)
            else:
                ax.set_xlabel('Month-Day-Hour (Local Time)', fontsize=fontsize)
        else:
            ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xticklabels([])
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_ylabel('Risk (%)', fontsize=fontsize)
    ax.legend(prop=dict(size=fontsize - 1), loc=legend_loc)
    if plot_title and show_title: ax.set_title(plot_title, fontsize=fontsize)
    elif show_title: ax.set_title('% of Ensemble Members > ' + name_threshold, fontsize=fontsize)
    return ax

    
def plot_fcst_hydro(ax, t_fcst, fcst_vals, linelabel='_', ylabel=None,
                    fontsize=9, time_interval=24, ymin=0, ymax=None,
                    show_xticks=True, plot_title=None, linecolor=None,
                    linewidth=1.5, linestyle='-', legend_loc='upper left', is_zulu=True,
                    y_axis_fnc=None, x_axis_date_format=None, show_legend=True):
    z = 'z' if is_zulu else ''
    if x_axis_date_format is None: x_axis_date_format = f'%m-%d-%H{z}'
    if linecolor is None:
        ax.set_prop_cycle(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
    else:
        ax.set_prop_cycle(color=linecolor)
    ax.plot(t_fcst, fcst_vals, linewidth=linewidth, label=linelabel, linestyle=linestyle)
    if ymax is None:
        cur_ymax = ax.get_ylim()[1]
        ymax = max(cur_ymax, np.nanmax(fcst_vals) * 1.05)
    ax.relim() 
    ax.set_xlim(t_fcst[0], t_fcst[-1])
    ax.set_ylim(bottom=ymin)
    ax.set_ylim(top=ymax)
    if y_axis_fnc is not None:
        yfmt = tkr.FuncFormatter(y_axis_fnc)
        ax.yaxis.set_major_formatter(yfmt)
    else:
        ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    # Set the date tick interval
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=time_interval))
    if show_xticks:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(x_axis_date_format))
        ax.xaxis.set_tick_params(rotation = 45)
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment('right')
    else:
        ax.set_xticklabels([])
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if plot_title is not None: ax.set_title(plot_title, fontsize=fontsize)
    if show_legend: ax.legend(prop=dict(size=fontsize - 1), loc=legend_loc)
    return ax
    
    
