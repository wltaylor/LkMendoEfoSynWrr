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
from scipy.optimize import curve_fit
from efo.units import UnitsStandard
import operator


def exc_prob(*, arr):
    nRows = arr.size
    # Calculate exceedance probability
    arrSort = -np.sort(-arr.flatten())
    arrRank = np.arange(1, nRows+1)
    excProb = np.zeros(nRows)
    excProb = arrRank/(nRows+1)
    return excProb, arrSort


def calc_cumulative_wy(*, q, dateTime):
    qCumWy = np.empty(dateTime.size)
    qCumWy[0] = q[0] 
    for i in range(1,dateTime.size):
        if (dateTime[i].month == 10) and (dateTime[i].day == 1):
            qCumWy[i] = q[i]
        else:
            qCumWy[i] = qCumWy[i-1] + q[i]
    return qCumWy


def calc_wy_vol(*, T, df, col, scenarios=[None], units):
    vDate = T.get_vdate()
    years = np.unique(vDate[:, 0]).astype(int)
    if vDate[0,1] >= 10:
        years = years[1:]
    returnDf = pd.DataFrame()
    wyVol = np.empty(years.size)
    for j, curScen in enumerate(scenarios):
        if curScen:
            idxName = np.full(years.size, scenarios[j])
            curDf = df.xs(curScen)
        else:
            idxName = None
            curDf = df
        for i, curYr in enumerate(years):
            iCurWY = (curDf.index.year == curYr-1) & (curDf.index.month >= 10) | \
                (curDf.index.year == curYr) & (curDf.index.month <= 9)
            wyVol[i] = np.sum(curDf.loc[iCurWY][col].to_numpy()*units.flow2vol)
        if idxName:
            dfIndices = [idxName, years]
            dfIdx = pd.MultiIndex.from_arrays(dfIndices, names=('name_scenario', 'water_year'))
        else:
            dfIdx = [years]
        curDf = pd.DataFrame({'wy_vol': wyVol},
                             columns = ['wy_vol'], index=dfIdx)
        returnDf = returnDf.append(curDf)
    return returnDf


def calc_ndays_exceed_elev(*, hypso, T, df, col, scenarios, elevThresh):
    storThresh = np.interp(elevThresh, hypso.elev, hypso.stor)
    returnDf = pd.DataFrame()
    for j, curScen in enumerate(scenarios):
        curScenarioDf = df.xs(curScen)
        curDateTime = curScenarioDf.index
        years = np.unique(curDateTime.year)
        nDays = np.empty(years.size)
        idxName = np.full(years.size, scenarios[j])
        for i, curYr in enumerate(years):
            iCurWY = (curDateTime.year == curYr-1) & (curDateTime.month >= 10) | \
                (curDateTime.year == curYr) & (curDateTime.month <= 9)
            nDays[i] = np.sum(curScenarioDf.loc[iCurWY][col] > storThresh)
        dfIndices = [idxName, years]
        dfMultiIdx = pd.MultiIndex.from_arrays(dfIndices, names=('name_scenario', 'water_year'))
        curDf = pd.DataFrame({'days_abv_elev': nDays},
                        columns = ['days_abv_elev'], index=dfMultiIdx)
        returnDf = returnDf.append(curDf)
    return returnDf




