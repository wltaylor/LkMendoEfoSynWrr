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
# Date of Creation: 2020/05/09                              #
#############################################################


import abc
import numpy as np
import pandas as pd
# import time as tm
import datetime as dt
import calendar as cl
from datetime import timedelta
from scipy.interpolate import interp1d
import copy as cp


class LkupTblBase(metaclass=abc.ABCMeta):
    def __init__(self, name, idx_vals, lkup_vals):
        self.name = name
        self.idx_vals = idx_vals
        self.lkup_vals = lkup_vals
        self._min_idx = np.amin(self.idx_vals)
        self._max_idx = np.amax(self.idx_vals)
        super().__init__()

    @abc.abstractmethod
    def get_val(self, idx_val):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is LkupTblBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo))
        return result

    @property
    def max_idx(self):
        return self._max_idx

    @property
    def min_idx(self):
        return self._min_idx


class LkupTbl(LkupTblBase):
    def __init__(self, name, idx_vals, lkup_vals):
        # Call super constructor
        super().__init__(name, idx_vals, lkup_vals)

    def get_val(self, val_lkup, col=0):
        try:
            i_lkup = np.where(self.idx_vals <= val_lkup)[0]
            if i_lkup.size == 0: i_lkup = np.array([0])
            return self.lkup_vals[i_lkup[-1]].item() if self.lkup_vals.ndim == 1 \
                else self.lkup_vals[i_lkup[-1], col].item()
        except Warning as e:
            print(e)
            print(val_lkup)
        

class LkupTblElev(LkupTbl):
    def __init__(self, name, hypso, x_elev, lkup_vals):
        # Convert rating table to storage
        x_stor = hypso.elev2stor(x_elev)
        # Call super constructor
        super().__init__(name, x_stor, lkup_vals)


class LkupTblInterp(LkupTblBase):
    def __init__(
            self, name, idx_vals, lkup_vals, interp_type='linear', fill_value=None):
        if fill_value is None:
            fill_value=(np.min(lkup_vals), np.max(lkup_vals))
        # Call super constructor
        super().__init__(name, idx_vals, lkup_vals)
        self.interp_fnc = interp1d(
            self.idx_vals, self.lkup_vals, axis=0, kind=interp_type,
            fill_value=fill_value, bounds_error=False)

    def get_val(self, x_vals, col=0):
        interp_vals = self._get_val(x_vals)
        if interp_vals.ndim == 0:
            return_vals = interp_vals.item()
        elif  interp_vals.ndim == 1:
            return_vals = interp_vals[col]
        else:
            return_vals = interp_vals[: ,col]
        return return_vals
    
    def _get_val(self, x_vals, col=0):
        return self.interp_fnc(x_vals)
    
    
class LkupTblInterpPoly(LkupTblInterp):
    def __init__(self, name, idx_vals, lkup_vals, *, deg=2):
        # Call super constructor
        super().__init__(name, idx_vals, lkup_vals)
        self.x_low_bound = np.min(idx_vals)
        self.y_low_bound = np.min(lkup_vals)
        polyCoeff = np.polyfit(idx_vals, lkup_vals, deg)
        self.interp_fnc = np.poly1d(polyCoeff)

    def _get_val(self, x_vals, col=0):
        if np.any(x_vals < self.x_low_bound):
            interp_vals = np.full(np.shape(x_vals), self.y_low_bound)
        else:
            interp_vals = self.interp_fnc(x_vals)
        return interp_vals


class LkupTblAnn(LkupTblInterp):
    def __init__(
            self, name, time, month_day_hour, tbl_vals, tbl_type='interp', time_unit='h'):
        self.type = tbl_type
        interp_type = 'linear' if tbl_type=='interp' else 'next'
        if time_unit is not None:
            self.time_unit = time_unit
            if time_unit == 'M':
                self.t_delta = np.nan
            elif time_unit == 'D':
                self.t_delta = timedelta(days=1)
            else:
                self.t_delta = timedelta(hours=1)
        else:
            self.time_unit = time.time_unit
            self.t_delta = time.t_delta
        if np.isscalar(tbl_vals): tbl_vals = np.array([tbl_vals], dtype=float)
        tbl_vals = np.atleast_1d(tbl_vals)
        tbl_vals = tbl_vals.reshape(len(tbl_vals), tbl_vals.ndim) if tbl_vals.ndim == 1 else tbl_vals
        month_day_hour = month_day_hour.reshape(
            month_day_hour.ndim, month_day_hour.shape[0]) if month_day_hour.ndim == 1 else month_day_hour
        # Add Jan 1 if missing
        if month_day_hour[0, 2] > 0 or month_day_hour[0, 1] > 1 or month_day_hour[0, 0] > 1:
            month_day_hour = np.vstack(
                (np.array([1, 1, 0]), month_day_hour))
            tbl_vals = np.vstack((tbl_vals[0], tbl_vals))
        # Add Dec 31 if missing
        if month_day_hour[-1, 2] < 23 or month_day_hour[0, 1] < 31 or month_day_hour[0, 0] < 12:
            month_day_hour = np.vstack(
                (month_day_hour, np.array([12, 31, 23])))
            tbl_vals = np.vstack((tbl_vals, tbl_vals[-1]))
        x_datetime = np.empty(month_day_hour.shape[0])
        for i in range(0, month_day_hour.shape[0]):
            cur_datetime = dt.datetime(
                year=2020,
                month=int(month_day_hour[i, 0]),
                day=int(month_day_hour[i, 1]),
                hour=int(month_day_hour[i, 2]), tzinfo=dt.timezone.utc)
            x_datetime[i] = cl.timegm(cur_datetime.timetuple())
        # Call super constructor
        super().__init__(name, x_datetime, tbl_vals, interp_type=interp_type)

    def _get_ndate(self, time_stamp):
        time_tuple = dt.datetime(
            year=2020,
            month=time_stamp.month,
            day=time_stamp.day,
            hour=time_stamp.hour,
            tzinfo=dt.timezone.utc).timetuple()
        ndate = cl.timegm(time_tuple)
        return ndate

    def get_val(self, datetime, col=0):
        if isinstance(datetime, pd._libs.tslibs.timestamps.Timestamp):
            ndate = self._get_ndate(datetime)
        else:
            ndate = np.array([self._get_ndate(cur_dt) for cur_dt in datetime])
        interp_vals = self._get_val(ndate)
        if interp_vals.ndim == 0:
            return_vals = interp_vals.item()
        elif interp_vals.ndim == 1:
            return_vals = interp_vals[col]
        else:
            return_vals = interp_vals[:, col]
        return return_vals

