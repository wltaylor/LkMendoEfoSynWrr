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
# Date of Creation: 2020/05/18                              #
#############################################################


import abc
import numpy as np
from pandas import DatetimeIndex, Timestamp, date_range
import datetime as dt
from datetime import timedelta
from copy import copy
import copy as cp

class TimeBase(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name
        self._observers = []
        super().__init__()    

    @classmethod
    def __subclasshook__(cls,C):
        if cls is TimeBase:
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
    
    def bind_to(self, callback):
        self._observers.append(callback)
    

class TimeCont(TimeBase):
    def __init__(self, name, sim_bgn=None, sim_end=None, time_unit='h', datetimes=None, n_units=1):
        # Call super class constructor
        super().__init__(name)
        # Set subclass properties
        if sim_bgn is not None:
            self.sim_bgn = Timestamp(sim_bgn)
        else:
            self.sim_bgn = Timestamp(datetimes[0])
        if sim_end is not None:
            self.sim_end = Timestamp(sim_end)
        else:
            self.sim_end = Timestamp(datetimes[-1])
        self.time_unit = time_unit
        if time_unit == 'D':
            self.n_hrs = int(24 * n_units)
        else:
            self.n_hrs = int(n_units)
        self.t_delta = timedelta(hours=self.n_hrs)
        if datetimes is None:
            self.datetimes = date_range(
                start = self.sim_bgn,
                end = self.sim_end,
                freq = f'{self.n_hrs}h')
        else:
            self.datetimes = DatetimeIndex(datetimes)
        self.n_steps = self.datetimes.size
        self.end = self.n_steps - 1
        self.steps = np.arange(self.n_steps, dtype=int)
        self._step = self.steps[0]
        self._cur_dt = self.datetimes[0]

    @property
    def step(self):
        return self._step

    @step.setter        
    def step(self, i):
        self._step = self.steps[i].item()
        self._cur_dt = copy(self.datetimes[i])
        for callback in self._observers:
            callback(self._cur_dt)

    @property
    def cur_dt(self):
        return self._cur_dt
    
    @property
    def shape(self):
        return (self.n_steps, 1)

    def get_datetime_offset(self, ts_offset):
        return self.datetimes[min(self.n_steps - 1, self.step + ts_offset)]

    def get_period(self, dt_bgn, dt_end):
        pass
        
    def get_vdate(self):
        vdate = np.full((len(self.datetimes), 4), np.nan)
        for i, cur_dt in enumerate(self.datetimes):
            vdate[i, :] = [cur_dt.year, cur_dt.month, cur_dt.day, cur_dt.hour]
        return vdate
    
    def get_ndate(self):
        return self.datetimes.astype(np.int64)
    
    def get_ts_from_dt(self, dt):
        ts = [np.where(self.datetimes == dt)[0][-1] for dt in dt
              if np.any(self.datetimes == dt)]
        return ts
        
        
class TimeFcst(TimeBase):
    def __init__(
            self, name, time_cont, fcst_freq=24, fcst_horiz=360,
            t_fcst=None, fcst_init_dates=None, fcst_lead_datetimes=None):
        # Call super class constructor
        super().__init__(name)
        self.t_cont = time_cont
        self.n_hrs = time_cont.n_hrs
        self.time_unit = time_cont.time_unit
        self.t_delta = time_cont.t_delta
        self._n_steps = int(fcst_horiz / self.t_cont.n_hrs + 1)
        self._end = self._n_steps - 1
        self._steps = np.arange(self._n_steps)
        self._step = self._steps[0]
        self.fcst_freq = int(fcst_freq)
        self._row_cur_fcst = 0
        self.t_cont.bind_to(self.update_current_t_fcst)
        t_delta_fcst = np.timedelta64(self.fcst_freq, 'h')
        if fcst_init_dates is None:
            self.fcst_dates = DatetimeIndex(
                np.arange(np.datetime64(self.t_cont.sim_bgn),
                          np.datetime64(Timestamp(self.t_cont.sim_end) + self.t_delta),
                          t_delta_fcst, dtype='datetime64[h]').astype('datetime64[h]'))
        else:
            self.fcst_dates = DatetimeIndex(fcst_init_dates + self.t_delta)
        self.n_fcsts = self.fcst_dates.shape[0]
        if t_fcst is None and fcst_lead_datetimes is None:
            t_fcst = np.empty(self.n_fcsts, dtype=TimeCont)
            for i in range(0, self.n_fcsts):
                idx_tcont_bgn = np.where(time_cont.datetimes == self.fcst_dates[i])[0][0] - 1
                t_bgn = time_cont.datetimes[idx_tcont_bgn] if idx_tcont_bgn > 0 else time_cont.datetimes[0] - self.t_delta
                t_end = np.datetime64(t_bgn + np.timedelta64(fcst_horiz, 'h'))
                # Alternative using Pandas:
                t_fcst[i] = TimeCont(
                    name = 'timeFcst'+str(i), 
                    sim_bgn= t_bgn,
                    sim_end= t_end,
                    time_unit= time_cont.time_unit)
        elif t_fcst is None and fcst_lead_datetimes is not None:
            t_fcst = np.empty(self.n_fcsts, dtype=TimeCont)
            for i in range(0, fcst_lead_datetimes.shape[0]):
                t_fcst[i] = TimeCont(
                    name = 'timeFcst'+str(i), 
                    time_unit= time_cont.time_unit,
                    datetimes=fcst_lead_datetimes[i, :self._n_steps])
        self.t_fcst = t_fcst
        if np.where(self.fcst_dates <= self.t_cont.sim_bgn)[0].shape[0] > 0:
            idxBgn = np.where(self.fcst_dates <= self.t_cont.sim_bgn)[0][-1]
        else:
            idxBgn = 0
        self._t_fcst_cur = self.t_fcst[idxBgn]

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, i):
        self._step = self._t_fcst_cur.steps[i].item()
        self._cur_dt = copy(self._t_fcst_cur.datetimes[i])
        self.get_fcst_time().step = self.step
        # for callback in self._observers:
        #         callback(self._rowCurFcst, self.step)

    @property
    def cur_dt(self):
        return self._cur_dt
    
    @property
    def shape(self):
        return (self.t_cont.n_steps, self.n_steps)
    
    @property
    def n_steps(self):
        return self._n_steps
    
    @property
    def end(self):
        return self._end
    
    @property
    def steps(self):
        return np.arange(self._n_steps)

    def update_current_t_fcst(self, cur_dt):
        chk_cur_row = np.where(self.fcst_dates <= cur_dt)[0][-1]
        if chk_cur_row != self._row_cur_fcst:
            self._row_cur_fcst = chk_cur_row
            self._t_fcst_cur = copy(self.t_fcst[self._row_cur_fcst])
            for callback in self._observers:
                callback(self._row_cur_fcst)
                  
        
    def get_fcst_time(self):
        return self._t_fcst_cur
                
    def get_fcst_step(self):
        return np.where(self._t_fcst_cur.datetimes == self.t_cont.cur_dt)[0][0]
    
    def get_datetime_offset(self, ts_offset):
        return self._t_fcst_cur.get_datetime_offset(ts_offset)



