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
# Date of Creation: 2020/04/28                              #
#############################################################


import abc
import numpy as np
# import h5py
from efo.time import TimeBase
from efo.lookup import LkupTblAnn
import copy as cp

class QinBase(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name
        super().__init__()

    @abc.abstractmethod
    def get_qin(self):
        pass

    @classmethod
    def __subclasshook__(cls,C):
        if cls is QinBase:
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
        

class Qin(QinBase):
    def __init__(self, name, time, qin):
        # Call super class constructor
        super().__init__(name)
        # Set class properties
        self.qin = qin
        self.t = time
        
    def get_qin(self, ts_offset=0):
        return self.qin[min(self.t.end, self.t.step + ts_offset)].item()
        

class QinRouted(QinBase):
    def __init__(self, name, time, reach_us):
        # Call super class constructor
        super().__init__(name)
        # Set class properties
        self.reach_us = reach_us
        if issubclass(type(time), TimeBase):
            self.T = time
            
    def get_qin(self, ts_offset=0):
        return self.reach_us.calc_qout(ts_offset=ts_offset)
    
    
class QinLkup(QinBase):
    def __init__(self, name, time, month_day_hour, q_tbl_vals, *,
                 tbl_type='interp', time_unit=None):
        # Call super class constructor
        super().__init__(name)
        # Set class properties
        self.q_lkup_tbl = LkupTblAnn(
            name +'Tbl', time, month_day_hour, q_tbl_vals,
            tbl_type=tbl_type, time_unit=time_unit)
        if issubclass(type(time), TimeBase):
            self.t = time
            self.qin = np.empty(time.n_steps)

    def get_qin(self, ts_offset=0):
        qin = self.q_lkup_tbl.get_val(self.t.get_datetime_offset(ts_offset))
        self.qin[min(self.t.end, self.t.step + ts_offset)] = qin
        return qin
        
     
class Qloss(QinBase):
    def __init__(self, name, qin):
        # Call super class constructor
        super().__init__(name)
        if issubclass(type(qin), QinBase):
            self.qin = qin
            
    def get_qin(self, tsOffset=0):
        return -self.qin.get_qin(tsOffset)
    
    
class QinSpecified(Qin):
    def __init__(self, name, time):
        # Call super class constructor
        super().__init__(name, time, qin=[])
        # Set class properties
        self.t = time
        self.qin = np.full(self.t.n_steps, 0.)
        
    def set_qin(self, ts_offset=0, *, q_specified=np.nan):
        self.qin[min(self.t.end, self.t.step + ts_offset)] = q_specified


class QinFcst(QinBase):
    def __init__(self, name, time_fcst, qin_fcst):
        # Call super class constructor
        super().__init__(name)
        # Set class properties
        self.qin_fcst = qin_fcst
        self.t_fcst = time_fcst
        self.t_fcst.bind_to(self.update_current_qfcst)
        self.qin = qin_fcst[0, :].flatten() if qin_fcst is not None else None
    
    def update_current_qfcst(self, row_fcst, col_fcst=0):
        self.qin = self.qin_fcst[row_fcst, col_fcst:].flatten().copy()
        
    def get_qin(self, ts_offset=0):
        return self.qin[min(self.t_fcst.end, self.t_fcst.step + ts_offset)].item()
        

class QinFcstSpecified(QinBase):
    def __init__(self, name, time_fcst):
        # Call super class constructor
        super().__init__(name)
        # Set class properties
        self.t_fcst = time_fcst
        self.qin = np.full(self.t_fcst.n_steps, np.nan)
        
    def set_qin_fcst(self, q_specified):
        cur_lead = q_specified.shape[0]
        self.qin = np.hstack(
            (q_specified.flatten(), np.full(self.t_fcst.n_steps - cur_lead, np.nan)))
        
    def get_qin(self, ts_offset=0):
        return self.qin[min(self.t_fcst.end, self.t_fcst.step + ts_offset)].item()
            

class QinFcstPerfect(QinFcst):
    def __init__(self, name, time_fcst, qin_perfect):
        # Call super class constructor
        super().__init__(name, time_fcst, None)
        # Set class properties
        self.qin_perfect = qin_perfect
        self.qin = np.full(self.t_fcst.n_steps, np.nan)
        self.update_current_qfcst(0)
        self._set_init_qfcst()
        
    def _set_init_qfcst(self):
        row_end = min(self.t_fcst.n_steps, self.qin_perfect.shape[0])
        self.qin[0:row_end] = np.hstack((np.nan, self.qin_perfect[0: row_end - 1].copy()))
    
    def update_current_qfcst(self, row_fcst):
        row_bgn = self.t_fcst.t_cont.step - 1
        row_end = min(self.qin_perfect.shape[0], self.t_fcst.t_cont.step + self.t_fcst.n_steps - 1)
        if row_bgn == -1:
            self.qin[0: row_end + 1] = np.hstack((np.nan, self.qin_perfect[row_bgn + 1: row_end].copy()))
        elif row_end - row_bgn < self.t_fcst.n_steps:
            self.qin[0: row_end - row_bgn] = self.qin_perfect[row_bgn: row_end].copy()
        else:
            self.qin[0: row_end - row_bgn + 1] = self.qin_perfect[row_bgn: row_end].copy()
        
    def get_qin(self, ts_offset=0):
        return self.qin[min(self.t_fcst.end, self.t_fcst.step + ts_offset)]
        