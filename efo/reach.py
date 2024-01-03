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
from efo.time import TimeBase
import copy as cp


class ReachBase(metaclass=abc.ABCMeta):
    def __init__(self, name, time, junction_us):
        self.name = name
        self.junction_us = junction_us
        self.junction_us.bind_to_reach(self._send_self_2_us_junction)
        if issubclass(type(time), TimeBase):
            self.t = time
            self.qout = np.full(time.n_steps, np.nan)
        super().__init__()

    @abc.abstractmethod
    def calc_qout(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is ReachBase:
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
    
    def _send_self_2_us_junction(self):
        return self


class ReachConstLag(ReachBase):
    def __init__(self, name, time, junction_us, lag_hrs=0):
        # Call super class constructor
        super().__init__(name, time, junction_us)
        self.lag_time = int(round(lag_hrs / time.n_hrs))

    def calc_qout(self, *, ts_offset=0):
        qOut = self.junction_us.get_qout(ts_offset=ts_offset - self.lag_time)
        self.qout[min(self.t.end, self.t.step + ts_offset)] = qOut
        return qOut
    

class ReachMuskingum(ReachBase):
    def __init__(self, name, time, junction_us, hrs_k, x):
        # Call super class constructor
        super().__init__(name, time, junction_us)
        self.K = hrs_k * 3600.
        self.x = x
        tsec = self.t.t_delta.seconds
        self.c0 = (0.5*tsec - self.K*x)/(self.K*(1 - x) + 0.5*tsec)
        self.c1 = (0.5*tsec + self.K*x)/(self.K*(1 - x) + 0.5*tsec)
        self.c2 = (self.K*(1 - x) - 0.5*tsec)/(self.K*(1 - x) + 0.5*tsec)
        self.lag_time = int(round(hrs_k / time.n_hrs))

    def calc_qout(self, *, ts_offset=0):
        if self.t.step > 0:
            qout = self.junction_us.qout[self.t.step] * self.c0 + \
                   self.junction_us.qout[self.t.step - 1] * self.c1 + \
                   self.qout[self.t.step - 1] * self.c2
        else:
            qout = self.junction_us.qout[self.t.step]
        self.qout[min(self.t.end, self.t.step + ts_offset)] = qout
        return qout

