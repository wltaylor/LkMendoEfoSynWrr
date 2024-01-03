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
# Date of Creation: 2020/05/07                              #
#############################################################

import abc
import numpy as np
from efo.lookup import LkupTblAnn
from efo.time import TimeCont

class EvapBase(metaclass=abc.ABCMeta):
    def __init__(self, name, time):
        self.name = name
        self.t = time
        self.evap = np.empty(self.t.n_steps)
        super().__init__()  
    @abc.abstractmethod
    def calc_evap(self, stor):
        pass
    @classmethod
    def __subclasshook__(cls,C):
        if cls is EvapBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
        

class EvapLkupTbl(EvapBase):
    def __init__(self, name, time, hypso, month_day_hour, evap_vals, *,
                 typ='step', time_unit=None):
        # Call super class constructor
        super().__init__(name, time)
        # Create lookup table
        self.evapRateTbl = LkupTblAnn(name +'RateTbl', time, month_day_hour, evap_vals,
                                      tbl_type=typ, time_unit=time_unit)
        self.hypso = hypso
    def calc_evap(self, stor):
         curEvapRate = self.evapRateTbl.get_val(self.t.cur_dt)
         area = self.hypso.stor2area(stor)
         self.evap[self.t.step] = curEvapRate / 12 * area * self.t.n_hrs / 24
         return self.evap[self.t.step]
     
        