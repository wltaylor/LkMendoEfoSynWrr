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
# Date of Creation: 2020/05/08                              #
#############################################################


import abc
import numpy as np
from efo.rule import RuleBase
from efo.lookup import LkupTblElev, LkupTblInterp
from efo.time import TimeBase
from scipy import interpolate


class RuleEmgcBase(RuleBase):
    def __init__(self, name, time, operating_junction,
                 rule_type=RuleBase.MIN, release_type=RuleBase.CTRL_RLS):
        super().__init__(name, time, operating_junction,
                         rule_type=rule_type, release_type=release_type)

    @classmethod
    def __subclasshook__(cls,C):
        if cls is RuleEmgcBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
        
        
class RuleEmgcCtrlOutlet(RuleEmgcBase):
    def __init__(self, name, time, operating_junction, units,
                 elev=None, rls_emgc=None):
        # Call super class constructor
        super().__init__(name, time, operating_junction,
                         rule_type=RuleBase.MIN, release_type=RuleBase.CTRL_RLS)
        # Class properties
        self.units = units
        self.rls_emgc_sched = []
        if all([operating_junction.hypso, np.any(elev), np.any(rls_emgc)]):
            self.set_emergency_table(name, operating_junction.hypso, elev, rls_emgc)

    def set_emergency_table(self, name, hypso, elev, rls_emgc):
        self.rls_emgc_sched = LkupTblElev(name, hypso, elev, rls_emgc)

    def _get_rule_q(self):
        rls_unctrl_prev = self.operating_jnc.rls_unctrl[max(0, self.t.step - 1)]
        stor_prev = self.operating_jnc.stor[max(0, self.t.step - 1)]
        if stor_prev > self.rls_emgc_sched.min_idx:
            rls_emgc = self.rls_emgc_sched.get_val(stor_prev)
            # Calculate storage after spill and emergency releases
            stor_prev = stor_prev - (rls_unctrl_prev + rls_emgc)*self.units.flow2vol
            # If storage is below the emergency pool then adjust emergency release
            if rls_unctrl_prev > 0 and rls_emgc > 0 and stor_prev < self.rls_emgc_sched.min_idx:
                rls_emgc = rls_emgc + (stor_prev - self.rls_emgc_sched.minIdx) * self.units.vol2flow
                if rls_emgc < 0:
                    rls_emgc = 0
        else:
            rls_emgc = 0
        return rls_emgc


    
    
                
        