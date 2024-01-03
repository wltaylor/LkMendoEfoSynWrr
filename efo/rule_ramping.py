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
# Date of Creation: 2020/05/27                              #
#############################################################

import abc
from efo.rule import RuleBase
from efo.rule_compliance import RuleComplianceBase
from efo.lookup import LkupTbl


class RuleRampingBase(RuleComplianceBase):
    def __init__(self, name, time, operating_jnc, rls_sched, ramp_sched,
                 rule_type, time_unit='h', n_hrs=1):
         # Call super class constructor
        super().__init__(
            name, time, operating_jnc,
            rule_type=rule_type, release_type=RuleComplianceBase.CTRL_RLS)
        if time_unit != time.time_unit or n_hrs != time.n_hrs:
            if time_unit == 'h': ramp_sched = ramp_sched * (time.n_hrs / n_hrs)
        # Create lookup table
        self.rampTbl = LkupTbl(name +'_RampSched', rls_sched, ramp_sched)
    
    @classmethod
    def __subclasshook__(cls, C):
        if cls is RuleRampingBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented


class RuleDecreaseRateChng(RuleRampingBase):
    def __init__(self, name, time, operating_jnc, rls_sched, droc_sched, time_unit='h', n_hrs=1):
        # Call super class constructor
        super().__init__(name, time, operating_jnc, rls_sched, droc_sched, rule_type=RuleBase.MIN)
    
    def _get_rule_q(self):
        rls_prev = self.operating_jnc.rls_ctrl[max(0, self.t.step - 1)]
        rls_droc = rls_prev - self.rampTbl.get_val(rls_prev)
        return rls_droc


class RuleIncreaseRateChng(RuleRampingBase):
    def __init__(self, name, time, operating_jnc, rls_sched, iroc_sched, time_unit='h', n_hrs=1):
        # Call super class constructor
        super().__init__(name, time, operating_jnc, rls_sched, iroc_sched, rule_type=RuleBase.MAX)
    
    def _get_rule_q(self):
        rls_prev = self.operating_jnc.rls_ctrl[max(0, self.t.step - 1)]
        rls_iroc = rls_prev + self.rampTbl.get_val(rls_prev)
        return rls_iroc