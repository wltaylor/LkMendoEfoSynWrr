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
from efo.lookup import LkupTblAnn
from efo.time import TimeBase


class RuleCurveBase(RuleBase):
    def __init__(self, name, time, units, rule_type=RuleBase.MIN, save_results=False):
        self.units = units
        super().__init__(name, time, rule_type=rule_type, save_results=save_results)

    @classmethod
    def __subclasshook__(cls,C):
        if cls is RuleCurveBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
    

class RuleCurve(RuleCurveBase):
    def __init__(self, name, time, operating_junction, units, month_day_hour, rule_curve,
                 hypso=None, curve_type='interp', time_unit=None, is_elev=True, ts_offset=0,
                 save_results=False, rule_type=RuleBase.MIN, ):
        # Call super class constructor
        super().__init__(name, time, units, rule_type=rule_type, save_results=save_results)
        # Class properties
        if hypso is None: hypso = operating_junction.hypso
        if is_elev is True:
            # Convert elev to storage
            rule_curve = np.interp(rule_curve, hypso.elev, hypso.stor)
        # Build lookup table
        if month_day_hour is None: month_day_hour = np.array([1, 1, 0])
        self.set_rule_curve(name, time, month_day_hour, rule_curve,
                            curve_type=curve_type, time_unit=time_unit)
        self.ts_rule_curve = np.empty(self.t.n_steps)
        # Create reference variable to rule parameters
        self.stor_eval = operating_junction.stor
        self.ts_offset = ts_offset

    def set_rule_curve(self, name, time, month_day_hour, rule_curve, *,
                       curve_type='interp', time_unit=None):
        self.rule_curve_tbl = LkupTblAnn(name + '_ruleCurve', time, month_day_hour, rule_curve,
                                         tbl_type=curve_type, time_unit=time_unit)
    
    def _get_rule_q(self):
        # Initialize flood release to zero
        rls_rule_curve = 0.
        # Check if storage exceeds rule curve
        self.ts_rule_curve[self.t.step] = self.rule_curve_tbl.get_val(self.t.cur_dt)
        stor_cur = self.stor_eval[min(self.t.end, self.t.step + self.ts_offset)]
        rls_rule_curve = max(0., (stor_cur - self.ts_rule_curve[self.t.step]) * self.units.vol2flow)
        return rls_rule_curve
                
        