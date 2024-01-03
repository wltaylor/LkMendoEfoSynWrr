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
from efo.rule import RuleBase
from efo.lookup import LkupTblInterp, LkupTblInterpPoly
import copy as cp


class OutletBase(RuleBase):
    def __init__(self, name, time, operating_junction, units):
        super().__init__(name, time, operating_junction, rule_type=RuleBase.MAX)
        # self.name = name
        # self.T = time
        self.rls_outlet = np.empty(self.t.n_steps)
        self.units = units

    @classmethod
    def __subclasshook__(cls, C):
        if cls is OutletBase:
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
    
    
class OutletRatedElev(OutletBase):
    def __init__(self, name, time, operating_junction, units, hypso, x_elev, y_q, *,
                 kind='linear', eval_outlet=True, fill_value=None):
        # Call super class constructor
        super().__init__(name, time, operating_junction, units)
        # Convert rating table to storage
        self.eval_outlet = eval_outlet
        if kind == 'infinite' or eval_outlet == False:
            self.eval_outlet = False
        else:
            x_stor = hypso.elev2stor(x_elev)
            self.stor_invert = np.min(x_stor)
            if kind == 'polyfit':
                self.stor_vs_q_tbl = LkupTblInterpPoly(name, x_stor, y_q, deg=2)
            else:
                self.stor_vs_q_tbl = LkupTblInterp(
                    name, x_stor, y_q, interp_type=kind, fill_value=fill_value)
        
    def _get_rule_q(self):
        if self.eval_outlet:
            stor_init = self.operating_jnc.stor[max(0, self.t.step - 1)]
            stor_delta = (self.operating_jnc.stor[self.t.step] - stor_init) / self.t.n_hrs
            rls_vol_tot = 0.
            stor_cur = stor_init
            for i in range(1, self.t.n_hrs + 1):
                if stor_cur > self.stor_invert:
                    rlsVolCur = max(
                        0., self.stor_vs_q_tbl.get_val(stor_cur)) * self.units.flow2vol / self.t.n_hrs
                    rls_vol_tot += rlsVolCur
                stor_cur = stor_init - rls_vol_tot + i*stor_delta
            q_outlet = rls_vol_tot*self.units.vol2flow
            self.rls_outlet[self.t.step] = q_outlet
        else:
            q_outlet = np.inf
        return q_outlet
    
        
        
    
