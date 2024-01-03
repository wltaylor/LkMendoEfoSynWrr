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
from copy import copy
from efo.network import Network, NetworkFcst
from efo.rule import RuleBase, RuleRlsSpecified
from efo.time import TimeBase, TimeFcst
from efo.junction import Junction, JunctionRegulated, ReservoirJunction
from efo.qin import QinRouted, Qin, QinFcstPerfect
from efo.rule_compliance import RuleComplianceBase
from efo.rule_ramping import RuleRampingBase


class RuleDownstreamBase(RuleComplianceBase):
    def __init__(self, name, time, junctions, *, rule_type=RuleBase.MAX):
        self.junctions = junctions
        super().__init__(name, time, rule_type=rule_type)
        self.max_lag = 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is RuleDownstreamBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented


class RuleDwnStrmConstLag(RuleDownstreamBase):
    def __init__(self, name, time, junctions, rule_type=RuleBase.MAX):
        # Call super class constructor
        super().__init__(name, time, junctions, rule_type=rule_type)
        # Set class variable
        for cur_jnc in self.junctions[1:]:
            for cur_qin in cur_jnc.qin:
                if isinstance(cur_qin, QinRouted):
                    self.max_lag += cur_qin.reach_us.lag_time

    def _calc_delta(self, rls_prop):
        q_delta = np.empty([self.max_lag + 1, len(self.junctions[1:])])
        for i in range(0, self.max_lag + 1):
            self.junctions[0].set_qout(qout_specified=rls_prop, ts_offset=i)
            for j, cur_jnc in enumerate(self.junctions[1:]):
                if isinstance(cur_jnc, JunctionRegulated):
                    q_delta[i, j] = cur_jnc.calc_delta(rule_type=self.rule_type, ts_offset=i)
                else:
                    q_delta[i, j] = np.nan
                    cur_jnc.calc_qout()
        return q_delta
    
    def _get_rule_q(self):
        qDelta = self._calc_delta(rls_prop=0.) if self.rule_type == RuleBase.MIN\
            else -self._calc_delta(rls_prop=0.)
        maxDelta = np.nanmax(qDelta[:])
        return max(0., maxDelta)

            
                    
