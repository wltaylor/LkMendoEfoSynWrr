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
from efo.rule import RuleBase
from efo.qin import QinSpecified
import numpy as np
from efo.lookup import LkupTbl, LkupTblAnn, LkupTblElev, LkupTblInterp
from scipy import interpolate


class RuleComplianceBase(RuleBase):
    
    def __init__(self, name, time, operating_junction=None, *, rule_type, release_type=RuleBase.CTRL_RLS):
        super().__init__(name, time, operating_junction, rule_type=rule_type, release_type=release_type)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is RuleComplianceBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
    

class RuleMinQ(RuleComplianceBase):
    def __init__(self, name, time, min_q, *, operating_junction=None):
        # Call super class constructor
        super().__init__(name, time, operating_junction, rule_type=RuleBase.MIN)
        # Class properties
        self.min_q = min_q
    
    def _get_rule_q(self):
        return self.min_q


class RuleMaxQ(RuleComplianceBase):
    def __init__(self, name, time, max_q,
                 operating_junction=None, pct_buffer=0., release_type=RuleBase.CTRL_RLS):
        # Call super class constructor
        super().__init__(name, time, operating_junction, rule_type=RuleBase.MAX, release_type=release_type)
        # Class properties
        self.pct_buffer = pct_buffer
        self.max_q = max_q - max_q * pct_buffer

    def _get_rule_q(self):
        return self.max_q


class RuleMaxLkupTblElev(RuleComplianceBase):
    def __init__(self, name, time, operating_junction, x_elev, y_q,
                 release_type=RuleBase.CTRL_RLS, hypso=None):
        # Call super class constructor
        super().__init__(name, time, operating_junction, rule_type=RuleBase.MAX, release_type=release_type)
        # Build lookup table
        if hypso is None: hypso = operating_junction.hypso
        self.comp_q_tbl = LkupTblElev(name + '_table', hypso, x_elev, y_q)
    
    def _get_rule_q(self):
        return self.comp_q_tbl.get_val(self.operating_jnc.stor[max(0, self.t.step - 1)])


class RuleMinAnnSched(RuleComplianceBase):
    def __init__(self, name, time, month_day_hour, min_q_sched, *,
                 release_type=RuleBase.CTRL_RLS, rule_type=RuleBase.MIN,
                 tbl_type='step', time_unit=None, hydro_cond=None, ts_offset=0):
         # Call super class constructor
        super().__init__(name, time, rule_type=rule_type)
        # Class properties
        self.comp_q_sched = LkupTblAnn(
            name + '_sched', time, month_day_hour, min_q_sched, tbl_type=tbl_type, time_unit=time_unit)
        self.hydro_cond = hydro_cond
        self.ts_offset = ts_offset
    
    def _get_rule_q(self):
        cur_dt = self.t.get_datetime_offset(self.ts_offset)
        if self.hydro_cond is None:
            return self.comp_q_sched.get_val(cur_dt)
        else:
            hc = self.hydro_cond.get_hydrologic_cond(ts_offset=self.ts_offset)
            return self.comp_q_sched.get_val(cur_dt, self.hydro_cond.cur_cond)

class RuleMaxAnnSched(RuleMinAnnSched):
    def __init__(self, name, time, month_day_hour, max_q_sched, *,
                 release_type=RuleBase.CTRL_RLS, tbl_type='step', time_unit=None, hydro_cond=None):
        # Call super class constructor
        super().__init__(
            name, time, month_day_hour, max_q_sched,
            release_type=release_type, rule_type=RuleBase.MAX,
            tbl_type=tbl_type, time_unit=time_unit, hydro_cond=hydro_cond)
        # Class properties
        self.comp_q_sched = LkupTblAnn(name + '_sched', time, month_day_hour, max_q_sched,
                                       tbl_type=tbl_type, time_unit=time_unit)
        self.hydro_cond = hydro_cond
    

class RuleDiversionNetwork(RuleComplianceBase):
    def __init__(self, name, time, *, rule_max, jnc_diversion):
        # Call super class constructor
        super().__init__(name, time, rule_type=RuleBase.MAX)
        self.rule_max_div = rule_max
        self.jnc_diversion = jnc_diversion
        self.qin_specified = QinSpecified(name + '_qInSpecified', time)
        self.qin_specified.set_qin(ts_offset=0, q_specified=0.)
        self.jnc_diversion.append_qin(self.qin_specified)
        
    def _get_rule_q(self):
        qmax_div = self.rule_max_div.get_rule_q()
        q_demand = -self.jnc_diversion.calc_delta(rule_type=self.ruleType)
        return min(qmax_div, q_demand)
    
    def calc_release(self, rls_prop):
        rls_prop, is_ctrl = super().calc_release(rls_prop)
        qdiv = min(self.get_rule_q(), rls_prop)
        self.qin_specified.set_qin(ts_offset=0, q_specified=qdiv)
        return qdiv, True if qdiv != rls_prop else False


class HydrologicCond:
    def __init__(self, name):
        self.name = name
        # Set class properties
        self._cur_cond = np.nan

    @property
    def cur_cond(self):
        return self._cur_cond

    @cur_cond.setter
    def cur_cond(self, cur_hydro_cond):
        self._cur_cond = cur_hydro_cond

    @abc.abstractmethod
    def get_hydrologic_cond(self, ts_offset):
        pass
        

class RuleQinMaxLkupTbl(RuleComplianceBase):
    def __init__(self, name, time, operating_junction, x_qin_max, y_rls, *,
                 period_hrs=24, interp_type='step', release_type=RuleBase.CTRL_RLS,
                 elev_chng=0., elev_ave_per=1):
         # Call super class constructor
        super().__init__(name, time, operating_junction, rule_type=RuleBase.MAX)
        if interp_type== 'interp':
            self.qin_max_tbl = LkupTblInterp(name + '_QinMax', x_qin_max, y_rls)
        else:
            self.qin_max_tbl = LkupTbl(name + '_QinMax', x_qin_max, y_rls)
        self.period = int(period_hrs / time.n_hrs)
        self.elev_chng = elev_chng
        self.elev_ave_per = int(elev_ave_per)
    
    def _get_rule_q(self):
        is_evald = True
        if self.elev_chng != 0.:
            elev_prev = self.operating_jnc.hypso.stor2elev(
                self.operating_jnc.stor[max(0, self.t.step - self.elev_ave_per)])
            elev_cur = self.operating_jnc.hypso.stor2elev(
                self.operating_jnc.stor[self.t.step])
            elev_diff = elev_cur - elev_prev
            is_evald = ((elev_diff > 0.) & (elev_diff <= self.elev_chng)) | \
                      (self.is_ctrl & ((elev_diff <= self.elev_chng) | (elev_diff >= -self.elev_chng)))
        if is_evald:
            q_per_max = np.max(self.operating_jnc.qin_tot[max(0, self.t.step - self.period + 1):self.t.step + 1])
        else:
            q_per_max = np.inf
        return self.qin_max_tbl.get_val(q_per_max)





        
    
    
        
    

