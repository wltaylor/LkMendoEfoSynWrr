
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
import matplotlib.pyplot as plt
from efo.rule import RuleBase, RuleRlsSpecified
from efo.junction import ReservoirJunction


class RuleFcstBase(RuleBase):
    def __init__(
            self, name, time_fcst,
            n_hrs_lead=None, rule_type=RuleBase.MIN,
            save_rls_sch=False, operating_junction=None):
        # Call super class constructor
        super().__init__(
            name, time_fcst, rule_type=rule_type, operating_junction=operating_junction)
        self.release = np.zeros(time_fcst.t_cont.n_steps)
        self.release_fcst = np.full(self.t.t_cont.n_steps, np.nan)
        self.ts_rls = np.full(self.t.t_cont.n_steps, int(0))
        self.release_sched = np.zeros(self.t.n_steps)
        self._n_steps_lead = int(n_hrs_lead / time_fcst.n_hrs) + 1 if n_hrs_lead else time_fcst.n_steps
        self.save_rls_sch = save_rls_sch
        if save_rls_sch:
            self.rls_sch_archive = np.full((self.t.n_fcsts, self.t.shape[1]), np.nan)
            self.rls_sch_dt = np.full(self.t.n_fcsts, np.nan)
    
    @property
    def n_steps_lead(self):
        return min(self._n_steps_lead, self.t.n_steps)

    def _get_rule_q(self):
        if np.any(self.t.fcst_dates == self.t.t_cont.cur_dt):
            if self.rls_type == self.CTRL_RLS:
                rls_prev = self.operating_jnc.rls_ctrl[max(0, self.t.t_cont.step - 1)]
            else:
                rls_prev = self.operating_jnc.rlsTot[max(0, self.t.t_cont.step - 1)]
            self.release_sched, _, _ = self.calc_release_fcst(
                rls_sch_prop=np.hstack((rls_prev, np.zeros(self.n_steps_lead - 1))),
                rls_min_fcst=np.zeros(self.n_steps_lead),
                rls_max_fcst=np.full(self.n_steps_lead, np.inf),
                )
            release = self.release_sched[1].item()
        else:
            idx_fcst_ts = min(self.t.get_fcst_step(), np.where(~np.isnan(self.release_sched))[0][-1])
            release = self.release_sched[idx_fcst_ts].item() if idx_fcst_ts else 0.
            self.release_fcst[self.t.t_cont.step] = release
        return release
    
    def _set_release(self, release):
        self.release[self.t.t_cont.step] = release

    def calc_release_fcst(self, rls_sch_prop, rls_min_fcst, rls_max_fcst):
        idx_lead = np.where(~np.isnan(rls_sch_prop[1:]))[0][-1] + 2
        rls_sch_prop = rls_sch_prop[:idx_lead]
        rls_min_fcst = rls_min_fcst[:idx_lead] if rls_min_fcst is not None else None
        rls_max_fcst = rls_max_fcst[:idx_lead] if rls_max_fcst is not None else None
        rls_sch, rls_min_fcst, rls_max_fcst = \
            self._calc_release_fcst(rls_sch_prop, rls_min_fcst, rls_max_fcst)
        nan_append = np.full(self.t.n_steps - rls_sch.shape[0], np.nan)
        rls_sch = np.hstack((rls_sch, nan_append))
        self.rls_min_fcst = np.hstack((rls_min_fcst, nan_append)) if rls_min_fcst is not None else None
        self.rls_max_fcst = np.hstack((rls_max_fcst, nan_append)) if rls_max_fcst is not None else None
        self.release_fcst[self.t.t_cont.step: self.t.t_cont.step + int(self.t.fcst_freq / self.t.n_hrs)] = \
            rls_sch[1: int(self.t.fcst_freq / self.t.n_hrs) + 1].copy()
        if self.save_rls_sch:
            self.rls_sch_archive[self.t.t_cont.cur_dt == self.t.fcst_dates, :] = rls_sch
            self.rls_sch_dt[
                self.t.t_cont.cur_dt == self.t.fcst_dates] = self.t.t_cont.cur_dt.to_numpy()
        return rls_sch, self.rls_min_fcst, self.rls_max_fcst


    @abc.abstractmethod
    def _calc_release_fcst(self, rls_sch_prop, rls_min_fcst, rls_max_fcst):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is RuleFcstBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented

    def _get_adjusted_release_eqldist(self, rls_sch_prop, rls_min_fcst, rls_max_fcst):
        rls_tot = np.sum(rls_sch_prop)
        rls_min_tot = np.sum(rls_min_fcst)
        if rls_min_tot > rls_tot:
            rls_fcst = rls_min_fcst.copy()
        else:
            rls_fcst = rls_sch_prop.copy()
            i_less_min = rls_sch_prop < rls_min_fcst
            if np.any(i_less_min):
                rls_fcst[i_less_min] = \
                    rls_min_fcst[i_less_min] + \
                    (np.sum(rls_sch_prop[i_less_min]) - \
                     np.sum(rls_min_fcst[i_less_min])) / len(rls_sch_prop[i_less_min])
            rls_fcst_check = np.zeros(len(rls_fcst))
            i_val = ~np.isnan(rls_fcst) & ~np.isnan(rls_max_fcst)
            i_maxd = np.full(len(rls_fcst), False)
            i_mind = np.full(len(rls_fcst), False)
            while np.any(i_maxd==False) and np.any(rls_fcst[i_val] - rls_fcst_check[i_val] != 0.):
                rls_fcst_check = np.copy(rls_fcst)
                # 1 Check for min
                i_less_min = rls_fcst < rls_min_fcst
                i_mind = i_mind | i_less_min
                rls_min_vol = np.sum(rls_min_fcst[i_less_min] - rls_fcst[i_less_min])
                rls_fcst[i_less_min] = rls_min_fcst[i_less_min].copy()
                rls_subtract = rls_min_vol/max(1,np.sum(~i_mind & i_val))
                rls_fcst[~i_mind] -= rls_subtract
                # 2 Check for less than zero
                i_less_zero = rls_fcst < 0.
                rls_add = -np.sum(rls_fcst[i_less_zero])
                rls_fcst[i_less_zero] = 0.
                rls_fcst[~i_less_zero] += rls_add
                # 3 Check for max
                i_grtr_max = rls_fcst > rls_max_fcst
                i_maxd = i_maxd | i_grtr_max
                rls_max_vol = np.sum(rls_fcst[i_grtr_max] - rls_max_fcst[i_grtr_max])
                rls_fcst[i_grtr_max] = rls_max_fcst[i_grtr_max]
                rls_add = rls_max_vol/max(1,np.sum(~i_maxd & i_val))
                rls_fcst[~i_maxd] += rls_add
        return rls_fcst


class RuleFcstReleaseSched(RuleFcstBase):
    def __init__(
            self, name, time_fcst, fcst_resvr, fcst_network, units,
            rule_release_specified=None, max_itr=1000, stor_max=None,
            n_hrs_lead=None, conv_crit=1e-3, operating_junction=None):
        # Call super class constructor
        super().__init__(name, time_fcst, n_hrs_lead, operating_junction=operating_junction)
        # Set sub class properties
        self.fcst_net = fcst_network
        self.fcst_resvr = fcst_resvr
        self.rule_rls_specified = rule_release_specified
        self.max_itr = max_itr
        self.conv_crit = conv_crit
        self.stor_max = stor_max if stor_max else fcst_resvr.stor_max
        self.units = units
        self.rls_min_fcst = np.full(self.t.n_steps, np.nan)
        self.rls_max_fcst = np.full(self.t.n_steps, np.nan)
        
    def _calc_release_fcst(self, rls_sch_prop, rls_min_fcst=None, rls_max_fcst=None):
        ts_lead = rls_sch_prop.shape[0] - 1
        n_steps_fcst_cyc = int(np.ceil(self.t.fcst_freq / self.t.n_hrs))
        cur_itr = 0
        # Create reference variables to ensemble reservoir objects
        rls_max = self.fcst_resvr.rule_stack.rls_max[1:ts_lead + 1].copy()
        rls_min = self.fcst_resvr.rule_stack.rls_min[1:ts_lead + 1].copy()
        rls_ctrl_applied = self.fcst_resvr.rls_ctrl[1:ts_lead + 1]
        stor_fcst = self.fcst_resvr.stor[1:ts_lead + 1]
        if self.fcst_resvr.outlet_unctrl:
            rls_unctrl_fcst = self.fcst_resvr.rls_unctrl[1:ts_lead + 1]
        else:
            rls_unctrl_fcst = np.zeros(ts_lead)
        rls_fcst = rls_sch_prop[1:ts_lead + 1].copy()
        rls_fcst_init = np.copy(rls_fcst)
        rls_fcst_prev = np.array([np.copy(rls_fcst[:n_steps_fcst_cyc])])
        rls_abv_stor_max = np.zeros(rls_fcst.size)
        rls_max_adj = np.full(rls_fcst.size, np.inf)
        while True:
            # Recaculate storage
            for i in range(1, ts_lead+1):
                if issubclass(type(self.fcst_resvr), ReservoirJunction):
                    self.t.step = i
                    if issubclass(type(self.rule_rls_specified), RuleRlsSpecified):
                        self.rule_rls_specified.set_release(i, rls_fcst[i - 1].item())
                    self.fcst_net.process_fcst_junctions()
                    rls_max[i-1] = self.fcst_resvr.rule_stack.rls_max[i].item()
                    rls_min[i-1] = self.fcst_resvr.rule_stack.rls_min[i].item()
                    # Account for storage above spillway
                    # We're reducing the max release by the amount that exceeds our storage target
                    # so the release above max can be redistributed to another fcst timestep
                    if self.rls_type == RuleBase.TOT_RLS:
                        rls_abv_stor_max[i-1] = max(
                            0, (stor_fcst[i-1] - self.stor_max) * self.units.vol2flow + rls_unctrl_fcst[i - 1])
                        if rls_abv_stor_max[i-1] > 0.:
                            if i > 1:
                                rls_max_adj[i-1] = rls_ctrl_applied[i-1] + rls_abv_stor_max[i-1] - rls_abv_stor_max[i-2]
                            else:
                                rls_max_adj[i-1] = rls_ctrl_applied[i-1] + rls_abv_stor_max[i-1]
                            rls_max[i-1] = max(0., min(rls_max[i-1], rls_max_adj[i-1]))
                            rls_min[i-1] = max(0., min(rls_min[i-1], rls_max_adj[i-1]))
            if rls_max_fcst is not None:
                rls_max = np.min(np.vstack((rls_max, rls_max_fcst[1:])), axis=0)
                rls_min[rls_min > rls_max] = rls_max[rls_min > rls_max]
            diff_from_prev =  rls_fcst_prev[-1, :] - rls_ctrl_applied[:n_steps_fcst_cyc]
            if (cur_itr > 0 and (np.all(np.abs(diff_from_prev) < self.conv_crit, axis=0) or cur_itr >= self.max_itr)) \
                    or (sum(rls_fcst_init) - sum(rls_ctrl_applied) < self.conv_crit):
                rls_sch_return = np.hstack((rls_sch_prop[0], rls_ctrl_applied))
                self.rls_min_fcst = np.hstack((np.nan, rls_min, np.full(self.t.n_steps - len(rls_min) - 1, np.nan)))
                self.rls_max_fcst = np.hstack((np.nan, rls_max, np.full(self.t.n_steps - len(rls_min) - 1, np.nan)))
                break
            rls_fcst_prev = np.vstack((rls_fcst_prev, np.copy(rls_ctrl_applied[:n_steps_fcst_cyc])))
            rls_fcst = self._get_adjusted_release_eqldist(rls_fcst_init, rls_min, rls_max)
            cur_itr += 1
        return rls_sch_return, np.hstack((np.nan, rls_min)), np.hstack((np.nan, rls_max))



    

    


    


     