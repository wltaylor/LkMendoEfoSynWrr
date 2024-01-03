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
# Date of Creation: 2020/05/30                              #
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from efo.lookup import LkupTblAnn
from efo.rule import RuleBase, RuleRlsSpecified
from efo.rule_fcst import RuleFcstBase
from efo.junction import JunctionRegulated, ReservoirJunction
from copy import copy, deepcopy
from inspect import currentframe, getframeinfo, getmodulename
from pathlib import Path


class RuleEfoBase(RuleFcstBase):
    def __init__(
            self, name, time_fcst,
             n_hrs_lead=None, rule_type=RuleBase.MIN, save_rls_sch=False,
            operating_junction=None, efo_resvr=None):
        # Call super class constructor
        super().__init__(
            name, time_fcst, n_hrs_lead=n_hrs_lead, rule_type=rule_type,
            save_rls_sch=save_rls_sch, operating_junction=operating_junction)
        if efo_resvr is not None:
            self.efo_resvr = efo_resvr
            self.n_members = len(efo_resvr)
            self.mbr_rls = np.full(self.t.t_cont.n_steps, np.nan)
            self.ts_rls = np.full(self.t.t_cont.n_steps, int(0))


class RuleFullEfo(RuleEfoBase):
    def __init__(self, name, time_fcst, efo_resvr, stor_thr, risk_tol, units,
                 rule_release_specified=None, efo_release_scheduler=None,
                 n_hrs_lead=None, operating_junction=None, vol_inc_factor=1.001):
        # Call super class constructor
        super().__init__(
            name, time_fcst, n_hrs_lead, operating_junction=operating_junction,
            efo_resvr=efo_resvr)
        # Set sub class properties
        self.rule_rls_spec = rule_release_specified
        self.efo_rls_scheduler = efo_release_scheduler
        self.stor_thr = float(stor_thr)
        self.risk_tol = risk_tol.flatten()
        self.units = units
        self.vol_inc_factor = vol_inc_factor

    def _calc_release_fcst(self, rls_sch_prop=None, rls_min_fcst=None, rls_max_fcst=None):
        # Initialize variables
        t = self.t.get_fcst_time()
        self.stor_fcst = np.full((t.n_steps, self.n_members, t.n_steps - 1), np.nan)
        self.stor_fcst[0, :, :] = self.efo_resvr[0].stor[0]
        self.stor_fcst_pre_rls = self.stor_fcst.copy()
        self.rls_unctrl_fcst = np.full((t.n_steps, self.n_members, t.n_steps - 1), np.nan)
        if self.efo_resvr[0].outlet_unctrl:
            self.rls_unctrl_fcst[0, :, :] = self.efo_resvr[0].rls_unctrl[0]
        else:
            self.rls_unctrl_fcst[0, :, :] = 0.
        self.rls_unctrl_pre_rls = self.rls_unctrl_fcst.copy()
        self.rls_fcst = np.zeros((t.n_steps, self.n_members, t.n_steps - 1))
        for i in range(t.n_steps - 1):
            self.rls_fcst[i + 2:, 0:self.n_members, i] = np.nan
        self.rls_fcst[0, :, :] = self.efo_resvr[0].rls_ctrl[0]
        self.rls_max = np.full((t.n_steps, self.n_members, t.n_steps - 1), np.nan)
        self.rls_no_constraint = self.rls_fcst.copy()
        self.pr_exc_thr = np.full(t.n_steps, np.nan)
        self.pr_exc_thr_pre_rls = np.full((t.n_steps, t.n_steps - 1), np.nan)
        self.i_risky_mbrs = np.full((t.n_steps, self.n_members), False)
        risky_select = np.arange(0, self.n_members, dtype=int)
        risky_mbr = risky_select.copy()
        i_risky = np.full(self.n_members, False)
        i_risky_chk = np.full(self.n_members, True)
        vol_abv_thresh = np.zeros([t.n_steps, self.n_members])
        vol2rls = np.zeros([t.n_steps, self.n_members])
        rls_fcst = np.full(t.n_steps, np.nan)
        rls_fcst[0] = self.efo_resvr[0].rls_ctrl[0]
        rls_today = np.empty((t.n_steps - 1, self.n_members))
        rls_sch = np.zeros(t.n_steps)
        for ts_lead in t.steps[1:self.n_steps_lead]:
            self.t.step = ts_lead
            if ts_lead > 1:
                self.stor_fcst[:, :, ts_lead - 1] = self.stor_fcst[:, :, ts_lead - 2]
                self.rls_unctrl_fcst[:, :, ts_lead - 1] = self.rls_unctrl_fcst[:, :, ts_lead - 2]
            for cur_mbr in risky_select:
                self.rule_rls_spec[cur_mbr].set_release(ts_lead, 0.)
                self.efo_resvr[cur_mbr].calc_qout()
                self.stor_fcst[ts_lead, cur_mbr, ts_lead - 1] = \
                    self.efo_resvr[cur_mbr].stor[ts_lead].item()
                self.rls_unctrl_fcst[ts_lead, cur_mbr, ts_lead - 1] = \
                    self.efo_resvr[cur_mbr].rls_ctrl[ts_lead].item()
                if self.efo_resvr[cur_mbr].outlet_unctrl:
                    self.rls_unctrl_fcst[ts_lead, cur_mbr, ts_lead - 1] += \
                        self.efo_resvr[cur_mbr].rls_unctrl[ts_lead].item()
            self.stor_fcst_pre_rls[:, :, ts_lead - 1] = self.stor_fcst[:, :, ts_lead - 1].copy()
            self.rls_unctrl_pre_rls[:, :, ts_lead - 1] = self.rls_unctrl_fcst[:, :, ts_lead - 1].copy()
            risky = True
            while risky:
                stor_plus_unctrl = self.stor_fcst[ts_lead, :, ts_lead - 1] \
                                 + self.rls_unctrl_fcst[ts_lead, :, ts_lead - 1] * self.units.flow2vol
                i_risky = (stor_plus_unctrl > self.stor_thr) & i_risky_chk
                self.pr_exc_thr[ts_lead] = np.sum(i_risky) / self.n_members
                if self.pr_exc_thr[ts_lead] - self.risk_tol[ts_lead] > 1 / self.n_members / 2:
                    self.pr_exc_thr_pre_rls[:, ts_lead - 1] = self.pr_exc_thr.copy()
                    i_risky_chk = i_risky.copy()
                    risky_mbr = np.where(i_risky)[0]
                    n_mbrs2reduce = \
                        int(len(risky_mbr) - np.floor(self.risk_tol[ts_lead] * self.n_members))
                    vol_abv_thresh[ts_lead, risky_mbr] = \
                        self.stor_fcst[ts_lead, risky_mbr, ts_lead - 1] + \
                        sum(self.rls_unctrl_fcst[1:ts_lead + 1, risky_mbr, ts_lead - 1]) * self.units.flow2vol \
                        - self.stor_thr
                    mbrs_sorted = np.argsort(vol_abv_thresh[ts_lead, risky_mbr])
                    risky_select = risky_mbr[mbrs_sorted[0:n_mbrs2reduce]]
                    self.i_risky_mbrs[ts_lead, risky_select] = True
                    vol2rls[ts_lead, risky_select] = \
                        self.vol_inc_factor * vol_abv_thresh[ts_lead, risky_select]
                    for cur_mbr in risky_select:
                        rls_fcst[1:ts_lead + 1] = np.sum(
                            vol2rls[:, cur_mbr]) * self.units.vol2flow / ts_lead
                        self.rls_no_constraint[1:ts_lead + 1, cur_mbr, ts_lead - 1] = \
                            rls_fcst[1:ts_lead + 1].copy()
                        rls_fcst, _, _ = self.efo_rls_scheduler[cur_mbr].calc_release_fcst(
                            rls_sch_prop=rls_fcst.copy(), rls_min_fcst=None, rls_max_fcst=None)
                        self.stor_fcst[1:ts_lead + 1, cur_mbr, ts_lead - 1] = \
                            self.efo_resvr[cur_mbr].stor[1:ts_lead + 1].copy()
                        if self.efo_resvr[cur_mbr].outlet_unctrl:
                            self.rls_unctrl_fcst[1:ts_lead + 1, cur_mbr, ts_lead - 1] = \
                                self.efo_resvr[cur_mbr].rls_unctrl[1:ts_lead + 1].copy()
                        else:
                            self.rls_unctrl_fcst[1:ts_lead + 1, cur_mbr, ts_lead - 1] = 0.
                        self.rls_max[1:ts_lead + 1, cur_mbr, ts_lead - 1] = \
                            self.efo_rls_scheduler[cur_mbr].rls_max_fcst[1:ts_lead + 1].copy()
                        if np.abs(np.sum(vol2rls[:, cur_mbr]) - np.sum(
                                rls_fcst[1:ts_lead + 1]) * self.units.flow2vol) > 0:
                            i_risky_chk[cur_mbr] = False
                            vol2rls[ts_lead, cur_mbr] -= \
                                (np.sum(vol2rls[:, cur_mbr]) - np.sum(rls_fcst[1:ts_lead + 1]) * self.units.flow2vol) \
                                - np.sum(self.rls_unctrl_fcst[1:ts_lead + 1, cur_mbr, ts_lead - 1])
                        self.rls_fcst[1:ts_lead + 1, cur_mbr, ts_lead - 1] = rls_fcst[1:ts_lead + 1].copy()
                    risky = True
                else:
                    if np.isnan(self.pr_exc_thr_pre_rls[ts_lead, ts_lead - 1]):
                        self.pr_exc_thr_pre_rls[:, ts_lead - 1] = self.pr_exc_thr.copy()
                    risky_select = np.arange(0, self.n_members, dtype=int)
                    risky_mbr = risky_select.copy()
                    i_risky_chk = np.full(self.n_members, True)
                    risky = False
        rls_today[:, :] = (np.nansum(
            self.rls_fcst[1:int(self.t.fcst_freq / self.t.n_hrs) + 1, :, :], axis=0) / (self.t.fcst_freq / self.t.n_hrs)).T
        if np.any(rls_today[:] > 0.):
            ts_max, mbr_max = np.where(rls_today == np.max(rls_today))
            self.ts_rls[self.t.t_cont.step] = int(ts_max[0]) + 1
            self.mbr_rls[self.t.t_cont.step] = int(mbr_max[0])
            self.release[self.t.t_cont.step] = self.rls_fcst[1, mbr_max[0], ts_max[0]]
            rls_sch = self.rls_fcst[:, mbr_max[0], ts_max[0]].copy()
            # Fill in the period to next Fcst if needed
            if np.any(np.isnan(rls_sch[:int(self.t.fcst_freq / self.t.n_hrs) + 1])):
                # Average release over the forecast cycle
                rls_sch[:int(self.t.fcst_freq / self.t.n_hrs) + 1] = \
                    np.nansum(rls_sch) / (self.t.fcst_freq / self.t.n_hrs)
        return rls_sch, None, None

