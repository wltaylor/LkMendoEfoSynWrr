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
# Date of Creation: 2020/04/29                              #
#############################################################


import abc
import numpy as np
from efo.units import UnitsBase
from efo.time import TimeBase
from efo.hypso import HypsoBase
from efo.outlet import OutletRatedElev
from efo.rule import RuleBase, RuleStack
import copy as cp


class JunctionBase(metaclass=abc.ABCMeta):
    def __init__(self, name, time, qin=[]):
        self.name = name
        super().__init__()
        # Set class properties
        self._reach_callback = None
        self.set_qin(qin)
        if issubclass(type(time), TimeBase):
            self.t = time
            self.qout = np.full(self.t.n_steps, np.nan)
            self.qout[0] = 0.
            self.qin_tot = np.full(self.t.n_steps, np.nan)
    
    def set_qin(self, qin):
        self.qin = qin
        
    def append_qin(self, q_in):
        self.qin.append(q_in)

    def _get_qin(self, *, ts_offset=0):
        q_tot = 0.
        # print(self.name)
        for qin_cur in self.qin:
            if qin_cur:
                q_tot += qin_cur.get_qin(ts_offset)
        try:
            self.qin_tot[self.t.step] = q_tot
        except:
            print('e')    
        return q_tot
    
    def get_qout(self, *, ts_offset=0):
        return self.qout[min(self.t.end, max(0, self.t.step + ts_offset))].item()

    @abc.abstractmethod
    def calc_qout(self):
        qout = max(0., self._get_qin())
        self.qout[self.t.step] = qout
        return qout
    
    def set_qout(self, qout_specified, *, ts_offset=0):
        self.qout[min(self.t.end, self.t.step + ts_offset)] = qout_specified

    @classmethod
    def __subclasshook__(cls, C):
        if cls is JunctionBase:
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
    
    def bind_to_reach(self, reach_callback):
        self._reach_callback = reach_callback
        
    def get_ds_reach(self):
        return self._reach_callback() if self._reach_callback else None


class Junction(JunctionBase):
    def __init__(self, name, time, qin=[]):
        # Call super class constructor
        super().__init__(name, time, qin=qin)
        
    def calc_qout(self):
        return super().calc_qout()
        
        
class JunctionRegulated(JunctionBase):
    def __init__(self, name, time, qin,
                 rule_min_q=None, rule_max_q=None, rule_diversion=[]):
        # Call super class constructor
        super().__init__(name, time, qin)
        # Set class properties
        self.rule_minq = rule_min_q
        self.rule_maxq = rule_max_q
        self.set_diversions(rule_diversion)
        self.continuity = np.full(self.t.n_steps, np.nan)

    def set_rule_minq(self, rule_minq):
        self.rule_minq = rule_minq

    def set_rule_maxq(self, rule_maxq):
        self.rule_maxq = rule_maxq
        
    def set_diversions(self, rule_diversion):
        self.rule_diversion = rule_diversion
        if type(rule_diversion) == list:
            self.qdiv = np.full(self.t.n_steps, np.nan)
    
    def _calc_q(self, qin, qdiv, rule):
        qout = qin - qdiv
        if qout < 0.:
            qdiv += qout
            rule.release[self.t.step] = qdiv
            qout = 0.
        return qout, qdiv
    
    def calc_water_balance(self, qin, qout):
        qout = self._calc_water_balance(qin, qout)
        return qout

    def _calc_water_balance(self, qin, qout):
        qdiv = self.qdiv[self.t.step]
        qout = qin - self.qdiv[self.t.step]
        if qout < 0.:
            qdiv += qout
            qout = 0.
            self.qdiv[self.t.step] = qdiv
        return qout, qdiv
   
    def calc_delta(self, *, rule_type=RuleBase.MAX, ts_offset=0):
        if rule_type == RuleBase.MAX and self.rule_maxq:
            target_q = self.rule_maxq.get_rule_q()
        elif rule_type == RuleBase.MIN and self.rule_minq:
            target_q = self.rule_minq.get_rule_q()
        else:
            target_q = np.nan
        qdiv = 0.
        qin_init = self._get_qin(ts_offset=ts_offset)
        if self.rule_diversion:
            for curRule in self.rule_diversion:
                qdiv += curRule.get_rule_q()
        qin_net = qin_init - qdiv
        self.qout[min(self.t.end, self.t.step + ts_offset)] = max(0., qin_net)
        return qin_net - target_q if rule_type == RuleBase.MAX else target_q - qin_net

    def calc_qout(self):
        qin_tot = super().calc_qout()
        qdiv = 0.
        qout = qin_tot
        qmin = self.rule_minq.get_rule_q() if self.rule_minq is not None else 0.
        qdiv_max = qin_tot - qmin
        if self.rule_diversion:
            for cur_rule in self.rule_diversion:
                # cur_div, is_ctrl = cur_rule.calc_release(rlsProposed=max(0., qdiv_max-qDiv), qIn=qInTot)
                cur_div, is_ctrl = cur_rule.calc_release(rls_prop=max(0., qdiv_max - qdiv))
                qout, cur_div = self._calc_q(qout, cur_div, cur_rule)
                qdiv += cur_div
            self.qdiv[self.t.step] = qdiv
        self.continuity[self.t.step] = qout + qdiv - qin_tot
        self.qout[self.t.step] = qout
        return qout


class ReservoirJunction(JunctionRegulated):
    
    def __init__(self, name, time, qin, stor_init, units,
                 *, hypso=None, rules=[], evap_obj=None, seepage_obj=None,
                 outlet_ctrl=None, outlet_unctrl=None,
                 stor_inactive=0., stor_top_dam=np.inf, stor_max=None,
                 elev_inactive=None, elev_top_dam=None, elev_max=None, conv_crit=10e-3,
                 release_init=None):
        # Call super class constructor
        super().__init__(name, time, qin)
        # Set class properties
        self.set_rule_stack(rules)
        self.set_ctrl_outlet(outlet_ctrl)
        self.set_unctrl_outlet(outlet_unctrl)
        self.stor_inactive = stor_inactive
        self.stor_max = stor_max
        if issubclass(type(units), UnitsBase):
            self.units = units
        if issubclass(type(hypso), HypsoBase):
            self.hypso = hypso
        if elev_top_dam is not None:
            self.stor_top_dam = self.hypso.elev2stor(elev_top_dam)
        else:
            self.stor_top_dam = stor_top_dam
        # Create storage array
        self.stor = np.full(self.t.n_steps, np.nan)
        self.stor_init = stor_init
        self.stor[0] = stor_init
        # Other propoerties
        self.set_evap(evap_obj)
        self.set_seepage(seepage_obj)
        self.qin_net = np.nan
        self.conv_crit = conv_crit
        if release_init is not None:
            self.qout[0] = release_init
        self.release_init = release_init

    def create_ctrl_outlet(
            self, name, elev, q_outlet, eval_outlet=True, kind='linear', release_ctrl_init=0.,
            fill_value=None):
        hypso = self.hypso if eval_outlet else None
        outletCtrl = OutletRatedElev(self.name +'_ctrlOutlet_' + name,
                                     self.t, self, self.units, hypso, elev, q_outlet,
                                     eval_outlet=eval_outlet, kind=kind, fill_value=fill_value)
        self.set_ctrl_outlet(outletCtrl, release_ctrl_init=release_ctrl_init)

    def create_unctrl_outlet(self, name, elev, q_outlet, kind='linear', fill_value='extrapolate'):
        outlet_unctrl = OutletRatedElev(
            self.name+'_unCtrlOutlet_'+name, self.t, self, self.units,
            self.hypso, elev, q_outlet, kind=kind, fill_value=fill_value)
        self.set_unctrl_outlet(outlet_unctrl)

    def set_ctrl_outlet(self, outlet_ctrl, release_ctrl_init=0.):
        self.outlet_ctrl = outlet_ctrl
        if outlet_ctrl:
            self.rls_ctrl = np.full(self.t.n_steps, np.nan)
            self.rls_ctrl[0] = release_ctrl_init

    def set_unctrl_outlet(self, outlet_unctrl):
        self.outlet_unctrl = outlet_unctrl
        if outlet_unctrl: self.rls_unctrl = np.full(self.t.n_steps, np.nan)

    def set_evap(self, evap_obj):
        self.evap_obj = evap_obj
        if evap_obj: self.loss_evap = np.full(self.t.n_steps, np.nan)
        
    def set_seepage(self, seepage_obj):
        self.seepage_obj = seepage_obj
        if seepage_obj: self.loss_seep = np.full(self.t.n_steps, np.nan)
        
    def set_stor_init(self, storInit):
        self.stor_init = storInit
        self.stor[0] = storInit
            
    def set_rule_stack(self, rules):
        if type(rules) == list:
            self.rule_stack = RuleStack(self.name + '_ruleStack', self.t, self, rules)
            
    def append_rule(self, rule):
        self.rule_stack.append_rule(rule)
    
    def insert_rule(self, rule, idx=0):
        self.rule_stack.insert_rule(rule, idx=idx)
            
    def calc_delta(self, *, rule_type=RuleBase.MAX, ts_offset=0):
        if self.stor_max:
            stor_prev = self.stor[max(0, self.t.step - 1)].item()
            qin_net = self._get_qin(ts_offset=ts_offset)
            loss_evap, qin_net = self._calc_evap(stor_prev, qin_net)
            loss_seep, qin_net = self._calc_seepage(stor_prev, qin_net)
            q_div, qin_net = self._calc_diversions(stor_prev, qin_net)
            qdelta_ret = -max(0., self.stor_max - stor_prev + qin_net * self.units.flow2vol) * self.units.vol2flow
        else: 
            qdelta_ret = np.nan
        self.calc_qout()
        return qdelta_ret

    def _calc_storage(self, stor_prev, qin_net, qout):
        stor = stor_prev - qout * self.units.flow2vol + qin_net * self.units.flow2vol
        if stor < self.stor_inactive:
            qout += (stor - self.stor_inactive) * self.units.vol2flow
            if qout < 0.: qout = 0.
            stor = self.stor_inactive
        elif stor > self.stor_top_dam:
            qout += (stor - self.stor_top_dam) * self.units.vol2flow
            stor = self.stor_top_dam
        return stor, qout
    
    def _calc_water_balance(self, qin, qout):
        stor, qout = self._calc_storage(self.stor[max(0, self.t.step - 1)], qin, qout)
        return qout
    
    def _calc_evap(self, stor, qin):
        if self.evap_obj:
            loss_evap = self.evap_obj.calc_evap(stor + qin * self.units.flow2vol)
            stor, loss_evap = self._calc_storage(stor, qin, loss_evap)
            qin -= loss_evap
        else:
            loss_evap = 0.
        return loss_evap, qin
    
    def _calc_seepage(self, stor, qin):
        if self.seepage_obj:
            lossSeep = self.seepage_obj.calc_seepage(stor + qin * self.units.flow2vol)
            stor, lossSeep = self._calc_storage(stor, qin, lossSeep)
            qin -= lossSeep
        else:
            lossSeep = 0.
        return lossSeep, qin
    
    def _calc_diversions(self, stor, qin):
        qdiv = 0.
        stor_cur = stor
        if self.rule_diversion:
            for cur_rule in self.rule_diversion:
                cur_div, isCtrl = cur_rule.calc_release(
                    rls_prop=qin + stor * self.units.vol2flow,
                    rlsPrev=cur_rule.release[max(0, self.t.step - 1)].item(),
                    stor=stor, qIn=qin)
                stor_cur, cur_div = self._calc_storage(stor, qin, cur_div)
                qin -= cur_div
                qdiv += cur_div
        return qdiv, qin

    def calc_qout(self):
        qout_tot_cur = 0.
        self.qin_net = qin_init = super()._get_qin()
        if self.t.step==0:
            stor_prev = self.stor_init
        else:
            stor_prev = self.stor[self.t.step - 1].item()
        stor_cur = stor_prev
        # Get evaporation
        loss_evap, self.qin_net = self._calc_evap(stor_prev, self.qin_net)
        if self.evap_obj: self.loss_evap[self.t.step] = loss_evap
        # Get seepage
        loss_seep, self.qin_net = self._calc_seepage(stor_prev, self.qin_net)
        if self.seepage_obj: self.loss_seep[self.t.step] = loss_seep
        # Calc diversions
        qdiv, self.qin_net = self._calc_diversions(stor_prev, self.qin_net)
        if self.rule_diversion: self.qdiv[self.t.step] = qdiv
        # Get uncontrolled release
        stor_cur, qout_tot_cur = self._calc_storage(stor_prev, self.qin_net, qout_tot_cur)
        if self.outlet_unctrl: self.rls_unctrl[self.t.step] = qout_tot_cur
        self.stor[self.t.step] = stor_cur
        rls_unctrl = qout_tot_start = qout_tot_cur
        rls_stack = 0.
        if self.outlet_ctrl:
            while True:
                # Spill first approach
                rls_unctrl_prev = rls_unctrl
                if self.outlet_unctrl:
                    rls_unctrl = qout_tot_start + self.outlet_unctrl.calc_release(np.inf)[0]
                    self.rls_unctrl[self.t.step] = rls_unctrl
                else: 
                    rls_unctrl = 0.
                self.stor[self.t.step], qout_tot_cur = self._calc_storage(
                    stor_prev, self.qin_net, qout_tot_start + rls_unctrl)
                rls_stack, is_ctrl = self.rule_stack.calc_release(0.)
                # Make sure we can make this release with constraints of controlled outlet
                if rls_stack > 0:
                    rls_stack, is_outlet_max = self.outlet_ctrl.calc_release(rls_stack)
                    if self.outlet_ctrl.q_rule_cur < self.rule_stack.rls_max[self.t.step]:
                        self.rule_stack.rls_max[self.t.step] = self.outlet_ctrl.q_rule_cur
                        if self.rule_stack.rls_min[self.t.step] > self.outlet_ctrl.q_rule_cur:
                            self.rule_stack.rls_min[self.t.step] = self.outlet_ctrl.q_rule_cur
                    if is_outlet_max:
                        self.rule_stack.ctrl_rule[self.t.step] = -1
                # Check if there is a defined initial release
                if self.t.step == 0 and self.release_init is not None:
                    rls_stack = max(rls_stack, self.release_init)
                self.stor[self.t.step], qout_tot_cur = self._calc_storage(
                    stor_prev, self.qin_net, qout_tot_start + rls_stack)
                if qout_tot_cur < qout_tot_start + rls_stack:
                    self.rule_stack.rls_max[self.t.step] = qout_tot_cur
                if abs(rls_unctrl - rls_unctrl_prev) < self.conv_crit:
                    break
        else:
            rls_unctrl = self.outlet_unctrl.calc_release(np.inf)[0] if self.outlet_unctrl else 0.
        # KEEP THIS! It may not look correct but it is.
        self.stor[self.t.step], qout_tot_cur = self._calc_storage(stor_prev, self.qin_net, qout_tot_start + rls_stack + rls_unctrl)
        if self.outlet_ctrl:
            self.rls_ctrl[self.t.step] = rls_stack
        self.qout[self.t.step] = qout_tot_cur
        self.continuity[self.t.step] = \
            (self.stor[self.t.step] - self.stor[max(self.t.step - 1, 0)]) * self.units.vol2flow + \
            qout_tot_cur - qin_init + loss_evap + loss_seep + qdiv
        return qout_tot_cur
