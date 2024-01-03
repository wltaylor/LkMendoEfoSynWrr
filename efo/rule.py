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
from efo.time import TimeBase, TimeCont
import copy as cp


class RuleBase(metaclass=abc.ABCMeta):
    MIN = int(0)
    MAX = int(1)
    CTRL_RLS = int(0)
    TOT_RLS = int(1)

    def __init__(self, name, time, operating_junction=None, *, rule_type=MIN, release_type=CTRL_RLS,
                 save_results=False):
        self.name = name
        self.sub_rule = None
        self.rule_type = rule_type
        self.rls_type = release_type
        self.q_rule_cur = 0. if rule_type == self.MIN else np.inf
        self.operating_jnc = operating_junction
        self.save_results = save_results
        self.is_ctrl = False
        self._observers = []
        if issubclass(type(time), TimeBase):
            self.t = time
            self.release = np.zeros(time.n_steps)
        if self.save_results: self.q_rule = np.full(time.n_steps, np.nan)
        super().__init__()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is RuleBase:
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

    def set_subrule(self, sub_rule):
        self.sub_rule = sub_rule

    def set_ctrl_rule(self, condition):
        self.is_ctrl = condition

    @abc.abstractmethod
    def _get_rule_q(self):
        return

    def get_rule_q(self):
        self.q_rule_cur = self._get_rule_q()
        if self.rls_type == self.TOT_RLS:
            self.q_rule_cur = max(0., self.q_rule_cur - self.operating_jnc.rls_unctrl[
                self.t.step].item()) if self.rule_type == self.MAX \
                else min(self.q_rule_cur,
                         max(0., self.q_rule_cur - self.operating_jnc.rls_unctrl[self.t.step].item()))
        if self.save_results: self.q_rule[self.t.step] = self.q_rule_cur
        return self.q_rule_cur

    def calc_release(self, rls_prop):
        is_ctrl = False
        if self.sub_rule:
            rls_prop, is_ctrl = self.sub_rule.calc_release(rls_prop)
        q_comp = self.get_rule_q()
        release = min(rls_prop, q_comp) if self.rule_type == RuleBase.MAX else max(rls_prop, q_comp)
        self._set_release(release)
        return release, True if release != rls_prop else False

    def _set_release(self, release):
        self.release[self.t.step] = release

    def bind_to(self, callback):
        self._observers.append(callback)


class RuleRlsSpecified(RuleBase):
    def __init__(self, name, time, release_specified=None, rule_type=RuleBase.MIN):
        # Call super class constructor
        super().__init__(name, time, rule_type=rule_type)
        if release_specified is not None: self.release = release_specified

    def _get_rule_q(self):
        return self.release[self.t.step].item()

    def set_release(self, time_step, rls_specified):
        self.release[time_step] = rls_specified

    def calc_release(self, rls_prop):
        return self.release[self.t.step].item(), False


class RuleStack(RuleBase):
    def __init__(self, name, time, operating_junction, rules=[], *,
                 rule_type=RuleBase.MIN, release_type=RuleBase.CTRL_RLS):
        # Call super class constructor
        super().__init__(name, time, operating_junction, rule_type=rule_type, release_type=release_type)
        self.rules = rules
        self.rls_min = np.full(self.t.n_steps, np.nan)
        self.rls_max = np.full(self.t.n_steps, np.nan)
        # Initialize the controlling rule
        self.ctrl_rule = np.full(self.t.n_steps, np.nan)

    def append_rule(self, rule):
        if type(rule) == list:
            self.rules = self.rules + rule
        else:
            self.rules.append(rule)

    def insert_rule(self, rule, idx=0):
        if type(rule) != list:
            rule = [rule]
        for i, _ in enumerate(rule):
            self.rules.insert(idx + i, rule[i])

    def _get_rule_q(self):
        rls_stack = 0. if self.rule_type == RuleBase.MIN else np.inf
        rls_min = 0.
        rls_max = np.inf
        is_ctrl = False
        cur_ctrl_rule = 0
        rls_unctrl = self.operating_jnc.rls_unctrl[self.t.step] if self.operating_jnc.outlet_unctrl else 0.
        if self.rules:
            for i, cur_rule in enumerate(self.rules):
                rls_stack, is_ctrl = cur_rule.calc_release(rls_stack)
                if is_ctrl:
                    cur_ctrl_rule = i
                    self.ctrl_rule[self.t.step] = i
                if cur_rule.rule_type == RuleBase.MIN:
                    if cur_rule.q_rule_cur > rls_min: rls_min = cur_rule.q_rule_cur
                elif cur_rule.rule_type == RuleBase.MAX:
                    if cur_rule.q_rule_cur < rls_max: rls_max = cur_rule.q_rule_cur
                rls_adj = self.operating_jnc.calc_water_balance(
                    self.operating_jnc.qin_net, rls_unctrl + rls_stack)
                rls_stack = rls_adj - rls_unctrl
                if self.operating_jnc.stor[self.t.step] == self.operating_jnc.stor_inactive:
                    rls_max = rls_stack
            self.rls_min[self.t.step] = rls_min
            self.rls_max[self.t.step] = rls_max
            # Notify rules if they have control
            for j, cur_rule in enumerate(self.rules):
                condition = True if j == cur_ctrl_rule else False
                cur_rule.set_ctrl_rule(condition)
        return rls_stack


class RuleUserDefined(RuleBase):
    def __init__(self, name, time, operating_junction, user_fnc, *,
                 rule_type=RuleBase.MIN, release_type=RuleBase.CTRL_RLS):
        # Call super class constructor
        super().__init__(name, time, operating_junction, rule_type=rule_type, release_type=release_type)
        self.user_fnc = user_fnc

    def _get_rule_q(self, args=None):
        rls = self.user_fnc(args)
        return rls
