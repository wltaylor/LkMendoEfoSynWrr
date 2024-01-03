# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:48:33 2020

@author: cd
"""

import numpy as np
from efo.rule_compliance import RuleMinQ, RuleMinAnnSched, HydrologicCond

# TODO: Create a water supply condition class
class D1610HydroCondUrr(HydrologicCond):
    def __init__(self, name, time, jncLm, wsc, storLkPils):
        # Call super class constructor
        super().__init__(name)
        # Class variable
        # TODO: Use error handling to make sure everything is legal
        self.jncLm = jncLm
        self.T = time
        self.wsc = wsc
        self.storLkPils = storLkPils
        # self.curHC = HydrologicCond
        # self.minQLkup = RuleCompAnnSched('name', time, minQsched + bufferSched, 'D')
        self.HC = np.zeros(time.n_steps, dtype='int')

    def get_hydrologic_cond(self, ts_offset=0):
        hc = self.HC[min(self.T.end, self.T.step + ts_offset)]
        # If hydrologic cond has already been set then no need to check it again
        if hc == 0:
            curDT = self.T.get_datetime_offset(ts_offset)
            hc = self.wsc[min(self.T.end, self.T.step + ts_offset)]
            prevHc = self.HC[min(self.T.end, max(0, self.T.step + ts_offset - 1))]
            if hc == 1 and curDT.month == 6 and curDT.day == 1:
                # We can't offset Lm stor because it hasn't been calculated yet so use previous
                curLmStor = self.jncLm.stor[max(0, self.T.step - 1)]
                curLpStor = self.storLkPils[min(self.T.end, self.T.step + ts_offset)]
                if curLmStor + curLpStor < 130000:
                    hc = 3
            elif prevHc > 1 and prevHc < 4 and \
                curDT.month >= 6 and curDT.month <= 12 and not(curDT.month == 6 and curDT.day == 1):
                hc = prevHc
            elif prevHc == 1 and curDT.month >= 10 \
                and self.jncLm.stor[max(0, self.T.step - 1)] < 30000:
                hc = 2
            # if tsOffset == 0: self.HC[self.T.step] = hc
            self.HC[min(self.T.end, self.T.step + ts_offset)] = hc
        self.cur_cond = hc - 1
        return hc
    
    
# class RuleD1610MinQ(RuleMinAnnSched):
#     def __init__(self, name, time, monDayHr, minQsched, bufferSched, hydroCond):
#         # Call super class constructor
#         super().__init__(name, time, monDayHr, minQsched + bufferSched, hydroCond=hydroCond)
#         self.qMin = np.full(self.T.nSteps, np.nan)

#     def get_qcomp(self, *, rlsPrev=None, rlsUnCtrl=None, stor=None, qIn=None, tsOffset=0):
#         hc = self.hydroCond.get_hydrologic_cond(tsOffset=tsOffset)
#         # qMin = super().get_qcomp(tsOffset=tsOffset, col=hc-1)
#         qMin = super().get_qcomp(tsOffset=tsOffset)
#         self.qMin[min(self.T.end, self.T.step + tsOffset)] = qMin
#         return qMin


# class RuleD1610MinQ(RuleMinAnnSched):
#     def __init__(self, name, time, jncLm, monDayHr, minQsched, bufferSched, wsc, storLkPils):
#         # Call super class constructor
#         super().__init__(name, time, monDayHr, minQsched + bufferSched, hydroCond=wsc)
#         # Class variable
#         # TODO: Use error handling to make sure everything is legal
#         self.jncLm = jncLm
#         self.wsc = wsc
#         self.storLkPils = storLkPils
#         # self.curHC = HydrologicCond
#         # self.minQLkup = RuleCompAnnSched('name', time, minQsched + bufferSched, 'D')
#         self.HC = np.empty(time.nSteps, dtype='int')
#         self.qMin = np.empty(time.nSteps)

#     def get_qcomp(self, tsOffset):
#         curDT = self.T.get_datetime_offset(tsOffset)
#         hc = self.wsc[min(self.T.nSteps-1, self.T.step + tsOffset)]
#         prevHc = self.HC[min(self.T.nSteps-1, max(0, self.T.step + tsOffset - 1))]
#         if hc == 1 and curDT.month == 6 and curDT.day == 1:
#             # We can't offset Lm stor because it hasn't been calculated
#             curLmStor = self.jncLm.stor[max(0, self.T.step - 1)]
#             curLpStor = self.storLkPils[min(self.T.nSteps-1, self.T.step + tsOffset)]
#             if curLmStor + curLpStor < 130000:
#                 hc = 3
#         elif prevHc > 1 and prevHc < 4 and \
#             curDT.month >= 6 and curDT.month <= 12 and not(curDT.month == 6 and curDT.day == 1):
#             hc = prevHc
#         elif prevHc == 1 and curDT.month >= 10 \
#             and self.jncLm.stor[max(0, self.T.step - 1)] < 30000:
#                     hc = 2
#         self.HC[min(self.T.nSteps-1, self.T.step + tsOffset)] = hc
#         # self.curHC.curCond = hc
#         qMin = super().get_qcomp(tsOffset, hc-1)
#         self.qMin[min(self.T.nSteps-1, self.T.step + tsOffset)] = qMin
#         return qMin


