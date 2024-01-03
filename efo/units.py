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
# Date of Creation: 2020/05/06                              #
#############################################################


import abc
import copy as cp

class UnitsBase(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name
        super().__init__()    
    
    @classmethod
    def __subclasshook__(cls,C):
        if cls is UnitsBase:
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

class UnitsStandard:
    def __init__(self, time):
        # Assumes and hourly time step
        self.flow2vol = 1.98347109902 * time.n_hrs / 24
        self.vol2flow = 0.50416666040 / time.n_hrs * 24
        
        

    