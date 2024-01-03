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
import numpy as np
from scipy.interpolate import interp1d
import copy as cp

class HypsoBase(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name
        super().__init__()  
    @abc.abstractmethod
    def stor2elev(self, stor):
        pass
    @abc.abstractmethod
    def stor2area(self, stor):
        pass
    @classmethod
    def __subclasshook__(cls,C):
        if cls is HypsoBase:
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
        
        
class Hypso(HypsoBase):
    def __init__(self, name, elev=None, stor=None, area=None):
        # Call super class constructor
        super().__init__(name)
        self.elev = elev
        self.stor = stor
        self.area = area
        if np.any(elev): self.interpFncElev = interp1d(self.stor, self.elev, kind='linear', fill_value='extrapolate')
        if np.any(area): self.interpFncArea = interp1d(self.stor, self.area, kind='linear', fill_value='extrapolate')
        if np.any(stor): self.interpFncStor = interp1d(self.elev, self.stor, kind='linear', fill_value='extrapolate')

    def stor2elev(self, stor):
        return np.atleast_1d(self.interpFncElev(stor)) if self.interpFncElev else np.nan
    
    def stor2area(self, stor):
        return np.atleast_1d(self.interpFncArea(stor)) if self.interpFncArea else np.nan
    
    def elev2stor(self, elev):
        return np.atleast_1d(self.interpFncStor(elev)) if self.interpFncStor else np.nan