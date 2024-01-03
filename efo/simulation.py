# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:28:51 2022

@author: cd
"""

#############################################################
# Author: Chris Delaney <cjdelaney@ucsd.edu>                #
#                                                           #
#         Center for Western Weather and Water Extremes     #
#         Scripps Institution of Oceanography               #
#         UC San Diego                                      #
#                                                           #
#         https://github.com/hydrophile                     #
#         https://cw3e.ucsd.edu/                            #
#                                                           #
# Date of Creation: 2022/05/08                              #
#############################################################

import abc
# import numpy as np
# from efo.time import TimeBase, TimeCont
import time
import copy as cp

class SimulationBase(metaclass=abc.ABCMeta):
    def __init__(self, name, network, print_tag, metadata=None, print_time_step=True):
        self.name = name
        self.network = network
        self.t = network.t
        self.sim_time = 0.
        self.print_tag = f'{name}, {print_tag}' if print_tag else name
        self.metadata = metadata
        self.print_time_step = print_time_step

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo))
        return result
        
    def run_simulation(self):
        start = start = time.time()
        for ts in self.t.steps:
            self.t.step = ts
            if self.print_time_step:
                out_str = f'{self.print_tag}, {ts}, ' + self.t.cur_dt.strftime('%Y/%m/%d %H')
                print(out_str)
            self.network.process_junctions()
        end = end = time.time()
        self.sim_time = end - start
        return self
            
            
class Simulation(SimulationBase):
    def __init__(self, name, network, print_tag=None, metadata=None, print_time_step=True):
        # Call super class constructor
        super().__init__(
            name, network, print_tag=print_tag,
            metadata=metadata, print_time_step=print_time_step)
    
    