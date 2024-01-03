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
# Date of Creation: 2020/05/19                              #
#############################################################


import abc
import numpy as np
# from collections import OrderedDict
from efo.time import TimeBase, TimeFcst
from efo.junction import ReservoirJunction
from efo.qin import QinRouted
import copy as cp
# import concurrent.futures


class NetworkBase(metaclass=abc.ABCMeta):
    def __init__(self, name, time, junctions):
        self.name = name
        self.junctions = junctions
        self.n_junctions = len(junctions)
        self.create_net_maps()
        if issubclass(type(time), TimeBase):
            self.t = time
        super().__init__()

    @abc.abstractmethod
    def process_junctions(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is NetworkBase:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented

    def create_net_maps(self):
        self.jnc_map = dict([(curJnc.name, curJnc) for curJnc in self.junctions])
        self.net_idx_map = {curJnc: ii for ii, curJnc in enumerate(self.junctions)}
        return

    def get_junction_index(self, junction):
        idx = self.net_idx_map[junction]
        return idx

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo))
        return result
    
    
class Network(NetworkBase):
    def __init__(self, name, time, junctions):
        # Call super class constructor
        super().__init__(name, time, junctions)

    def set_process_junctions_callback(self, rule):
        rule.bind_to(self.process_junctions)

    def process_junctions(self, idx_end=None):
        if idx_end is None: idx_end = self.n_junctions
        for i, cur_jnc in enumerate(self.junctions[:idx_end]):
            cur_jnc.calc_qout()
    
    def append_junctions(self, junctions):
        if isinstance(junctions, list):
            self.junctions += junctions
        else:
            self.junctions.append(junctions)
        self.create_net_maps()
    
    def append_network(self, network, name=None):
        comb_junctions = self.junctions + network.junctions
        if name is None: name = self.name + '-' + network.name
        return Network(name, self.t, junctions=comb_junctions)


# This provides a link between your network and your forecast network
class NetworkFcst(Network):
    def __init__(self, name, time_fcst, junctions=None, fcst_junctions=None,
                 network=None, fcst_network=None):
        # Call super class constructor
        if network:
            super().__init__(name, time_fcst, network.junctions)
        else:
            super().__init__(name, time_fcst, junctions)
        self.tFcst = time_fcst
        if fcst_network:
            self.fcst_network = fcst_network
        else:
            self.fcst_network = Network(name + 'fcstNetwork', time_fcst, fcst_junctions)
        self.tFcst.bind_to(self.set_init_fcst_cond)
        self.fcst_net_map = {
            jnc: fcst_jnc for jnc, fcst_jnc in zip(self.junctions, self.fcst_network.junctions)}
        return
    
    def set_init_fcst_cond(self, rowFcst, colFcst=0):
        for i, cur_fcst_jnc in enumerate(self.fcst_network.junctions):
            cur_fcst_jnc.qout[0] = self.junctions[i].qout[max(0, self.t.t_cont.step - 1)]
            if issubclass(type(self.fcst_network.junctions[i]), ReservoirJunction):
                cur_fcst_jnc.stor[0] = \
                    self.junctions[i].stor[max(0, self.t.t_cont.step - 1)]
                if cur_fcst_jnc.outlet_ctrl:
                    cur_fcst_jnc.rls_ctrl[0] = \
                        self.junctions[i].rls_ctrl[max(0, self.t.t_cont.step - 1)]
                if cur_fcst_jnc.outlet_unctrl:
                    cur_fcst_jnc.rls_unctrl[0] = \
                        self.junctions[i].rls_unctrl[max(0, self.t.t_cont.step - 1)]
            cur_fcst_rch = cur_fcst_jnc.get_ds_reach()
            if cur_fcst_rch:
                cur_fcst_rch.qout[0] = \
                    self.junctions[i].get_ds_reach().qout[max(0, self.t.t_cont.step - 1)]

    def process_fcst_junctions(self):
        self.fcst_network.process_junctions()


                
            
            
        
        
        