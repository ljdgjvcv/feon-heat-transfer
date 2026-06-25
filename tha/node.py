# -*- coding: utf-8 -*-
# ------------------------------------

from ..base import NodeBase
import numpy as np


class Node(NodeBase):

    def __init__(self, *coord): 
        NodeBase.__init__(self, *coord)
        
        if len(coord) == 1:
            coord_len = len(coord[0])
        else:
            coord_len = len(coord)
        
        if coord_len == 2:
            self.dim = 2
            self._nAk = ("T",)
            self._nBk = ("Q",)
            self._nCk = ("H_conv", "T_ambient", "area")
        else:
            self.dim = 3
            self._nAk = ("T",)
            self._nBk = ("Q",)
            self._nCk = ("H_conv", "T_ambient", "area")
        
        self._dof = len(self._nAk)
        self._temp = dict.fromkeys(self._nAk, 0.0)
        self._heat_flow = dict.fromkeys(self._nBk, 0.0)
        self._convection = dict.fromkeys(self._nCk, 0.0)

    @property
    def temp(self):
        return self._temp
    
    @property
    def heat_flow(self):
        return self._heat_flow

    @property
    def convection(self):
        return self._convection
    
    @property
    def nAk(self):
        return self._nAk
    
    @property
    def nBk(self):
        return self._nBk

    @property
    def nCk(self):
        return self._nCk
   
    def init_keys(self):
        """Kept for compatibility"""
        pass
            
    def init_unknowns(self, *unknowns):
        """Initialize temperature DOFs"""
        for key in unknowns:
            if key in self._nAk:
                self._temp[key] = None
            else:
                raise AttributeError(f"Unknown temperature name: {key}. Available: {self._nAk}")
    
    def set_heat_flow(self, **heat_flows):
        """Add heat flow (accumulative)"""
        for key in heat_flows.keys():
            if key in self._nBk:
                self._heat_flow[key] += heat_flows[key]
            else:
                raise AttributeError(f"Unknown heat flow name: {key}. Available: {self._nBk}")
    
    def set_heat_flow_value(self, **heat_flows):
        """Set heat flow (non-accumulative)"""
        for key in heat_flows.keys():
            if key in self._nBk:
                self._heat_flow[key] = heat_flows[key]
            else:
                raise AttributeError(f"Unknown heat flow name: {key}. Available: {self._nBk}")
    
    def clear_heat_flow(self):
        """Clear all heat flows"""
        for key in self._nBk:
            self._heat_flow[key] = 0.0
            
    def get_heat_flow(self):
        """Get heat flow dictionary"""
        return self._heat_flow
     
    def set_temp(self, **temp):
        """Set temperature values"""
        for key in temp.keys():
            if key in self._nAk:
                self._temp[key] = temp[key]
            else:
                raise AttributeError(f"Unknown temperature name: {key}. Available: {self._nAk}")
    
    def add_temp(self, **temp):
        """Add temperature values (for superposition)"""
        for key in temp.keys():
            if key in self._nAk:
                if self._temp[key] is None:
                    self._temp[key] = temp[key]
                else:
                    self._temp[key] += temp[key]
            else:
                raise AttributeError(f"Unknown temperature name: {key}. Available: {self._nAk}")
            
    def clear_temp(self):
        """Clear all temperatures"""
        for key in self._nAk:
            self._temp[key] = 0.0
        
    def get_temp(self):
        """Get temperature dictionary"""
        return self._temp

    def set_convection(self, **conv):
        """Add convection parameters (accumulative)"""
        for key in conv.keys():
            if key in self._nCk:
                self._convection[key] += conv[key]
            else:
                raise AttributeError(f"Unknown convection parameter: {key}. Available: {self._nCk}")

    def set_convection_value(self, **conv):
        """Set convection parameters (non-accumulative)"""
        for key in conv.keys():
            if key in self._nCk:
                self._convection[key] = conv[key]
            else:
                raise AttributeError(f"Unknown convection parameter: {key}. Available: {self._nCk}")

    def clear_convection(self):
        """Clear all convection parameters"""
        for key in self._nCk:
            self._convection[key] = 0.0

    def get_convection(self):
        """Get convection parameter dictionary"""
        return self._convection