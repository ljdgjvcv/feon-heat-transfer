# -*- coding: utf-8 -*-
# ------------------------------------

import numpy as np
from .solver import *
from ..base import SystemBase
from .post_process import PostProcess


class ConvectionFace3D:
    """3D triangular face convection element"""
    
    def __init__(self, nodes, H_conv, T_ambient):
        self.nodes = nodes
        self.H_conv = H_conv
        self.T_ambient = T_ambient
        self._Ke = None
        self._Q_conv = None
        self._calc_by_gauss_quadrature()
    
    def _calc_by_gauss_quadrature(self):
        """Compute using 3-point Gauss quadrature"""
        coords = np.array([[nd.x, nd.y, nd.z] for nd in self.nodes])
        
        self._Ke = np.zeros((3, 3))
        self._Q_conv = np.zeros((3, 1))
        
        gp = np.array([[1.0/6.0, 1.0/6.0], [2.0/3.0, 1.0/6.0], [1.0/6.0, 2.0/3.0]])
        gw = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        
        for i, (xi, eta) in enumerate(gp):
            N = np.array([1 - xi - eta, xi, eta])
            dN_dxi = np.array([[-1, -1], [1, 0], [0, 1]])
            
            J = np.zeros((3, 2))
            for a in range(3):
                J[:, 0] += coords[a] * dN_dxi[a, 0]
                J[:, 1] += coords[a] * dN_dxi[a, 1]
            
            cross = np.cross(J[:, 0], J[:, 1])
            detJ = np.linalg.norm(cross)
            
            N_mat = N.reshape(-1, 1)
            self._Ke += self.H_conv * np.dot(N_mat, N_mat.T) * detJ * gw[i]
            self._Q_conv += self.H_conv * self.T_ambient * N.reshape(-1, 1) * detJ * gw[i]
    
    def calc_Ke(self):
        """Called by FEON library"""
        pass
    
    def get_conv_Q(self):
        """Return convection load vector"""
        return self._Q_conv


class System(SystemBase):

    def __init__(self):
        SystemBase.__init__(self)
        self._HeatFlow = {}
        self._Temp = {}
        self._is_inited = False
        self._is_heatflow_added = False
        self._is_temp_added = False
        self._is_system_solved = False

        self._conv_elements = []
        self._convection_Q = None

    def __repr__(self):
        return "%dD System: \nNodes: %d\nElements: %d"\
               %(self.dim, self.non, self.noe,)
    
    @property
    def HeatFlow(self):
        return self._HeatFlow

    @property
    def Temp(self):
        return self._Temp

    @property
    def KG(self):
        return self._KG

    @property
    def MG(self):
        return self._MG

    @property
    def KG_keeped(self):
        return self._KG_keeped

    @property
    def MG_keeped(self):
        return self._MG_keeped

    @property
    def HeatFlow_keeped(self):
        return self._HeatFlow_keeped

    @property
    def Temp_keeped(self):
        return self._Temp_keeped

    @property
    def deleted(self):
        return self._deleted

    @property
    def keeped(self):
        return self._keeped

    @property
    def nonzeros(self):
        return self._nonzeros

    def init(self):
        """Initialize system"""
        if self.noe == 0:
            return
        self._mndof = max(el.ndof for el in self.get_elements())
        self._nAk = self.nodes[0].nAk[:self.mndof]
        self._nBk = self.nodes[0].nBk[:self.mndof]
        self._dim = self.nodes[0].dim
        
        self._HeatFlow = {nd.ID: nd.get_heat_flow() for nd in self.get_nodes()}
        self._Temp = {nd.ID: nd.get_temp() for nd in self.get_nodes()}

    def add_convection_element(self, conv_elem):
        """Add a convection boundary element (ConvectionFace3D)"""
        self._conv_elements.append(conv_elem)
        self._is_heatflow_added = True

    def add_convection_face(self, nodes, H_conv, T_ambient, t=1.0):
        """
        Add a 3D triangular face convection boundary
        
        Parameters
        ----------
        nodes : 3 nodes on the boundary
        H_conv : convection heat transfer coefficient (W/m²·K)
        T_ambient : ambient temperature (K)
        t : thickness (only for 2D, ignored for 3D)
        """
        if len(nodes) == 2:
            raise NotImplementedError("2D convection face not implemented in this version")
        elif len(nodes) == 3:
            conv_elem = ConvectionFace3D(nodes, H_conv, T_ambient)
        else:
            raise ValueError("Convection face must have 2 (2D) or 3 (3D) nodes")
        self.add_convection_element(conv_elem)

    def calc_KG(self):
        """Assemble global stiffness matrix (conduction + convection)"""
        self.init()
        n = self.non
        m = self.mndof
        shape = n * m
        self._KG = np.zeros((shape, shape))
        self._convection_Q = np.zeros(shape)

        for el in self.get_elements():
            ID = [nd.ID for nd in el.nodes]
            el.calc_Ke()
            M = el.ndof
            for N1, I in enumerate(ID):
                for N2, J in enumerate(ID):
                    self._KG[m*I:m*I+M, m*J:m*J+M] += el.Ke[M*N1:M*(N1+1), M*N2:M*(N2+1)]
            
            if hasattr(el, 'get_conv_Q'):
                q_conv = el.get_conv_Q()
                if q_conv is not None:
                    for i, nd in enumerate(el.nodes):
                        idx = nd.ID * m
                        self._convection_Q[idx] += q_conv[i, 0]

        for conv_el in self._conv_elements:
            conv_el.calc_Ke()
            nodes = conv_el.nodes
            if hasattr(conv_el, '_Ke') and conv_el._Ke is not None:
                for i, nd_i in enumerate(nodes):
                    for j, nd_j in enumerate(nodes):
                        idx_i = nd_i.ID * m
                        idx_j = nd_j.ID * m
                        self._KG[idx_i, idx_j] += conv_el._Ke[i, j]
            if hasattr(conv_el, 'get_conv_Q'):
                q_conv = conv_el.get_conv_Q()
                if q_conv is not None:
                    for i, nd in enumerate(nodes):
                        idx = nd.ID * m
                        self._convection_Q[idx] += q_conv[i, 0]

        self._is_inited = True

    def calc_MG(self):
        """Assemble global heat capacity matrix (not implemented)"""
        self._MG = np.zeros((0, 0))

    def add_element_heat_source(self, eid, ltype, val):
        """Add element heat source (not implemented)"""
        if not self._is_inited:
            self.calc_KG()
        assert eid < self.noe, "Element does not exist"
        self.elements[eid].heat_source_equivalent(ltype=ltype, val=val)
        self._is_heatflow_added = True

    def add_node_heatflow(self, nid, **heatflows):
        """Add node heat flow (point heat source)"""
        if not self._is_inited:
            self.calc_KG()
        assert nid < self.non, "Node does not exist"
        for key in heatflows.keys():
            assert key in self.nBk, "Check if the node heat flows applied are correct"
        self.nodes[nid].set_heat_flow(**heatflows)   
        self._is_heatflow_added = True

    def add_node_temp(self, nid, **temp):
        """Add fixed temperature boundary condition"""
        if not self._is_inited:
            self.calc_KG()
        assert nid < self.non, "Node does not exist"
        for key in temp.keys():
            assert key in self.nAk, "Check if the node temperature applied are correct"  
        self.nodes[nid].set_temp(**temp)
        if len(temp.values()):
            self._is_temp_added = True

    def add_fixed_temp(self, *nids, value=0.0):
        """Apply fixed temperature to multiple nodes"""
        if not self._is_inited:
            self.calc_KG()
        for nid in nids:
            if isinstance(nid, (list, tuple, np.ndarray)):
                for n in nid:
                    for key in self.nAk:
                        self.nodes[n]._temp[key] = value
            else:
                for key in self.nAk:
                    self.nodes[nid]._temp[key] = value

    def calc_deleted_KG_matrix(self):
        """
        Apply fixed temperature BCs by deleting constrained DOFs
        and adding convection contributions to the load vector
        """
        self._HeatFlow = [nd.get_heat_flow() for nd in self.get_nodes()]
        self._Temp = [nd.get_temp() for nd in self.get_nodes()]
        
        self._HeatFlowValue = [val[key] for val in self.HeatFlow for key in self.nBk]
        self._TempValue = [val[key] for val in self.Temp for key in self.nAk]
        
        self._deleted = [row for row, val in enumerate(self._TempValue) if val is not None]
        self._keeped = [row for row, val in enumerate(self._TempValue) if val is None]
        
        if self._is_temp_added:
            self._apply_boundary_condition()
        
        self._HeatFlow_keeped = np.delete(self._HeatFlowValue, self._deleted, 0)
        self._KG_keeped = np.delete(np.delete(self._KG, self._deleted, 0), self._deleted, 1)
        
        if self._convection_Q is not None:
            conv_Q_keeped = np.delete(self._convection_Q, self._deleted, 0)
            self._HeatFlow_keeped += conv_Q_keeped

    def _apply_boundary_condition(self):
        """Apply fixed temperature BCs to the heat flow vector"""
        self._nonzeros = [(row, val) for row, val in enumerate(self._TempValue) if val]
        if len(self._nonzeros):
            for i, val in self._nonzeros:
                for j in self._keeped:
                    self._HeatFlowValue[j] -= self._KG[i, j] * val

    def calc_deleted_MG_matrix(self):
        """Apply BCs to heat capacity matrix (steady-state only)"""
        if self._MG.size > 0:
            self._MG_keeped = np.delete(np.delete(self._MG, self._deleted, 0), self._deleted, 1)
        else:
            self._MG_keeped = self._MG

    def check_deleted_KG_matrix(self):
        """Check if reduced KG matrix is singular"""
        if self.KG_keeped.size == 0:
            raise ValueError("No degrees of freedom left!")
        zero_rows = 0
        for i in range(self.KG_keeped.shape[0]):
            if np.all(self.KG_keeped[i, :] == 0.):
                zero_rows += 1
        if zero_rows > 0:
            print(f"Warning: {zero_rows} zero rows found in reduced KG matrix")

    def check_deleted_MG_matrix(self):
        pass

    def solve(self, model="static_thermal"):
        """Solve heat conduction problem (static or transient)"""
        if model == "static_thermal":
            from .solver import solve_static_thermal
            solve_static_thermal(self)
        elif model == "transient_thermal":
            from .solver import solve_transient_thermal
            solve_transient_thermal(self)
        else:
            raise ValueError(f"Unknown model: {model}")

    def results(self):
        """Print post-processing results"""
        self.postp = PostProcess(self.get_elements(), self.get_nodes(), self.dim)
        self.postp.results()


if __name__ == "__main__":
    pass