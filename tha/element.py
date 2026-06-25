# -*- coding: utf-8 -*-
# ------------------------------------


from __future__ import division
import numpy as np
from ..base import ElementBase
from ..tools import gl_quad2d


# =====================================================================
# 1D Line Element Base Class (with side convection)
# =====================================================================

class StructElement(ElementBase):
    """1D line element base class with side convection support"""

    def __init__(self, nodes):
        ElementBase.__init__(self, nodes)
        self.init_nodes(nodes)
        self.init_unknowns()
        self._heat_flow = dict.fromkeys(self.eIk, 0.)
        self.dens = 2.
        self.thickness = 1.

        self.H_conv = 0.0
        self.T_ambient = 0.0
        self.P = 0.0
        self._Q_conv = None

    def init_nodes(self, nodes):
        v = np.array(nodes[0].coord) - np.array(nodes[1].coord)
        self._volume = np.linalg.norm(v)

    def init_keys(self):
        if self.dim == 2:
            self.set_eIk(("qx", "qy"))
        if self.dim == 3:
            self.set_eIk(("qx", "qy", "qz"))

    def init_unknowns(self):
        pass

    @property
    def transformation(self):
        return self._transformation

    @property
    def me(self):
        return self._me

    @property
    def ke(self):
        return self._ke

    @property
    def heat_flow(self):
        return self._heat_flow

    def calc_ke(self):
        pass

    def calc_me(self):
        pass

    def calc_transformation(self):
        pass

    def calc_Ke(self):
        pass

    def calc_Me(self):
        self.calc_transformation()
        self.calc_me()
        self._Me = np.dot(np.dot(self.transformation.T, self.me), self.transformation)

    def evaluate(self):
        T = np.array([[nd.temp[key] for nd in self.nodes for key in nd.nAk[:self.ndof]]])
        self._undealed_heat_flow = np.dot(self.transformation, np.dot(self.Ke, T.T))
        self.distribute_heat_flow()

    def distribute_heat_flow(self):
        n = len(self.eIk)
        for i, val in enumerate(self.eIk):
            self._heat_flow[val] += self._undealed_heat_flow[i::n]

    def heat_source_equivalent(self, ltype, val):
        raise NotImplementedError

    def set_convection(self, H_conv, T_ambient, P):
        """Set side convection parameters for line elements"""
        self.H_conv = H_conv
        self.T_ambient = T_ambient
        self.P = P

    def _add_conv_contribution(self):
        """Add side convection contribution to stiffness matrix and equivalent heat flow"""
        if self.H_conv <= 0 or self.P <= 0:
            return
        L = self._volume
        coeff = self.H_conv * self.P * L / 6.0
        K_conv = coeff * np.array([[2, 1], [1, 2]])
        self._Q_conv = (self.H_conv * self.P * L * self.T_ambient / 2.0) * np.ones((2, 1))
        if hasattr(self, '_ke') and self._ke is not None:
            self._ke += K_conv
        else:
            self._ke = K_conv

    def get_conv_Q(self):
        return self._Q_conv if self._Q_conv is not None else np.zeros((2, 1))


# =====================================================================
# Continuum Element Base Class (with face convection)
# =====================================================================

class SoildElement(ElementBase):
    """2D/3D continuum element base class with face convection support"""

    def __init__(self, nodes):
        ElementBase.__init__(self, nodes)
        self.init_unknowns()
        self._heat_flux = dict.fromkeys(self.eIk, 0.)
        self.init_nodes(nodes)
        self.dens = 2.
        self.thickness = 1.

        self._conv_faces = []
        self._Q_conv = None

    def init_nodes(self, nodes):
        pass

    def init_unknowns(self):
        pass

    @property
    def B(self):
        return self._B

    @property
    def D(self):
        return self._D

    @property
    def ke(self):
        return self._ke

    @property
    def heat_flux(self):
        return self._heat_flux

    def calc_B(self):
        pass

    def calc_D(self):
        pass

    def calc_Ke(self):
        self.calc_B()
        self.calc_D()
        self._Ke = self.thickness * self.volume * np.dot(np.dot(self.B.T, self.D), self.B)
        self._assemble_convection()

    def evaluate(self):
        T = np.array([[nd.temp[key] for nd in self.nodes for key in nd.nAk[:self.ndof]]])
        self._undealed_heat_flux = np.dot(np.dot(self.D, self.B), T.T)
        self.distribute_heat_flux()

    def distribute_heat_flux(self):
        n = len(self.eIk)
        for i, val in enumerate(self.eIk):
            self._heat_flux[val] += self._undealed_heat_flux[i::n]

    def heat_source_equivalent(self, ltype, val):
        raise NotImplementedError

    def add_convection_on_face(self, face_nodes, H_conv, T_ambient, area=None):
        """
        Add convection boundary condition on a face/edge

        Parameters
        ----------
        face_nodes : list of 2 (2D) or 3 (3D) nodes on the boundary
        H_conv : convection heat transfer coefficient (W/m²·K)
        T_ambient : ambient temperature (K)
        area : optional precomputed face area
        """
        if area is None:
            if len(face_nodes) == 2:
                v = np.array(face_nodes[0].coord) - np.array(face_nodes[1].coord)
                area = np.linalg.norm(v) * self.thickness
            elif len(face_nodes) == 3:
                p0 = np.array(face_nodes[0].coord)
                p1 = np.array(face_nodes[1].coord)
                p2 = np.array(face_nodes[2].coord)
                v1 = p1 - p0
                v2 = p2 - p0
                area = np.linalg.norm(np.cross(v1, v2)) / 2.0
            else:
                raise ValueError("Face must have 2 (2D) or 3 (3D) nodes")

        self._conv_faces.append({
            'nodes': face_nodes,
            'H_conv': H_conv,
            'T_ambient': T_ambient,
            'area': area
        })

    def _assemble_convection(self):
        """Assemble all face convection contributions to element matrix"""
        if not self._conv_faces:
            return

        if self.ndof != 1:
            raise RuntimeError("Convection requires ndof=1 (temperature DOF)")

        n_nodes = len(self.nodes)
        if self._Q_conv is None:
            self._Q_conv = np.zeros((n_nodes, 1))

        for face in self._conv_faces:
            face_nodes = face['nodes']
            H_conv = face['H_conv']
            T_amb = face['T_ambient']
            area = face['area']

            local_ids = [self.nodes.index(nd) for nd in face_nodes]
            m = len(local_ids)

            if m == 2:
                coeff = H_conv * area / 6.0
                K_conv = coeff * np.array([[2, 1], [1, 2]])
                Q_conv = (H_conv * area * T_amb / 2.0) * np.ones((2, 1))
            elif m == 3:
                coeff = H_conv * area / 12.0
                K_conv = coeff * np.array([[2, 1, 1],
                                           [1, 2, 1],
                                           [1, 1, 2]])
                Q_conv = (H_conv * area * T_amb / 3.0) * np.ones((3, 1))
            else:
                continue

            for i, id_i in enumerate(local_ids):
                for j, id_j in enumerate(local_ids):
                    self._Ke[id_i, id_j] += K_conv[i, j]
                self._Q_conv[id_i] += Q_conv[i]

    def get_conv_Q(self):
        if self._Q_conv is None:
            return np.zeros((len(self.nodes), 1))
        return self._Q_conv


# =====================================================================
# 3D Tetrahedral Heat Conduction Element
# =====================================================================

class Tetra3D11H(SoildElement):
    """3D tetrahedral heat conduction element with face convection support"""

    def __init__(self, nodes, K, dens=2.0):
        SoildElement.__init__(self, nodes)
        self.K = K
        self.dens = dens
        self.thickness = 1.0

    def init_nodes(self, nodes):
        V = np.ones((4, 4))
        for i, nd in enumerate(nodes):
            V[i, 1:] = nd.coord
        self._volume = abs(np.linalg.det(V) / 6.)

    def init_keys(self):
        self.set_eIk(("qx", "qy", "qz"))

    def init_unknowns(self):
        for nd in self.nodes:
            nd.init_unknowns("T")
        self._ndof = 1

    def calc_B(self):
        self._B = _calc_B_for_tetra3d11(self.nodes, self.volume)

    def calc_D(self):
        self._D = _calc_D_for_tetra3d11(self.K)

    def calc_Ke(self):
        self.calc_B()
        self.calc_D()
        self._Ke = self.volume * np.dot(np.dot(self.B.T, self.D), self.B)
        self._assemble_convection()


def _calc_B_for_tetra3d11(nodes, volume):
    """Compute B matrix (3x4) for 3D tetrahedral element"""
    if volume < 1e-12:
        raise ValueError(f"Invalid tetrahedron volume: {volume}")

    A = np.ones((4, 4))
    for i, nd in enumerate(nodes):
        A[i, 1:] = nd.coord

    belta = np.zeros(4)
    gama = np.zeros(4)
    delta = np.zeros(4)

    for i in range(4):
        belta[i] = (-1)**(i+1) * np.linalg.det(np.delete(np.delete(A, i, 0), 1, 1))
        gama[i] = (-1)**(i+2) * np.linalg.det(np.delete(np.delete(A, i, 0), 2, 1))
        delta[i] = (-1)**(i+1) * np.linalg.det(np.delete(np.delete(A, i, 0), 3, 1))

    factor = 1.0 / (6.0 * volume)
    return factor * np.array([
        [belta[0], belta[1], belta[2], belta[3]],
        [gama[0], gama[1], gama[2], gama[3]],
        [delta[0], delta[1], delta[2], delta[3]]
    ])


def _calc_D_for_tetra3d11(K=1.0):
    """Compute D matrix (3x3) for 3D isotropic/anisotropic conduction"""
    if np.isscalar(K):
        return K * np.eye(3)
    return np.array(K)


# =====================================================================
# 2D Triangular Heat Conduction Element
# =====================================================================

class Tri2D11H(SoildElement):
    """
    2D triangular heat conduction element with edge convection support
    Each node has 1 DOF (temperature)
    """

    def __init__(self, nodes, K, thickness=1.0, dens=2.0):
        SoildElement.__init__(self, nodes)
        self.K = K
        self.dens = dens
        self.thickness = thickness

    def init_nodes(self, nodes):
        p1 = np.array(nodes[0].coord[:2])
        p2 = np.array(nodes[1].coord[:2])
        p3 = np.array(nodes[2].coord[:2])
        v1 = p2 - p1
        v2 = p3 - p1
        self._volume = np.abs(np.cross(v1, v2)) / 2.0

    def init_keys(self):
        self.set_eIk(("qx", "qy"))

    def init_unknowns(self):
        for nd in self.nodes:
            nd.init_unknowns("T")
        self._ndof = 1

    def calc_B(self):
        self._B = _calc_B_for_tri2d11(self.nodes, self.volume)

    def calc_D(self):
        self._D = _calc_D_for_tri2d11(self.K)

    def calc_Ke(self):
        self.calc_B()
        self.calc_D()
        self._Ke = self.thickness * self.volume * np.dot(np.dot(self.B.T, self.D), self.B)
        self._assemble_convection()

    def add_convection_on_edge(self, edge_nodes, H_conv, T_ambient):
        """
        Add convection on a specific edge of the triangle

        Parameters
        ----------
        edge_nodes : list of 2 nodes on the boundary edge
        H_conv : convection heat transfer coefficient (W/m²·K)
        T_ambient : ambient temperature (K)
        """
        p1 = np.array(edge_nodes[0].coord[:2])
        p2 = np.array(edge_nodes[1].coord[:2])
        length = np.linalg.norm(p2 - p1)
        area = length * self.thickness
        self.add_convection_on_face(edge_nodes, H_conv, T_ambient, area)

    def calc_heat_flux(self, temperatures):
        """
        Compute heat flux (qx, qy) for the element

        Parameters
        ----------
        temperatures : array of 3 nodal temperatures

        Returns
        -------
        (qx, qy) : heat flux components (W/m²)
        """
        self.calc_B()
        self.calc_D()
        q = -np.dot(np.dot(self.D, self.B), temperatures)
        return q[0], q[1]


def _calc_B_for_tri2d11(nodes, area):
    """
    Compute B matrix (2x3) for 2D triangular element
    B = [dN1/dx, dN2/dx, dN3/dx]
        [dN1/dy, dN2/dy, dN3/dy]
    """
    if area < 1e-12:
        raise ValueError(f"Invalid triangle area: {area}")

    x1, y1 = nodes[0].x, nodes[0].y
    x2, y2 = nodes[1].x, nodes[1].y
    x3, y3 = nodes[2].x, nodes[2].y

    b1 = y2 - y3
    c1 = x3 - x2
    b2 = y3 - y1
    c2 = x1 - x3
    b3 = y1 - y2
    c3 = x2 - x1

    factor = 1.0 / (2.0 * area)
    return factor * np.array([
        [b1, b2, b3],
        [c1, c2, c3]
    ])


def _calc_D_for_tri2d11(K=1.0):
    """Compute D matrix (2x2) for 2D isotropic/anisotropic conduction"""
    if np.isscalar(K):
        return K * np.eye(2)
    return np.array(K)