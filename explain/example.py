# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from feon.tha.node import Node
from feon.tha.element import Tetra3D11H
from feon.tha.system import System
from feon.tha.solver import solve_static_thermal

from meshpy.tet import MeshInfo, build
from meshpy.geometry import generate_surface_of_revolution, GeometryBuilder


def generate_cylinder_mesh_meshpy(r=1.0, h=2.0, radial_subdiv=20, max_volume=0.01):
    """Generate cylinder mesh using meshpy"""
    rz = [(0, 0), (r, 0), (r, h), (0, h)]
    geob = GeometryBuilder()
    geob.add_geometry(*generate_surface_of_revolution(
        rz, radial_subdiv=radial_subdiv, ring_markers=[1, 2, 3]
    ))
    mesh_info = MeshInfo()
    geob.set(mesh_info)
    mesh = build(mesh_info, max_volume=max_volume)
    return np.array(mesh.points), np.array(mesh.elements), np.array(mesh.faces)


def triangle_area_3d(face_points):
    """Compute area of a 3D triangle"""
    p0 = face_points[0]
    p1 = face_points[1]
    p2 = face_points[2]
    return np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2.0


if __name__ == "__main__":
    
    # Parameters
    r = 1.0
    h = 2.0
    k = 45.0
    Q_total = 1000.0
    T_bottom = 293.15
    H_conv = 10.0
    T_ambient = 298.15
    radial_subdiv = 20
    max_volume = 0.01

    print(f"\n[Parameters] radius={r}m, height={h}m, k={k}W/m·K")
    print(f"  Top heat flux={Q_total}W, Bottom temperature={T_bottom}K")
    print(f"  Convection={H_conv}W/m²·K, Ambient={T_ambient}K")

    # ========== 1. Generate mesh ==========
    print("\n[1] Generating mesh...")
    points, cells, face_cells = generate_cylinder_mesh_meshpy(r, h, radial_subdiv, max_volume)
    print(f"    Nodes: {len(points)}, Elements: {len(cells)}, Faces: {len(face_cells)}")

    # ========== 2. Create nodes ==========
    print("\n[2] Creating nodes...")
    nodes = []
    for i, pt in enumerate(points):
        nd = Node(pt[0], pt[1], pt[2])
        nd.ID = i
        nd.init_keys()
        nd.init_unknowns("T")
        nodes.append(nd)
    print(f"    Created {len(nodes)} nodes")

    # ========== 3. Create elements ==========
    print("\n[3] Creating tetrahedral elements...")
    elements = []
    for c in cells:
        el = Tetra3D11H([nodes[idx] for idx in c], k)
        elements.append(el)
    print(f"    Created {len(elements)} elements")

    # ========== 4. Create system ==========
    print("\n[4] Creating FE system...")
    system = System()
    system.add_nodes(nodes)
    system.add_elements(elements)

    # ========== 5. Add side convection ==========
    print("\n[5] Adding side convection...")

    side_faces = []
    for fc in face_cells:
        z_vals = points[fc][:, 2]
        if (z_vals.min() > 1e-6) and (z_vals.max() < h - 1e-6):
            side_faces.append(fc)
    print(f"    Side triangular faces: {len(side_faces)}")

    for fc in side_faces:
        face_nodes = [nodes[idx] for idx in fc]
        system.add_convection_face(face_nodes, H_conv, T_ambient)

    print(f"    Side convection added (h={H_conv} W/m²·K)")

    # ========== 6. Add top heat flux ==========
    print("\n[6] Adding top heat flux...")

    top_faces = [fc for fc in face_cells if abs(points[fc][:, 2].mean() - h) < 1e-6]
    theo_top_area = np.pi * r ** 2

    top_areas = []
    for fc in top_faces:
        area = triangle_area_3d(points[fc])
        top_areas.append(area)

    node_heat = defaultdict(float)
    for fc, area in zip(top_faces, top_areas):
        q_per_node = Q_total * area / theo_top_area / 3.0
        for idx in fc:
            node_heat[idx] += q_per_node

    for nid, q in node_heat.items():
        nodes[nid].set_heat_flow(Q=q)

    total_applied = sum(node_heat.values())
    print(f"    Total applied heat flux: {total_applied:.2f} W")

    # ========== 7. Apply bottom temperature ==========
    print("\n[7] Applying fixed bottom temperature...")
    bottom_nodes = [i for i, nd in enumerate(nodes) if abs(nd.z) < 1e-6]
    for nid in bottom_nodes:
        system.add_node_temp(nid, T=T_bottom)
    print(f"    Bottom fixed temperature nodes: {len(bottom_nodes)}")

    # ========== 8. Assemble and solve ==========
    system.calc_KG()  
    print("    Assembly complete")

    print("\n[8] Solving steady-state heat conduction...")
    solve_static_thermal(system)

    # ========== 9. Results ==========
    print("\n[9] Results...")

    def get_temp(node):
        return node.temp.get("T", 0.0)

    temps = [get_temp(nd) for nd in nodes]

    print(f"\n{'=' * 60}")
    print("TEMPERATURE STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Max temperature: {max(temps):.2f} K ({max(temps) - 273.15:.2f} °C)")
    print(f"  Min temperature: {min(temps):.2f} K ({min(temps) - 273.15:.2f} °C)")
    print(f"  Avg temperature: {np.mean(temps):.2f} K ({np.mean(temps) - 273.15:.2f} °C)")

    # ========== 10. Visualization ==========
    print("\n[10] Generating visualization...")
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221, projection='3d')
    sc = ax1.scatter([nd.x for nd in nodes], [nd.y for nd in nodes], [nd.z for nd in nodes],
                     c=temps, cmap='hot', s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax1, label='Temperature (K)')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Temperature Distribution")

    axis_nodes = sorted([nd for nd in nodes if np.hypot(nd.x, nd.y) < 0.05], key=lambda nd: nd.z)
    side_nodes = [nd for nd in nodes if abs(np.hypot(nd.x, nd.y) - r) < 0.05]
    side_nodes.sort(key=lambda nd: nd.z)

    ax2 = fig.add_subplot(222)
    ax2.plot([nd.z for nd in axis_nodes], [get_temp(nd) for nd in axis_nodes], 'b-', lw=2, label='Centerline')
    ax2.plot([nd.z for nd in side_nodes], [get_temp(nd) for nd in side_nodes], 'r-', lw=2, label='Side surface')
    ax2.set_xlabel("Z (m)")
    ax2.set_ylabel("Temperature (K)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_title("Temperature vs Height")

    ax3 = fig.add_subplot(223)
    ax3.hist(temps, bins=20, color='coral', edgecolor='black')
    ax3.set_xlabel("Temperature (K)")
    ax3.set_ylabel("Number of Nodes")
    ax3.set_title("Temperature Histogram")

    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete!")