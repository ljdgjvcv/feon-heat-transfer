# -*- coding: utf-8 -*-

# -------------------------------------
import numpy as np


class PostProcess(object):
    """Post-processing class for heat conduction with convection"""

    def __init__(self, els, nds, dim):
        self.dim = dim
        self.nodes = nds
        self.els = els

    # ------------------------------------------------------------------
    # Temperature methods
    # ------------------------------------------------------------------

    def get_nodes_temp(self, key="T"):
        """Get temperatures at all nodes"""
        temps = []
        for nd in self.nodes:
            if hasattr(nd, "temp") and key in nd.temp:
                temps.append(nd.temp[key])
            else:
                temps.append(0.0)
        return np.array(temps)

    def get_max_temp(self):
        """Get maximum temperature and node ID"""
        arr = self.get_nodes_temp("T")
        if len(arr) > 0:
            idx = np.argmax(arr)
            return arr[idx], idx
        return None, None

    def get_min_temp(self):
        """Get minimum temperature and node ID"""
        arr = self.get_nodes_temp("T")
        if len(arr) > 0:
            idx = np.argmin(arr)
            return arr[idx], idx
        return None, None

    def get_avg_temp(self):
        """Get average temperature"""
        arr = self.get_nodes_temp("T")
        return np.mean(arr) if len(arr) > 0 else None

    # ------------------------------------------------------------------
    # Heat flux methods
    # ------------------------------------------------------------------

    def get_elements_heat_flow(self, key):
        """Get heat flux component for all elements"""
        info = []
        for el in self.els:
            if hasattr(el, "heat_flux") and key in el.heat_flux:
                val = el.heat_flux[key]
                info.append(val[0] if isinstance(val, (list, np.ndarray)) else val)
            elif hasattr(el, "heat_flow") and key in el.heat_flow:
                val = el.heat_flow[key]
                info.append(val[0] if isinstance(val, (list, np.ndarray)) else val)
            else:
                info.append(0.0)
        return np.array(info)

    def get_max_qx(self):
        """Get maximum |qx| and element ID"""
        qx = self.get_elements_heat_flow("qx")
        if len(qx) > 0 and not np.all(qx == 0):
            idx = np.argmax(np.abs(qx))
            return qx[idx], idx
        return 0.0, -1

    def get_max_qy(self):
        """Get maximum |qy| and element ID"""
        qy = self.get_elements_heat_flow("qy")
        if len(qy) > 0 and not np.all(qy == 0):
            idx = np.argmax(np.abs(qy))
            return qy[idx], idx
        return 0.0, -1

    def get_max_qz(self):
        """Get maximum |qz| and element ID (3D only)"""
        if self.dim < 3:
            return 0.0, -1
        qz = self.get_elements_heat_flow("qz")
        if len(qz) > 0 and not np.all(qz == 0):
            idx = np.argmax(np.abs(qz))
            return qz[idx], idx
        return 0.0, -1

    def get_max_heat_flux_magnitude(self):
        """Get maximum heat flux magnitude and element ID"""
        if self.dim == 1:
            q = self.get_elements_heat_flow("qx")
            mag = np.abs(q)
        elif self.dim == 2:
            qx = self.get_elements_heat_flow("qx")
            qy = self.get_elements_heat_flow("qy")
            if len(qx) == 0 or len(qy) == 0:
                return 0.0, -1
            mag = np.sqrt(qx**2 + qy**2)
        else:
            qx = self.get_elements_heat_flow("qx")
            qy = self.get_elements_heat_flow("qy")
            qz = self.get_elements_heat_flow("qz")
            if len(qx) == 0 or len(qy) == 0 or len(qz) == 0:
                return 0.0, -1
            mag = np.sqrt(qx**2 + qy**2 + qz**2)

        if len(mag) > 0 and not np.all(mag == 0):
            idx = np.argmax(mag)
            return mag[idx], idx
        return 0.0, -1

    # ------------------------------------------------------------------
    # Convection heat loss methods
    # ------------------------------------------------------------------

    def get_elements_conv_heat_loss(self):
        """Get convection heat loss for all elements (W)"""
        losses = []
        for el in self.els:
            if hasattr(el, 'conv_heat_loss'):
                losses.append(el.conv_heat_loss)
            elif hasattr(el, '_conv_heat_loss'):
                losses.append(el._conv_heat_loss)
        return np.array(losses) if losses else np.array([])

    def get_total_conv_heat_loss(self):
        """Get total convection heat loss (W)"""
        losses = self.get_elements_conv_heat_loss()
        return np.sum(losses) if len(losses) > 0 else 0.0

    def get_max_conv_heat_loss(self):
        """Get maximum element convection heat loss and ID"""
        losses = self.get_elements_conv_heat_loss()
        if len(losses) > 0:
            idx = np.argmax(losses)
            return losses[idx], idx
        return 0.0, -1

    # ------------------------------------------------------------------
    # Energy balance check
    # ------------------------------------------------------------------

    def check_energy_balance(self):
        """
        Check energy balance: input = output + convection loss
        Returns dictionary with balance results
        """
        total_input = 0.0
        total_output = 0.0
        for nd in self.nodes:
            if hasattr(nd, '_heat_flow'):
                for val in nd._heat_flow.values():
                    if val > 0:
                        total_input += val
                    elif val < 0:
                        total_output += abs(val)

        conv_loss = self.get_total_conv_heat_loss()
        balance = total_input - total_output - conv_loss
        rel_error = abs(balance) / (total_input + 1e-10) * 100

        return {
            'total_input': total_input,
            'total_output': total_output,
            'conv_loss': conv_loss,
            'balance': balance,
            'relative_error': rel_error
        }

    # ------------------------------------------------------------------
    # Statistics and output methods
    # ------------------------------------------------------------------

    def get_stats(self):
        """Get all statistics"""
        max_temp, max_temp_id = self.get_max_temp()
        min_temp, min_temp_id = self.get_min_temp()
        avg_temp = self.get_avg_temp()
        max_qx, max_qx_id = self.get_max_qx()
        max_qy, max_qy_id = self.get_max_qy()
        max_qz, max_qz_id = self.get_max_qz()
        max_flux, max_flux_id = self.get_max_heat_flux_magnitude()
        total_conv_loss = self.get_total_conv_heat_loss()
        max_conv_loss, max_conv_id = self.get_max_conv_heat_loss()

        return {
            'dim': self.dim,
            'n_nodes': len(self.nodes),
            'n_elements': len(self.els),
            'max_temp': max_temp,
            'max_temp_id': max_temp_id,
            'min_temp': min_temp,
            'min_temp_id': min_temp_id,
            'avg_temp': avg_temp,
            'max_qx': max_qx,
            'max_qx_id': max_qx_id,
            'max_qy': max_qy,
            'max_qy_id': max_qy_id,
            'max_qz': max_qz,
            'max_qz_id': max_qz_id,
            'max_heat_flux': max_flux,
            'max_heat_flux_id': max_flux_id,
            'total_conv_loss': total_conv_loss,
            'max_conv_loss': max_conv_loss,
            'max_conv_loss_id': max_conv_id,
        }

    def results(self, verbose=True, check_balance=True):
        """Print formatted results with convection info"""
        s = self.get_stats()

        output = f"""
{'='*60}
                    HEAT CONDUCTION RESULTS
{'='*60}
  Dimension:           {s['dim']}D
  Nodes:               {s['n_nodes']}
  Elements:            {s['n_elements']}

{'─'*60}
  TEMPERATURE:
{'─'*60}
    Maximum:  {s['max_temp']:.4f} K (Node {s['max_temp_id']})
    Minimum:  {s['min_temp']:.4f} K (Node {s['min_temp_id']})
    Average:  {s['avg_temp']:.4f} K

{'─'*60}
  HEAT FLUX:
{'─'*60}
    Max |qx|: {s['max_qx']:.4e} W/m² (Elem {s['max_qx_id']})
    Max |qy|: {s['max_qy']:.4e} W/m² (Elem {s['max_qy_id']})"""

        if self.dim >= 3:
            output += f"""
    Max |qz|: {s['max_qz']:.4e} W/m² (Elem {s['max_qz_id']})"""

        output += f"""
    Max |q|:  {s['max_heat_flux']:.4e} W/m² (Elem {s['max_heat_flux_id']})

{'─'*60}
  CONVECTION:
{'─'*60}
    Total Loss:  {s['total_conv_loss']:.4f} W
    Max Loss:    {s['max_conv_loss']:.4f} W (Elem {s['max_conv_loss_id']})"""

        if check_balance:
            b = self.check_energy_balance()
            output += f"""
{'─'*60}
  ENERGY BALANCE:
{'─'*60}
    Input:   {b['total_input']:.4f} W
    Output:  {b['total_output']:.4f} W
    Conv:    {b['conv_loss']:.4f} W
    Error:   {b['balance']:.4e} W ({b['relative_error']:.4f} %)"""

        output += f"""
{'='*60}
"""
        if verbose:
            print(output)
        return s

    def export_to_vtk(self, filename="temperature_field.vtk"):
        """Export results to VTK format for ParaView visualization"""
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Heat Conduction Results\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")

            f.write(f"POINTS {len(self.nodes)} float\n")
            temp_values = []
            for nd in self.nodes:
                if self.dim == 2:
                    f.write(f"{nd.x} {nd.y} 0.0\n")
                else:
                    f.write(f"{nd.x} {nd.y} {nd.z}\n")
                temp_values.append(nd.temp.get("T", 0.0) if hasattr(nd, "temp") else 0.0)

            f.write(f"\nPOINT_DATA {len(self.nodes)}\n")
            f.write("SCALARS Temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for t in temp_values:
                f.write(f"{t}\n")

        print(f"Results exported to {filename}")


def print_simple_results(nodes, elements, dim):
    """Quick result summary"""
    PostProcess(elements, nodes, dim).results()


if __name__ == "__main__":
    print("=" * 50)
    print("Post-processing module test (with convection)")
    print("=" * 50)

    class MockNode:
        def __init__(self, x, y, z, temp_val):
            self.x, self.y, self.z = x, y, z
            self.temp = {"T": temp_val}
            self._heat_flow = {"Q": 0.0}

    class MockElement:
        def __init__(self, qx, qy, qz=0, conv_loss=0.0):
            self.heat_flux = {"qx": qx, "qy": qy, "qz": qz}
            self.conv_heat_loss = conv_loss

    nodes = [
        MockNode(0, 0, 0, 100.0),
        MockNode(1, 0, 0, 80.0),
        MockNode(0, 1, 0, 60.0),
        MockNode(0, 0, 1, 40.0),
    ]
    nodes[0]._heat_flow["Q"] = 1000.0
    nodes[3]._heat_flow["Q"] = -200.0

    elements = [
        MockElement(1000, 500, conv_loss=50.0),
        MockElement(800, 400, conv_loss=30.0),
        MockElement(600, 300, conv_loss=20.0),
    ]

    print("\n--- 2D Test ---")
    PostProcess(elements[:2], nodes[:3], dim=2).results()

    print("\n--- 3D Test ---")
    PostProcess(elements, nodes, dim=3).results()

    print("\n✅ Post-processing test complete!")