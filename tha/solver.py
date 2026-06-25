# -*- coding: utf-8 -*-
# ------------------------------------

import numpy as np


def solve_static_thermal(system):
    """
    Steady-state heat conduction solver with convection support
    Solves: (K + H) * T = Q + Q_conv
    """
    assert system._is_heatflow_added is True or system._is_temp_added is True, \
        "No heat flow or temperature boundary conditions on the system"

    system.calc_deleted_KG_matrix()
    system.check_deleted_KG_matrix()

    KG, HeatFlow = system.KG_keeped, system.HeatFlow_keeped
    system._Temp_keeped = np.linalg.solve(KG, HeatFlow)

    for i, val in enumerate(system.keeped):
        I = val % system.mndof
        J = int(val / system.mndof)
        system.nodes[J].temp[system.nAk[I]] = system.Temp_keeped[i]

    for el in system.get_elements():
        el.evaluate()
        if hasattr(el, 'calc_conv_heat_loss'):
            el.calc_conv_heat_loss()

    system._is_system_solved = True

    print_energy_balance(system)
    print(" Steady heat conduction (with convection) solved successfully!")


def solve_transient_thermal(system, dt, total_time, initial_temp=0.0):
    """
    Transient heat conduction solver (implicit Euler method) with convection
    """
    import scipy.linalg as sl

    assert system._is_heatflow_added is True or system._is_temp_added is True, \
        "No heat flow or temperature boundary conditions on the system"

    if not system._is_inited:
        system.calc_KG()
    system.calc_MG()

    system.calc_deleted_KG_matrix()
    system.calc_deleted_MG_matrix()

    K = system.KG_keeped
    C = system.MG_keeped

    if C.size == 0 or np.all(C == 0):
        raise NotImplementedError("Transient analysis requires heat capacity matrix")

    n_steps = int(total_time / dt)
    T0 = np.ones(len(system.keeped)) * initial_temp
    system._Temp_history = [T0]
    Q = system.HeatFlow_keeped
    A = C / dt + K

    for step in range(n_steps):
        B = np.dot(C / dt, T0) + Q
        T_new = np.linalg.solve(A, B)
        system._Temp_history.append(T_new)
        T0 = T_new

    for i, val in enumerate(system.keeped):
        I = val % system.mndof
        J = int(val / system.mndof)
        system.nodes[J].temp[system.nAk[I]] = system._Temp_history[-1][i]

    for el in system.get_elements():
        el.evaluate()

    system._is_system_solved = True
    print(" Transient heat conduction solved!")


def solve_dynamic_thermal(system):
    """Thermal modal analysis (not implemented)"""
    raise NotImplementedError("Dynamic thermal analysis not implemented yet")


def print_energy_balance(system):
    """Print energy balance based on node heat flow and convection loss"""
    total_input = 0.0
    total_output = 0.0
    for nd in system.get_nodes():
        for val in nd._heat_flow.values():
            if val > 0:
                total_input += val
            elif val < 0:
                total_output += abs(val)

    total_conv_loss = 0.0
    for el in system.get_elements():
        if hasattr(el, 'conv_heat_loss'):
            total_conv_loss += el.conv_heat_loss
        elif hasattr(el, '_conv_heat_loss'):
            total_conv_loss += el._conv_heat_loss

    if hasattr(system, '_conv_elements'):
        for conv_el in system._conv_elements:
            if hasattr(conv_el, 'conv_heat_loss'):
                total_conv_loss += conv_el.conv_heat_loss
            elif hasattr(conv_el, '_conv_heat_loss'):
                total_conv_loss += conv_el._conv_heat_loss

    balance = total_input - total_output - total_conv_loss
    rel_error = abs(balance) / (total_input + 1e-10) * 100

    print("\n" + "=" * 50)
    print("ENERGY BALANCE")
    print("=" * 50)
    print(f"  Total Heat Input (node Q>0)   : {total_input:10.4f} W")
    print(f"  Total Heat Output (node Q<0)  : {total_output:10.4f} W")
    print(f"  Total Convection Loss         : {total_conv_loss:10.4f} W")
    print(f"  Energy Balance Error          : {balance:10.4e} W")
    print(f"  Relative Error                : {rel_error:6.4f} %")
    print("=" * 50)


# Backward compatibility aliases
def solve_static_elastic(system):
    """Alias for solve_static_thermal (backward compatibility)"""
    solve_static_thermal(system)


def solve_transient_elastic(system, dt, total_time, initial_temp=0.0):
    """Alias for solve_transient_thermal (backward compatibility)"""
    solve_transient_thermal(system, dt, total_time, initial_temp)


def solve_dynamic_eigen_model(system):
    """Alias for solve_dynamic_thermal (backward compatibility)"""
    solve_dynamic_thermal(system)