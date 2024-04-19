"""
Linear stability analysis (LSA) of a coupled model

This script computes the least stable modes over a sequence of subglottal
pressures for a given parameter set and dynamical model.

Where the real part of the least stable mode's frequency crosses the real axis
, a Hopf bifurcation occurs.
"""

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

from libhopf import setup, hopf

dfn.set_log_level(50)

# Set `BIFPARAM` to toggle between using a flow driven Bernoulli model
# and a pressure driven Bernoulli model
BIFPARAM_KEY = 'psub'
# BIFPARAM = 'qsub'

if __name__ == '__main__':
    CLSCALE = 0.5
    # mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_name = f'M5_CB_GA3_CL{CLSCALE:.2f}'
    mesh_path = f'mesh/{mesh_name}.msh'
    hopf_model, *_ = setup.load_hopf_model(
        mesh_path,
        sep_method='fixed',
        sep_vert_label='separation-inf',
        bifparam_key=BIFPARAM_KEY,
    )

    props0 = hopf_model.prop.copy()
    setup.set_default_props(props0, hopf_model.res.solid.residual.mesh())
    hopf_model.set_prop(props0)
    res = hopf_model.res

    if BIFPARAM_KEY == 'qsub':
        lmbdas = np.arange(0, 100, 10)
    elif BIFPARAM_KEY == 'psub':
        lmbdas = np.arange(0, 500, 25) * 10
    else:
        raise ValueError("")

    def make_control(lmbda):
        control = res.control.copy()
        control[BIFPARAM_KEY] = lmbda
        return control

    xfps_info = [
        hopf.solve_fp(res, make_control(lmbda), props0, bifparam_key=BIFPARAM_KEY)
        for lmbda in lmbdas
    ]
    bad_fps = [xfp_info[1]['status'] != 0 for xfp_info in xfps_info]

    least_stable_modes = [
        hopf.solve_least_stable_mode(res, xfp_info[0], make_control(lmbda), props0)
        for lmbda, xfp_info in zip(lmbdas, xfps_info)
    ]
    least_stable_omegas = np.array([mode_info[0] for mode_info in least_stable_modes])

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(lmbdas, least_stable_omegas.real)
    axs[1].plot(lmbdas, least_stable_omegas.imag)
    # Mark points with bad fixed point solutions with a 'o'
    axs[0].plot(
        lmbdas[bad_fps],
        least_stable_omegas.real[bad_fps],
        marker='o',
        mfc='none',
        ls='none',
    )

    axs[0].set_ylabel("$\omega_{real}$")
    axs[1].set_ylabel("$\omega_{imag}$ $[\mathrm{rad}/\mathrm{s}]$")
    # axs[0].set_ylim(-10, 10)

    if BIFPARAM_KEY == 'qsub':
        axs[1].set_xlabel("$q_{sub}$ $[\mathrm{cm}^3/\mathrm{s}]$")
    elif BIFPARAM_KEY == 'psub':
        axs[1].set_xlabel("$p_{sub}$ $[0.1 \mathrm{Pa}]$")
    else:
        raise ValueError("")

    fig.tight_layout()
    if BIFPARAM_KEY == 'qsub':
        fig.savefig('fig/lsa_vs_flow.png')
    elif BIFPARAM_KEY == 'psub':
        fig.savefig('fig/lsa_vs_pressure.png')
    else:
        raise ValueError("")
