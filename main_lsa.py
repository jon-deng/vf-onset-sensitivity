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

import libsetup
import libhopf

dfn.set_log_level(50)

# Set `FLOW_DRIVEN` to toggle between using a constant flow-rate Bernoulli flow
# and a pressure driven Bernoulli flow
FLOW_DRIVEN = False
FLOW_DRIVEN = True

if __name__ == '__main__':
    CLSCALE = 0.5
    # mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_name = f'M5_CB_GA3_CL{CLSCALE:.2f}'
    mesh_path = f'mesh/{mesh_name}.msh'
    hopf, *_ = libsetup.load_hopf_model(
        mesh_path,
        sep_method='fixed',
        sep_vert_label='separation-inf',
        flow_driven=FLOW_DRIVEN
    )

    if FLOW_DRIVEN:
        bifparam_key = 'qsub'
    else:
        bifparam_key = 'psub'

    props0 = hopf.prop.copy()
    libsetup.set_default_props(props0, hopf.res.solid.residual.mesh())
    hopf.set_prop(props0)
    res = hopf.res

    if FLOW_DRIVEN:
        lmbdas = np.arange(0, 5000, 50)
    else:
        lmbdas = np.arange(0, 1000, 100)*10

    def make_control(lmbda):
        control = res.control.copy()
        control[bifparam_key] = lmbda
        return control

    least_stable_modes = [
        libhopf.solve_least_stable_mode(
            res,
            libhopf.solve_fp(res, make_control(lmbda), props0, bifparam_key=bifparam_key)[0],
            make_control(lmbda),
            props0
        )
        for lmbda in lmbdas
    ]
    least_stable_omegas = np.array([mode_info[0] for mode_info in least_stable_modes])

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(lmbdas, least_stable_omegas.real)
    axs[1].plot(lmbdas, least_stable_omegas.imag)


    axs[0].set_ylabel("$\omega_{real}$")
    axs[1].set_ylabel("$\omega_{imag}$ $[\mathrm{rad}/\mathrm{s}]$")

    if FLOW_DRIVEN:
        axs[1].set_xlabel("$q_{sub}$ $[\mathrm{cm}^3/\mathrm{s}]$")
    else:
        axs[1].set_xlabel("$p_{sub}$ $[10 \mathrm{Pa}]$")

    fig.tight_layout()
    if FLOW_DRIVEN:
        fig.savefig('fig/lsa_vs_flow.png')
    else:
        fig.savefig('fig/lsa_vs_pressure.png')
