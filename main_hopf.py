"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
import numpy as np
import h5py
import matplotlib.pyplot as plt

from femvf.meshutils import process_celllabel_to_dofs_from_forms
from blockarray import h5utils

from setup import setup_models, set_props
import libhopf
import libsignal

# pylint: disable=redefined-outer-name
# pylint: disable=no-member

TEST_FP = True
TEST_MODAL = True
TEST_HOPF_BIFURCATION = True

EBODY = 5e3 * 10
ECOV = 5e3 * 10
PSUB = 450 * 10

if __name__ == '__main__':
    ## Load the models
    # mesh_name = 'BC-dcov5.00e-02-coarse'
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    # mesh_name = 'vf-square'
    mesh_path = path.join('./mesh', mesh_name+'.xml')
    res, dres = setup_models(mesh_path)

    ## Set model properties
    region_to_dofs = process_celllabel_to_dofs_from_forms(
        res.solid.forms, res.solid.forms['fspace.scalar'])

    props = res.props.copy()
    props = set_props(props, region_to_dofs, res)
    # res.set_props(props)
    # dres.set_props(props)

    ## Initialize the Hopf system
    # This vector normalizes the real/imag components of the unstable eigenvector
    EREF = res.state.copy()
    EREF['q'].set(1.0)
    EREF.set(1.0)
    hopf = libhopf.HopfModel(res, dres, ee=EREF)
    hopf.set_props(props)
    # (
    #     xhopf, hopf_res, hopf_jac,
    #     apply_dirichlet_vec, apply_dirichlet_mat,
    #     labels, info) = libhopf.make_hopf_system(res, dres, props, EREF)
    (state_labels,
        mode_real_labels,
        mode_imag_labels,
        psub_labels,
        omega_labels) = hopf.labels_hopf_components

    IDX_DIRICHLET = hopf.IDX_DIRICHLET

    ## Test solving for fixed-points
    print("\n-- Test solution of fixed-points --")

    _control = res.control.copy()
    _control['psub'] = PSUB
    res.set_control(_control)
    res.set_props(props)

    newton_params = {
        'maximum_iterations': 20
    }
    xfp_0 = res.state.copy()
    xfp_n, info = libhopf.solve_fixed_point(res, xfp_0, newton_params=newton_params)

    ## Test solving for stabilty (modal analysis of the jacobian)
    print("\n-- Test modal analysis of system linearized dynamics --")

    omegas, eigvecs_real, eigvecs_imag = libhopf.solve_linear_stability(res, xfp_n)
    print(omegas)

    idx_hopf = 3
    omega_hopf = abs(omegas[idx_hopf].imag)
    mode_real_hopf = eigvecs_real[idx_hopf]
    mode_imag_hopf = eigvecs_imag[idx_hopf]

    ## Test solving the Hopf system for the Hopf bifurcation
    # set the initial guess based on the stability analysis and fixed-point solution
    print("\n-- Test solution of Hopf system for Hopf bifurcation point --")

    xhopf_0 = hopf.state.copy()
    xhopf_0[state_labels] = xfp_n
    xhopf_0[psub_labels[0]].array[:] = PSUB
    xhopf_0[omega_labels[0]].array[:] = omega_hopf

    xmode_real, xmode_imag = libhopf.normalize_eigenvector_by_hopf_condition(
        mode_real_hopf, mode_imag_hopf, EREF)
    xhopf_0[mode_real_labels] = xmode_real
    xhopf_0[mode_imag_labels] = xmode_imag

    newton_params = {
        'maximum_iterations': 20
    }
    xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
    print(xhopf_n.norm())
    print(info)

    with h5py.File("out/hopf_state.h5", mode='w') as f:
        h5utils.create_resizable_block_vector_group(
            f, xhopf_n.labels, xhopf_n.bshape)
        h5utils.append_block_vector_to_group(f, xhopf_n)

    ## Plot the obtained mode shape's glottal width waveform
    xfp = xhopf_n[state_labels]
    xmode_real = xhopf_n[mode_real_labels]
    xmode_imag = xhopf_n[mode_imag_labels]
    psub = xhopf_n['psub'][0]
    omega = xhopf_n['omega'][0]

    proc_glottal_width = libsignal.make_glottal_width(hopf, 100)
    unit_xmode_real, unit_xmode_imag = libhopf.normalize_eigenvector_amplitude(xmode_real, xmode_imag)

    fig, ax = plt.subplots(1, 1)

    print(xfp.norm())
    print(unit_xmode_real.norm())
    print(unit_xmode_imag.norm())
    for ampl in np.linspace(0, 100000.0, 5):
        gw = proc_glottal_width(xhopf_n.to_mono_ndarray(), np.array([ampl, 0.0]))
        ax.plot(gw, label=f"Amplitude {ampl:.2e}")
    ax.set_xlabel(f"Time [period]")
    ax.set_ylabel("Glottal width [cm]")
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig/glottal_width_vs_amplitude_main_hopf.png", dpi=250)
    # breakpoint()
