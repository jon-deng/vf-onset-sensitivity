"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
from petsc4py import PETSc
import numpy as np
import h5py
import matplotlib.pyplot as plt

from femvf.models.dynamical import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
from femvf.meshutils import process_celllabel_to_dofs_from_forms

import nonlineq as nleq

import blocktensor.subops as gops
from blocktensor import vec as bvec
from blocktensor import h5utils

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

def setup_models(mesh_path):
    """
    Return residual + linear residual needed to model the Hopf system
    """
    res = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.KelvinVoigt,
        FluidType = fldm.Bernoulli1DDynamicalSystem,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    dres = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.LinearizedKelvinVoigt,
        FluidType = fldm.LinearizedBernoulli1DDynamicalSystem,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    return res, dres

def set_properties(props, region_to_dofs, res):
    """
    Set the model properties
    """
    # VF material props
    gops.set_vec(props['emod'], ECOV)
    gops.set_vec(props['emod'], EBODY)
    gops.set_vec(props['eta'], 5.0)
    gops.set_vec(props['rho'], 1.0)
    gops.set_vec(props['nu'], 0.45)

    # Fluid separation smoothing props
    gops.set_vec(props['zeta_min'], 1.0e-4)
    gops.set_vec(props['zeta_sep'], 1.0e-4)

    # Contact and midline symmetry properties
    # y_gap = 0.5 / 10 # Set y gap to 0.5 mm
    # y_gap = 1.0
    y_gap = 0.01
    y_contact_offset = 1/10*y_gap
    y_max = res.solid.forms['mesh.mesh'].coordinates()[:, 1].max()
    y_mid = y_max + y_gap
    y_contact = y_mid - y_contact_offset
    gops.set_vec(props['ycontact'], y_contact)
    gops.set_vec(props['kcontact'], 1e16)
    gops.set_vec(props['ymid'], y_mid)

    return props


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

    props = res.properties.copy()
    props = set_properties(props, region_to_dofs, res)
    # res.set_properties(props)
    # dres.set_properties(props)

    ## Initialize the Hopf system
    # This vector normalizes the real/imag components of the unstable eigenvector
    EREF = res.state.copy()
    EREF['q'].set(1.0)
    EREF.set(1.0)
    hopf = libhopf.HopfModel(res, dres, ee=EREF)
    hopf.set_properties(props)
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
    res.set_properties(props)

    newton_params = {
        'maximum_iterations': 20
    }
    xfp_0 = res.state.copy()
    xfp_n, info = libhopf.solve_fixed_point(res, xfp_0, newton_params=newton_params)

    ## Test solving for stabilty (modal analysis of the jacobian)
    print("\n-- Test modal analysis of system linearized dynamics --")
    _XHOPF = hopf.state.copy()
    _XHOPF[state_labels] = xfp_n
    _XHOPF['psub'].array[:] = PSUB
    _XHOPF['omega'].array[:] = 1.0 # have to set omega=1 to get correct jacobian

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

    proc_glottal_width = libsignal.make_glottal_width(res, dres, 100)
    unit_xmode_real, unit_xmode_imag = libhopf.normalize_eigenvector_amplitude(xmode_real, xmode_imag)

    fig, ax = plt.subplots(1, 1)

    print(xfp.norm())
    print(unit_xmode_real.norm())
    print(unit_xmode_imag.norm())
    for ampl in np.linspace(0, 10000.0, 5):
        gw = proc_glottal_width(
            xfp.to_ndarray(),
            unit_xmode_real.to_ndarray(),
            unit_xmode_imag.to_ndarray(),
            psub, omega, ampl, 0.0)
        ax.plot(gw, label=f"Amplitude {ampl:.2e}")
    ax.set_xlabel(f"Time [period]")
    ax.set_ylabel("Glottal width [cm]")
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig/glottal_width_vs_amplitude_main_hopf.png", dpi=250)
    # breakpoint()
