"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
import h5py

from femvf.models.dynamical import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
from femvf.meshutils import process_celllabel_to_dofs_from_forms
import blocktensor.subops as gops
from blocktensor import h5utils

import libhopf, libfunctionals as libfuncs
from test_hopf import _test_taylor


# pylint: disable=redefined-outer-name
# pylint: disable=no-member

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

def setup_hopf_state(mesh_path):
    ## Load the models
    res, dres = setup_models(mesh_path)

    ## Set model properties
    region_to_dofs = process_celllabel_to_dofs_from_forms(
        res.solid.forms, res.solid.forms['fspace.scalar'])

    props = res.properties.copy()
    props = set_properties(props, region_to_dofs, res)

    ## Initialize the Hopf system
    # This vector normalizes the real/imag components of the unstable eigenvector
    EREF = res.state.copy()
    EREF['q'].set(1.0)
    EREF.set(1.0)
    hopf = libhopf.HopfModel(res, dres, ee=EREF)
    hopf.set_properties(props)

    (state_labels,
        mode_real_labels,
        mode_imag_labels,
        psub_labels,
        omega_labels) = hopf.labels_hopf_components

    ## Solve for the fixed point
    # this is used to get the initial guess for the Hopf system
    _control = res.control.copy()
    _control['psub'] = PSUB
    res.set_control(_control)
    res.set_properties(props)

    newton_params = {
        'maximum_iterations': 20
    }
    xfp_0 = res.state.copy()
    xfp_n, _ = libhopf.solve_fixed_point(res, xfp_0, newton_params=newton_params)

    ## Solve for linear stabilty at the fixed point
    # this is used to get the initial guess for the Hopf system
    omegas, eigvecs_real, eigvecs_imag = libhopf.solve_linear_stability(res, xfp_n)

    # The unstable mode is apriori known to be the 3rd one for the current test case
    # In the future, you should make this more general/automatic
    idx_hopf = 3
    omega_hopf = abs(omegas[idx_hopf].imag)
    mode_real_hopf = eigvecs_real[idx_hopf]
    mode_imag_hopf = eigvecs_imag[idx_hopf]

    ## Solve the Hopf system for the Hopf bifurcation
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

    with h5py.File("out/hopf_state.h5", mode='w') as f:
        h5utils.create_resizable_block_vector_group(
            f, xhopf_n.labels, xhopf_n.bshape)
        h5utils.append_block_vector_to_group(f, xhopf_n)
    return hopf, xhopf_n, props

def test_reduced_gradient(func, hopf, props0, dprops, xhopf0):
    """
    Test the reduced gradient

    Parameters
    ----------
    func :
        The functional to compute the gradient of
    hopf :
        The Hopf model
    props0, dprops :
        The initial properties and increment in properties
    xhopf0 :
        An initial guess for solving Hopf bifurcations. Solving Hopf bifurcations is implictly
        needed in computing the reduced gradient through finite differences
    """

    def res(props):
        x , info = libhopf.solve_hopf_newton(hopf, xhopf0)
        func.set_state(x)
        func.set_properties(props)
        return func.assem_g()

    def jac(props):
        x, info = libhopf.solve_hopf_newton(hopf, xhopf0)

        for xx in (func, hopf):
            xx.set_state(x)
            xx.set_properties(props)

        return libhopf.solve_reduced_gradient(func, hopf)

    _test_taylor(props0, dprops, res, jac)

if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    hopf, xhopf, props0 = setup_hopf_state(mesh_path)
    func = libfuncs.OnsetPressureFunctional(hopf)

    dprops = props0.copy()
    dprops['emod'].set(1.0)

    test_reduced_gradient(func, hopf, props0, dprops, xhopf)
