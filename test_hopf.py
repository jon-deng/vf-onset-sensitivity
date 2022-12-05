"""
Test the Hopf system residual derivatives

This module contains tests to check correctness of the `libhopf.Hopf` residual
sensitivity functions.
Correctness is tested by checking the residual sensitivity against finite
differences computed from the residual.
"""

from os import path
import numpy as np

# from femvf.meshutils import process_celllabel_to_dofs_from_forms
import blockarray.linalg as bla

from libsetup import set_default_props, load_hopf_model

# slepc4py.init(sys.argv)

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name

PSUB = 450.0 * 10

def taylor_convergence(x0, dx, res, jac, norm=None):
    """
    Test that the Taylor convergence order is 2
    """
    if norm is None:
        norm = bla.norm

    # Step sizes go from largest to smallest
    alphas = 2**np.arange(4)[::-1]
    res_ns = [res(x0+alpha*dx).copy() for alpha in alphas]
    res_0 = res(x0).copy()

    dres_exacts = [res_n-res_0 for res_n in res_ns]
    dres_linear = jac(x0, dx)

    errs = np.array([
        norm(dres_exact-alpha*dres_linear)
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ])
    magnitudes = np.array([
        1/2*norm(dres_exact+alpha*dres_linear)
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ])
    with np.errstate(invalid='ignore'):
        conv_rates = np.log(errs[:-1]/errs[1:])/np.log(alphas[:-1]/alphas[1:])
        rel_errs = errs/magnitudes

    print("")
    print(f"||dres_linear||, ||dres_exact|| = {norm(dres_linear)}, {norm(dres_exacts[-1])}")
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))
    return alphas, errs, magnitudes, conv_rates

def test_assem_dres_dstate(hopf, state0, dstate):

    def hopf_res(x):
        hopf.set_state(x)
        res = hopf.assem_res()
        hopf.apply_dirichlet_bvec(res)
        return hopf.assem_res()

    def hopf_jac(x):
        hopf.set_state(x)
        dres_dstate = hopf.assem_dres_dstate()
        hopf.apply_dirichlet_bmat(dres_dstate)
        return dres_dstate

    _test_taylor(state0, dstate, hopf_res, hopf_jac)

def test_assem_dres_dprops(hopf, props0, dprops):
    def hopf_res(x):
        hopf.set_props(x)
        return hopf.assem_res()

    def hopf_jac(x):
        hopf.set_props(x)
        dres_dprops = hopf.assem_dres_dprops()
        hopf.zero_rows_dirichlet_bmat(dres_dprops)
        return dres_dprops

    _test_taylor(props0, dprops, hopf_res, hopf_jac)


if __name__ == '__main__':
    ## Load the Hopf model to test
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.msh')
    model, res, dres = load_hopf_model(
        mesh_path, sep_method='smoothmin', sep_vert_label='separation-inf'
    )

    ## Set the Hopf model properties
    # get the scalar DOFs associated with the cover/body layers
    # region_to_dofs = process_celllabel_to_dofs_from_forms(
    #     res.solid.forms,
    #     res.solid.forms['fspace.scalar']
    # )

    props = model.props.copy()
    props = set_default_props(props, res.solid.forms['mesh.mesh'])

    ## Test sensitivities of the Hopf model residual w.r.t each block of
    ## `Hopf.state`

    # Set the linearization point
    (state_labels,
     mode_real_labels,
     mode_imag_labels,
     psub_labels,
     omega_labels) = model.labels_hopf_components

    x0 = model.state.copy()
    x0[mode_real_labels] = 1.0
    x0[mode_imag_labels] = 1.0
    x0[psub_labels] = PSUB
    x0[omega_labels] = 1.0
    model.apply_dirichlet_bvec(x0)
    # model.set_state(x0)

    # Test the residual derivative in multiple directions
    # Each direction corresponds to a block of `Hopf.state`
    dx = x0.copy()
    for label in model.state.labels[0]:
        print(f"\n -- Checking Hopf jacobian along {label} --")
        dx[:] = 0
        dx[label] = 1e-5
        model.apply_dirichlet_bvec(dx)

        test_assem_dres_dstate(model, x0, dx)

    ## Test sensitivities of the Hopf model residual w.r.t each block of
    ## `Hopf.props`
    # TODO: This only checks one block `Hopf.props` right now
    props0 = props.copy()
    dprops = props0.copy()
    dprops[:] = 0.0
    dprops['emod'] = 1.0
    print("\n -- Checking Hopf jacobian along emod --")
    test_assem_dres_dprops(model, props0, dprops)

