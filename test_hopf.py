"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
from os import path
import numpy as np

from femvf.meshutils import process_celllabel_to_dofs_from_forms
import blockarray.linalg as bla

from libhopf import HopfModel
from libsetup import setup_models, set_props

# slepc4py.init(sys.argv)

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name

PSUB = 450.0 * 10

def _test_taylor(x0, dx, res, jac, action=None, norm=None):
    """
    Test that the Taylor convergence order is 2
    """
    if action is None:
        action = bla.mult_mat_vec
    if norm is None:
        norm = bla.norm

    alphas = 2**np.arange(4)[::-1] # start with the largest step and move to original
    res_ns = [res(x0+alpha*dx) for alpha in alphas]
    res_0 = res(x0)

    dres_exacts = [res_n-res_0 for res_n in res_ns]
    dres_linear = action(jac(x0), dx)

    errs = [
        norm(dres_exact-alpha*dres_linear)
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    magnitudes = [
        1/2*norm(dres_exact+alpha*dres_linear)
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    with np.errstate(invalid='ignore'):
        conv_rates = [
            np.log(err_0/err_1)/np.log(alpha_0/alpha_1)
            for err_0, err_1, alpha_0, alpha_1
            in zip(errs[:-1], errs[1:], alphas[:-1], alphas[1:])]
        rel_errs = np.array(errs)/np.array(magnitudes)*100

    print("")
    print(f"||dres_linear||, ||dres_exact|| = {norm(dres_linear)}, {norm(dres_exacts[-1])}")
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))

def test_assem_dres_dstate(hopf, state0, dstate):

    def hopf_res(x):
        hopf.set_state(x)
        return hopf.assem_res()

    def hopf_jac(x):
        hopf.set_state(x)
        return hopf.assem_dres_dstate()

    _test_taylor(state0, dstate, hopf_res, hopf_jac)

def test_assem_dres_dprops(hopf, props0, dprops):
    def hopf_res(x):
        hopf.set_props(x)
        return hopf.assem_res()

    def hopf_jac(x):
        hopf.set_props(x)
        return hopf.assem_dres_dprops()

    _test_taylor(props0, dprops, hopf_res, hopf_jac)


if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    res, dres = setup_models(mesh_path)
    model = HopfModel(res, dres)

    ## Set model properties
    # get the scalar DOFs associated with the cover/body layers
    region_to_dofs = process_celllabel_to_dofs_from_forms(
        res.solid.forms, res.solid.forms['fspace.scalar'])

    props = model.props.copy()
    props = set_props(props, region_to_dofs, res)

    (state_labels,
     mode_real_labels,
     mode_imag_labels,
     psub_labels,
     omega_labels) = model.labels_hopf_components

    ## Test dF/dstate
    x0 = model.state.copy()
    x0[mode_real_labels].set(1.0)
    x0[mode_imag_labels].set(1.0)
    x0[psub_labels[0]].array[:] = PSUB
    x0[omega_labels[0]].array[:] = 1.0
    model.apply_dirichlet_bvec(x0)

    for label in model.state.labels[0]:
        print(f"\n -- Checking Hopf jacobian along {label} --")
        dx = x0.copy()
        dx.set(0)
        dx[label] = 1e-5
        model.apply_dirichlet_bvec(dx)

        test_assem_dres_dstate(model, x0, dx)

    ## Test dF/dprops
    props0 = props.copy()
    dprops = props0.copy()
    dprops.set(0.0)
    dprops['emod'] = 1.0
    print("\n -- Checking Hopf jacobian along emod --")
    test_assem_dres_dprops(model, props0, dprops)

