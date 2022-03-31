"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
from os import path
import numpy as np
import functools

from femvf.dynamicalmodels import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
from femvf.meshutils import process_meshlabel_to_dofs

import blocktensor.subops as gops
import blocktensor.linalg as bla

from hopf import make_hopf_system

# slepc4py.init(sys.argv)

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name

EBODY = 5e3 * 10
ECOV = 5e3 * 10
PSUB = 800 * 10

def set_properties(props, region_to_dofs, res):
    """
    Sets model properties
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
    if 'ymid' in props:
        gops.set_vec(props['ymid'], y_mid)

    return y_mid

def _test_taylor(x0, dx, res, jac):
    """
    Test that the Taylor convergence order is 2
    """
    alphas = 2**np.arange(4)[::-1] # start with the largest step and move to original
    res_ns = [res(x0+alpha*dx) for alpha in alphas]
    res_0 = res(x0)

    dres_exacts = [res_n-res_0 for res_n in res_ns]
    dres_linear = bla.mult_mat_vec(jac(x0), dx)

    errs = [
        (dres_exact-alpha*dres_linear).norm()
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    magnitudes = [
        1/2*(dres_exact+alpha*dres_linear).norm()
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    with np.errstate(invalid='ignore'):
        conv_rates = [
            np.log(err_0/err_1)/np.log(alpha_0/alpha_1)
            for err_0, err_1, alpha_0, alpha_1
            in zip(errs[:-1], errs[1:], alphas[:-1], alphas[1:])]
        rel_errs = np.array(errs)/np.array(magnitudes)*100

    print("")
    print(f"||dres_linear||, ||dres_exact|| = {dres_linear.norm()}, {dres_exacts[-1].norm()}")
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))

## Load 3 residual functions needed to model the Hopf system
mesh_name = 'BC-dcov5.00e-02-cl2.00'
# mesh_name = 'vf-square'
mesh_path = path.join('./mesh', mesh_name+'.xml')

res = load_dynamical_fsi_model(
    mesh_path, None, SolidType = sldm.KelvinVoigt,
    FluidType = fldm.Bernoulli1DDynamicalSystem,
    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

dres = load_dynamical_fsi_model(
    mesh_path, None, SolidType = sldm.LinearizedKelvinVoigt,
    FluidType = fldm.LinearizedBernoulli1DDynamicalSystem,
    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

## Set model properties
# get the scalar DOFs associated with the cover/body layers
mesh = res.solid.forms['mesh.mesh']
cell_func = res.solid.forms['mesh.cell_function']
func_space = res.solid.forms['fspace.scalar']
cell_label_to_id = res.solid.forms['mesh.cell_label_to_id']
region_to_dofs = process_meshlabel_to_dofs(mesh, cell_func, func_space, cell_label_to_id)

props = res.properties.copy()
y_mid = set_properties(props, region_to_dofs, res)

for model in (res, dres):
    model.ymid = y_mid

## Initialize the Hopf system
xhopf, hopf_res, hopf_jac, apply_dirichlet_vec, apply_dirichlet_mat, labels, info = make_hopf_system(res, dres, props)
state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = labels

HOPF_LABELS = functools.reduce(lambda a, b: a+b, labels)
IDX_DIRICHLET = info['dirichlet_dofs']

def test_hopf():
    """Test correctness of the Hopf jacobian/residual"""
    xhopf_0 = xhopf.copy()
    # xhopf_0.set(1e-5)
    xhopf_0[mode_real_labels].set(1.0)
    xhopf_0[mode_imag_labels].set(1.0)
    xhopf_0['psub'].array[:] = PSUB
    xhopf_0['omega'].array[:] = 1.0
    apply_dirichlet_vec(xhopf_0)

    for label in HOPF_LABELS:
        print(f"\n -- Checking Hopf jacobian along {label} --")
        dxhopf = xhopf.copy()
        dxhopf.set(0)
        dxhopf[label] = 1e-5
        apply_dirichlet_vec(dxhopf)

        _test_taylor(xhopf_0, dxhopf, hopf_res, hopf_jac)

if __name__ == '__main__':
    ## Test the Hopf jacobian
    test_hopf()
