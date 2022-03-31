"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

from femvf.dynamicalmodels import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
from femvf.meshutils import process_meshlabel_to_dofs
import nonlineq as nleq

import blocktensor.subops as gops
from blocktensor import vec as bvec

from hopf import make_hopf_system

import sys
import petsc4py
petsc4py.init(sys.argv)
# slepc4py.init(sys.argv)

# pylint: disable=redefined-outer-name

TEST_FP = True
TEST_MODAL = True
TEST_HOPF_BIFURCATION = True

EBODY = 5e3 * 10
ECOV = 5e3 * 10
PSUB = 800 * 10

PSUB = 1 * 10

def set_properties(props, region_to_dofs, res):
    """
    Set the model properties
    """

    # VF material props
    # TODO: Should replace these with gops.set_vec to be more general
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

def setup_models():
    """
    Return residual + linear residual needed to model the Hopf system
    """
    # mesh_name = 'BC-dcov5.00e-02-coarse'
    # mesh_name = 'BC-dcov5.00e-02-cl2.00'
    mesh_name = 'vf-square'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    res = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.KelvinVoigt,
        FluidType = fldm.Bernoulli1DDynamicalSystem,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    dres = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.LinearizedKelvinVoigt,
        FluidType = fldm.LinearizedBernoulli1DDynamicalSystem,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    return res, dres

def region_to_dofs_from_forms(forms, func_space):
    """
    Return a map from cell regions to associated dofs
    """
    mesh = res.solid.forms['mesh.mesh']
    cell_func = res.solid.forms['mesh.cell_function']
    cell_label_to_id = res.solid.forms['mesh.cell_label_to_id']
    return process_meshlabel_to_dofs(
        mesh, cell_func, func_space, cell_label_to_id)


if __name__ == '__main__':
    res, dres = setup_models()

    ## Set model properties
    # get the scalar DOFs associated with the cover/body layers
    region_to_dofs = region_to_dofs_from_forms(
        res.solid.forms, res.solid.forms['fspace.scalar'])

    props = res.properties.copy()
    y_mid = set_properties(props, region_to_dofs, res)

    for model in (res, dres):
        model.ymid = y_mid

    ## Initialize the Hopf system
    (
        xhopf, hopf_res, hopf_jac,
        apply_dirichlet_vec, apply_dirichlet_mat,
        labels, info) = make_hopf_system(res, dres, props)
    (
        state_labels, mode_real_labels, mode_imag_labels,
        psub_labels, omega_labels) = labels

    IDX_DIRICHLET = info['dirichlet_dofs']

    ## Test solving for fixed-points
    xhopf_0 = xhopf.copy()
    xhopf_0['psub'].array[:] = PSUB
    xhopf_0['omega'].array[:] = 1.0

    def linear_subproblem_fp(x_n):
        """Linear subproblem of a Newton solver"""
        xhopf_n = xhopf_0.copy()
        xhopf_n[state_labels] = x_n

        _res_n = hopf_res(xhopf_n)
        _jac_n = hopf_jac(xhopf_n)
        apply_dirichlet_vec(_res_n)
        apply_dirichlet_mat(_jac_n)

        res_n = _res_n[state_labels]
        jac_n = _jac_n[state_labels, state_labels]

        def assem_res():
            """Return residual"""
            return res_n

        def solve(rhs_n):
            """Return jac^-1 res"""
            _rhs_n = rhs_n.to_petsc()
            _jac_n = jac_n.to_petsc()
            _dx_n = _jac_n.getVecRight()

            ksp = PETSc.KSP().create()
            ksp.setType(ksp.Type.PREONLY)

            pc = ksp.getPC()
            pc.setType(pc.Type.LU)

            ksp.setOperators(_jac_n)
            ksp.setUp()
            ksp.solve(_rhs_n, _dx_n)

            dx_n = x_n.copy()
            dx_n.set_vec(_dx_n)
            return dx_n
        return assem_res, solve

    if TEST_FP:
        print("\n-- Test solution of fixed-points --")
        xfp_n = xhopf_0.copy()
        xfp_n = xfp_n[state_labels]

        newton_params = {
            'maximum_iterations': 20
        }
        xfp_n, info = nleq.newton_solve(xfp_n, linear_subproblem_fp, norm=bvec.norm, params=newton_params)
        print(xfp_n.norm())
        print(info)

    ## Test solving for stabilty (modal analysis of the jacobian)
    xhopf_0 = xhopf.copy()
    xhopf_0['psub'].array[:] = PSUB
    xhopf_0['omega'].array[:] = 1.0 # have to set omega=1 to get correct jacobian

    if TEST_MODAL:
        print("\n-- Test modal analysis of system linearized dynamics --")
        # Here we solve the eigenvalue problem
        # omega df/dxt ex = df/dx ex
        # in the transformed form
        # df/dxt ex = lambda df/dx ex
        # where lambda=1/omega, and ex is a generalized eigenvector
        xhopf_n = xhopf_0.copy()
        xhopf_n[state_labels] = xfp_n
        jac = hopf_jac(xhopf_n)
        df_dx = jac[state_labels, state_labels]
        df_dxt = jac[mode_real_labels, mode_imag_labels]

        # Set dirichlet conditions for the mass matrix
        df_dxt[0, 0].zeroRows(IDX_DIRICHLET, diag=1e-10)
        df_dxt[0, 1].zeroRows(IDX_DIRICHLET, diag=0)
        df_dxt[1, 0].zeroRows(IDX_DIRICHLET, diag=0)
        df_dxt[1, 1].zeroRows(IDX_DIRICHLET, diag=1e-10)

        _df_dx = df_dx.to_petsc()
        _df_dxt = df_dxt.to_petsc()

        eps = SLEPc.EPS().create()
        eps.setOperators(_df_dxt, _df_dx)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        # number of eigenvalues to solve for and dimension of subspace to approximate problem
        num_eig = 5
        num_col = 10*num_eig
        eps.setDimensions(num_eig, num_col)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
        eps.solve()

        eigvals = np.array([eps.getEigenvalue(jj) for jj in range(eps.getConverged())])
        omegas = -1/eigvals
        print(f"Omegas:", omegas)

    idx_hopf = 2
    omega_hopf = abs(omegas[idx_hopf].imag)
    mode_real_hopf = _df_dx.getVecRight()
    mode_imag_hopf = _df_dx.getVecRight()
    eps.getEigenvector(idx_hopf, mode_real_hopf, mode_imag_hopf)

    ## Test solving the Hopf system for the Hopf bifurcation
    # set the initial guess based on the stability analysis and fixed-point solution
    xhopf_0 = xhopf.copy()
    xhopf_0[state_labels] = xfp_n
    xhopf_0[mode_real_labels].set_vec(mode_real_hopf)
    xhopf_0[mode_imag_labels].set_vec(mode_imag_hopf)
    # xhopf_0[mode_real_labels] = 1.0
    # xhopf_0[mode_imag_labels] = 1.0
    xhopf_0['psub'].array[:] = PSUB
    xhopf_0['omega'].array[:] = omega_hopf

    _IDX = slice(None, None, None)
    def linear_subproblem_hopf(x_n):
        """Linear subproblem of a Newton solver"""
        # xhopf_n = xhopf_0.copy()
        # xhopf_n[state_labels] = x_n

        xhopf_n = xhopf_0.copy()
        xhopf_n[:] = x_n

        res_n = hopf_res(xhopf_n)
        jac_n = hopf_jac(xhopf_n)
        apply_dirichlet_vec(res_n)
        apply_dirichlet_mat(jac_n)

        norms = np.array(
            [[mat.norm() for mat in row] for row in jac_n])
        dbg_mat = jac_n.to_petsc()[:, :]
        iszero_row = np.all(dbg_mat == 0, axis=1)
        iszero_col = np.all(dbg_mat == 0, axis=0)
        num_zero_rows = np.sum(iszero_row)
        num_zero_cols = np.sum(iszero_col)
        breakpoint()

        def assem_res():
            """Return residual"""
            return res_n

        def solve(rhs_n):
            """Return jac^-1 res"""
            _rhs_n = rhs_n.to_petsc()
            _jac_n = jac_n.to_petsc()
            _dx_n = _jac_n.getVecRight()

            ksp = PETSc.KSP().create()
            ksp.setType(ksp.Type.PREONLY)
            ksp.setOperators(_jac_n)

            pc = ksp.getPC()
            pc.setType(pc.Type.LU)

            pc.setUp()
            ksp.setUp()
            ksp.solve(_rhs_n, _dx_n)

            dx_n = x_n.copy()
            dx_n.set_vec(_dx_n)
            return dx_n
        return assem_res, solve

    if TEST_HOPF_BIFURCATION:
        print("\n-- Test solution of Hopf system for Hopf bifurcation point --")
        xhopf_0['omega'].array[:] = 2.0
        xfp_n = xhopf_0[_IDX].copy()

        newton_params = {
            'maximum_iterations': 20
        }
        xfp_n, info = nleq.newton_solve(xfp_n, linear_subproblem_hopf, norm=bvec.norm, params=newton_params)
        print(xfp_n.norm())
        print(info)

