"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
import sys
from os import path
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

from femvf.dynamicalmodels import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
from femvf.meshutils import process_meshlabel_to_dofs
import nonlineq as nleq

import blocktensor.subops as gops
import blocktensor.linalg as bla
from blocktensor import vec as bvec

from hopf import make_hopf_system

# slepc4py.init(sys.argv)

# pylint: disable=redefined-outer-name
TEST_HOPF = True
TEST_FP = True
TEST_MODAL = True
TEST_HOPF_BIFURCATION = True 
# Very weird bug where the second eigenvalue problem has error code 73 even 
# though it uses the same matrices as the first case. Very uncertain what the cause of this
# bug is

def test_hopf(x0, dx, hopf_res, hopf_jac):
    """Test correctness of the Hopf jacobian/residual"""
    ## Test the Hopf system Jacobian
    x1 = x0 + dx

    g0 = hopf_res(x0)
    g1 = hopf_res(x1)
    dgdx = hopf_jac(x0)

    dg_exact = g1 - g0
    dg_linear = bla.mult_mat_vec(dgdx, dx)
    print(f"||g0|| = {g0.norm():e}")
    print(f"||g1|| = {g1.norm():e}")

    print(f"||dg_exact|| = {dg_exact.norm():e}")
    print(f"||dg_linear|| = {dg_linear.norm():e}")

    print(f"||dg_exact-dg_linear|| = {(dg_exact-dg_linear).norm():e}")


EBODY = 5e3 * 10
ECOV = 5e3 * 10
PSUB = 800 * 10   

def set_properties(props, region_to_dofs, res):

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

if __name__ == '__main__':
    ## Load 3 residual functions needed to model the Hopf system
    mesh_name = 'BC-dcov5.00e-02-cl2.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    res = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.KelvinVoigt,
        FluidType = fldm.Bernoulli1DDynamicalSystem,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    dres_u = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.LinearStateKelvinVoigt,
        FluidType = fldm.LinearStateBernoulli1DDynamicalSystem,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    dres_ut = load_dynamical_fsi_model(
        mesh_path, None, SolidType = sldm.LinearStatetKelvinVoigt,
        FluidType = fldm.LinearStatetBernoulli1DDynamicalSystem,
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
    
    for model in (res, dres_u, dres_ut):
        model.ymid = y_mid

    ## Initialize the Hopf system
    xhopf, hopf_res, hopf_jac, apply_dirichlet_vec, idx_dirichlet, labels = make_hopf_system(res, dres_u, dres_ut, props)
    state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = labels

    # Set the starting point of any iterative solutions
    xhopf_0 = xhopf.copy()
    xhopf_0['psub'].array[:] = PSUB
    # xhopf_0['psub'].array[:] = 1e-10
    # This value is set to ensure the correct df/dxt matrix when computing eigvals
    xhopf_0['omega'].array[:] = 1.0

    ## Test the Hopf jacobian
    if TEST_HOPF:
        dxhopf = xhopf.copy()
        for subvec in dxhopf:
            subvec.set(0)
        dxhopf['u'].array[:] = 1.0e-7

        apply_dirichlet_vec(dxhopf)
        apply_dirichlet_vec(xhopf_0)
        test_hopf(xhopf_0, dxhopf, hopf_res, hopf_jac)

    ## Test solve for fixed-points
    def linear_subproblem_fp(x_n):
        """Linear subproblem of a Newton solver"""
        xhopf_n = xhopf_0.copy()
        xhopf_n[state_labels] = x_n

        res_n = hopf_res(xhopf_n)[state_labels]
        jac_n = hopf_jac(xhopf_n)[state_labels, state_labels]

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
        x_n = xhopf_0.copy()
        x_n = x_n[state_labels]
            
        newton_params = {
            'maximum_iterations': 20
        }
        x_n, info = nleq.newton_solve(x_n, linear_subproblem_fp, norm=bvec.norm, params=newton_params)
        print(x_n.norm())
        print(info)

    ## Test solving for stabilty (modal analysis of the jacobian)
    if TEST_MODAL:
        # Here we solve the eigenvalue problem
        # omega df/dxt ex = df/dx ex
        # in the transformed form
        # df/dxt ex = lambda df/dx ex
        # where lambda=1/omega, and ex is a generalized eigenvector
        xhopf_n = xhopf_0.copy()
        xhopf_n[state_labels] = x_n
        jac = hopf_jac(xhopf_n)
        df_dx = jac[state_labels, state_labels]
        df_dxt = jac[mode_imag_labels, mode_imag_labels]

        # Set dirichlet conditions for the mass matrix
        df_dxt[0, 0].zeroRows(idx_dirichlet, diag=1e-10)
        df_dxt[0, 1].zeroRows(idx_dirichlet, diag=0)
        df_dxt[1, 0].zeroRows(idx_dirichlet, diag=0)
        df_dxt[1, 1].zeroRows(idx_dirichlet, diag=1e-10)

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

    ## Test solving the Hopf system for the Hopf bifurcation
    _IDX = [0, 1, 2, 3, 4]
    _IDX = slice(None, None, None)
    def linear_subproblem_hopf(x_n):
        """Linear subproblem of a Newton solver"""
        # xhopf_n = xhopf_0.copy()
        # xhopf_n[state_labels] = x_n
        
        xhopf_n = xhopf_0.copy()
        xhopf_n[_IDX] = x_n
        
        res_n = hopf_res(xhopf_n)[_IDX]
        jac_n = hopf_jac(xhopf_n)[_IDX, _IDX]

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

    if TEST_HOPF_BIFURCATION:
        x_n = xhopf_0[_IDX].copy()
            
        newton_params = {
            'maximum_iterations': 20
        }
        x_n, info = nleq.newton_solve(x_n, linear_subproblem_hopf, norm=bvec.norm, params=newton_params)
        print(x_n.norm())
        print(info)
    

    
