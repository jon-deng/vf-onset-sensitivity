"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
from os import path
import itertools
from functools import reduce
from petsc4py import PETSc
# from slepc4py import SLEPc
import numpy as np

from femvf.dynamicalmodels import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
import nonlineq as nleq

import blocklinalg.genericops as gops
import blocklinalg.linalg as bla
from blocklinalg import vec as bvec
from blocklinalg import mat as bmat

# pylint: disable=redefined-outer-name

def hopf_state(res):
    """
    Return the state vector for a Hopf system
    """
    X_state = res.state.copy()

    _mode_real_vecs = res.state.copy().array
    _mode_real_labels = [label+'_mode_real' for label in X_state.labels[0]]
    X_mode_real = bvec.BlockVec(_mode_real_vecs, _mode_real_labels)

    _mode_imag_vecs = res.state.copy().array
    _mode_imag_labels = [label+'_mode_imag' for label in X_state.labels[0]]
    X_mode_imag = bvec.BlockVec(_mode_imag_vecs, _mode_imag_labels)

    # breakpoint()
    X_psub = res.control[['psub']].copy()

    _omega = X_psub['psub'].copy()
    _omega_vecs = [_omega]
    _omega_labels = [['omega']]
    X_omega = bvec.BlockVec(_omega_vecs, _omega_labels)

    ret = bvec.concatenate_vec([X_state, X_mode_real, X_mode_imag, X_psub, X_omega])
    state_labels = list(X_state.labels[0])
    mode_real_labels = list(X_mode_real.labels[0])
    mode_imag_labels = list(X_mode_imag.labels[0])
    psub_labels = list(X_psub.labels[0])
    omega_labels = list(X_omega.labels[0])
    return ret, state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels

def make_hopf_system(res, dres_u, dres_ut, props, ee=None):
    """
    Return the residual and jacobian of a Hopf bifurcation system

    This system is based on the augmented system proposed by Griewank and Reddien (1983)

    Parameters
    ----------
    res:
        The fixed-point system residual, F
    dres_u:
        The linearized fixed-point system residual w.r.t u. This represents
        dF/du du
    dres_u:
        The linearized fixed-point system residual w.r.t ut. This represents
        dF/dut dut
    """
    for model in (res, dres_u, dres_ut):
        model.set_properties(props)

    def assign_state(x):
        """handles assignment of some parts of x to the underlying models"""
        for model in (res, dres_u, dres_ut):
            model.set_state(x[state_labels])

            _control = model.control.copy()
            _control['psub'].array[0] = x['psub'].array[0]
            model.set_control(_control)

    IDX_DIRICHLET = np.array(list(res.solid.forms['bc.dirichlet'].get_boundary_values().keys()), dtype=np.int32)
    def apply_dirichlet_vec(vec):
        """Zeros dirichlet associated indices"""
        for label in ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']:
            # zero the rows associated with each dirichlet DOF
            subvec = vec[label]
            subvec.array[IDX_DIRICHLET] = 0
            
    # Create the input vector for the system
    x, state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = hopf_state(res)
    labels = (state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels)

    HOPF_SYSTEM_LABELS = tuple(reduce(lambda a, b: a+b, labels))

    EBVEC = x[state_labels].copy()
    EBVEC['u'][0] = 1.0
    def hopf_res(x):
        """Return the Hopf system residual"""
        # Set the model state and subglottal pressure (bifurcation parameter)
        assign_state(x)

        res_state = res.assem_res()

        omega = x['omega'][0]
        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        res_mode_real= dres_u.assem_res() - omega*dres_ut.assem_res()

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_real_labels])
        dres_ut.set_dstate(x[mode_imag_labels])
        res_mode_imag = dres_u.assem_res() + omega*dres_ut.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_imag_labels])

        ret_bvec =  bvec.concatenate_vec(
            [res_state, res_mode_real, res_mode_imag, res_psub, res_omega], labels=HOPF_SYSTEM_LABELS)
        apply_dirichlet_vec(ret_bvec)
        return ret_bvec


    # Make null matrix constants
    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]] 
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_STATE = bmat.BlockMat(mats, (x[state_labels].labels[0], x[state_labels].labels[0]))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in [1]] 
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_SCALAR = bmat.BlockMat(mats, (x[state_labels].labels[0], ('1',)))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]] 
        for row_size in [1]]
    NULL_MAT_SCALAR_STATE = bmat.BlockMat(mats, (('1',), x[state_labels].labels[0]))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in [1]] 
        for row_size in [1]]
    NULL_MAT_SCALAR_SCALAR = bmat.BlockMat(mats, (('1',), ('1',)))

    def hopf_jac(x):
        """Return the Hopf system jacobian"""
        # Set the model state and subglottal pressure (bifurcation parameter)
        assign_state(x)

        # build the Jacobian row by row
        dres_dstate = res.assem_dres_dstate()
        dres_dstatet = res.assem_dres_dstatet()
        jac_1 = [
            dres_dstate, 
            NULL_MAT_STATE_STATE, 
            NULL_MAT_STATE_STATE, 
            res.assem_dres_dcontrol()[:, ['psub']], 
            NULL_MAT_STATE_SCALAR]

        # Set appropriate linearization directions
        omega = x['omega'][0]
        dres_u.set_dstate(x[mode_real_labels])
        dres_ut.set_dstate(x[mode_imag_labels])
        jac_3 = [
            dres_u.assem_dres_dstate() + omega*dres_ut.assem_dres_dstate(), 
            dres_dstate, 
            omega*dres_dstatet, 
            dres_u.assem_dres_dcontrol()[:, ['psub']] + omega*dres_ut.assem_dres_dcontrol()[:, ['psub']], 
            bvec.convert_bvec_to_petsc_colbmat(dres_ut.assem_res())]

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        jac_2 = [
            dres_u.assem_dres_dstate() - omega*dres_ut.assem_dres_dstate(), 
            -omega*dres_dstatet, 
            dres_dstate, 
            dres_u.assem_dres_dcontrol()[:, ['psub']] - omega*dres_ut.assem_dres_dcontrol()[:, ['psub']], 
            bvec.convert_bvec_to_petsc_colbmat(dres_ut.assem_res())]

        jac_4 = [
            NULL_MAT_SCALAR_STATE,
            bvec.convert_bvec_to_petsc_rowbmat(EBVEC),
            NULL_MAT_SCALAR_STATE,
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        jac_5 = [
            NULL_MAT_SCALAR_STATE,
            NULL_MAT_SCALAR_STATE,
            bvec.convert_bvec_to_petsc_rowbmat(EBVEC),
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        ret_mats = [jac_1, jac_2, jac_3, jac_4, jac_5]
        ret_axis_labels = tuple(
            state_labels + mode_real_labels + mode_imag_labels + psub_labels + omega_labels)
        ret_labels = (ret_axis_labels, ret_axis_labels)
        ret_bmat = bmat.concatenate_mat(ret_mats, ret_labels)

        # breakpoint()
        # Apply dirichlet BC by zeroing appropriate matrix rows
        for label in ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']:
            # zero the rows associated with each dirichlet DOF
            for mat in ret_bmat[label, :].array:
                mat.zeroRows(IDX_DIRICHLET, diag=0)

            # Set 1 on the diagonal (zero's one block twice but it shouldn't matter much)
            ret_bmat[label, label].zeroRows(IDX_DIRICHLET, diag=1)

        return ret_bmat

    
    return x, hopf_res, hopf_jac, apply_dirichlet_vec, labels


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

if __name__ == '__main__':
    ## Set up the Hopf system
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
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
    props = res.properties.copy()
    # VF material props
    props['emod'].array[:] = 5e3 * 10
    props['rho'].array[:] = 1.0

    # Fluid separation smoothing props
    # props['zeta_min'].array[:] = 1e-3
    # props['zeta_sep'].array[:] = 1e-3

    y_gap = 0.5 / 10 # Set y gap to 0.5 mm
    y_gap = 1.0
    y_contact_offset = 0.1 / 10
    y_max = res.solid.forms['mesh.mesh'].coordinates()[:, 1].max()
    y_mid = y_max + y_gap
    y_contact = y_mid - y_contact_offset
    props['ycontact'].array[:] = y_contact
    props['kcontact'].array[:] = 1e16

    for model in (res, dres_u, dres_ut):
        model.ymid = y_mid

    control = res.control.copy()

    for model in (res, dres_u, dres_ut):
        model.set_control(control)

    x, hopf_res, hopf_jac, apply_dirichlet_vec, labels = make_hopf_system(res, dres_u, dres_ut, props)
    state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = labels

    ## Test the Hopf jacobian
    x0 = x.copy()
    x0['psub'].array[:] = 800.0*10
    # This value is set to ensure the correct df/dxt matrix when computing eigvals
    # x0['omega'].array[:] = 1.0
    dx = x.copy()
    for subvec in dx:
        subvec.set(0)
    dx['u'].array[:] = 1.0e-7

    apply_dirichlet_vec(dx)
    apply_dirichlet_vec(x0)
    # test_hopf(x0, dx, hopf_res, hopf_jac)

    ## Test solving for fixed-points
    def linear_subproblem(x_n):
        """Linear subproblem of a Newton solver"""
        xhopf_n = x0.copy()
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
            ksp.solve(_rhs_n, _dx_n)

            dx_n = x_n.copy()
            dx_n.set_vec(_dx_n)
            return dx_n
        return assem_res, solve

    xhopf_n = x0.copy()
    x_n = xhopf_n[state_labels]
        
    newton_params = {
        'maximum_iterations': 20
    }
    x_n, info = nleq.newton_solve(x_n, linear_subproblem, norm=bvec.norm, params=newton_params)

    ## Test solving for stabilty (modal analysis of the jacobian)
    # jac = hopf_jac(x_n)
    # _jac = jac.to_petsc()
    # _x_n = x_n.to_petsc()

    # eps = SLEPc.EPS().create()
    # eps.setOperators(_df_dxt, _df_dx)
    # eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)

    # # number of eigenvalues to solve for and dimension of subspace to approximate problem
    # num_eig = 5
    # num_col = 5*num_eig
    # eps.setDimensions(num_eig, num_col)
    # eps.setWhichEigenpairs(SLEPc.EPS.WHICH.LARGEST_REAL)

    breakpoint()

    

    
