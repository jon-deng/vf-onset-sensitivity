"""
Contains code to create the Hopf system
"""
import itertools
from functools import reduce
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

# import blocktensor.subops as gops
import blocktensor.linalg as bla
from blocktensor import vec as bvec
from blocktensor import mat as bmat

import nonlineq as nleq

# pylint: disable=invalid-name

def hopf_state(res):
    """
    Return the state vector for a Hopf system
    """
    X_state = res.state.copy()

    _mode_real_vecs = res.state.copy().array
    _mode_real_labels = [label+'_mode_real' for label in X_state.labels[0]]
    X_mode_real = bvec.BlockVec(_mode_real_vecs, labels=[_mode_real_labels])

    _mode_imag_vecs = res.state.copy().array
    _mode_imag_labels = [label+'_mode_imag' for label in X_state.labels[0]]
    X_mode_imag = bvec.BlockVec(_mode_imag_vecs, labels=[_mode_imag_labels])

    # breakpoint()
    X_psub = res.control[['psub']].copy()

    _omega = X_psub['psub'].copy()
    _omega_vecs = [_omega]
    _omega_labels = [['omega']]
    X_omega = bvec.BlockVec(_omega_vecs, labels=_omega_labels)

    ret = bvec.concatenate_vec([X_state, X_mode_real, X_mode_imag, X_psub, X_omega])
    state_labels = list(X_state.labels[0])
    mode_real_labels = list(X_mode_real.labels[0])
    mode_imag_labels = list(X_mode_imag.labels[0])
    psub_labels = list(X_psub.labels[0])
    omega_labels = list(X_omega.labels[0])

    labels = [state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels]
    return ret, labels

def make_hopf_system(res, dres, props, ee=None):
    """
    Return the residual and jacobian of a Hopf bifurcation system

    This system is based on the augmented system proposed by Griewank and Reddien (1983)

    Parameters
    ----------
    res:
        The fixed-point system residual, F
    dres:
        The linearized fixed-point system residual. This represents
        dF/du * delta u + dF/dut * delta ut + dF/dg * delta g
    """
    for model in (res, dres):
        model.set_properties(props)

    def assign_hopf_system_state(xhopf):
        """
        Sets the fixed-point and bifurcation parameter components of the Hopf state

        The Hopf state consists of 5 blocks representing:
        [fixed-point state, real mode, imaginary mode, bifurcation parameter, onset frequency]

        """
        for model in (res, dres):
            model.set_state(xhopf[state_labels])

            _control = model.control.copy()
            _control['psub'].array[0] = xhopf['psub'].array[0]
            model.set_control(_control)

    IDX_DIRICHLET = np.array(list(res.solid.forms['bc.dirichlet'].get_boundary_values().keys()), dtype=np.int32)
    def apply_dirichlet_bvec(vec):
        """Zeros dirichlet associated indices"""
        for label in ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']:
            # zero the rows associated with each dirichlet DOF
            subvec = vec[label]
            subvec.array[IDX_DIRICHLET] = 0

    def apply_dirichlet_bmat(mat):
        """Zeros dirichlet associated indices"""
        # Apply dirichlet BC by zeroing appropriate matrix rows
        row_labels = ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']
        col_labels = HOPF_LABELS
        for row, col in itertools.product(row_labels, col_labels):
            submat = mat[row, col]
            if row == col:
                submat.zeroRows(IDX_DIRICHLET, diag=1.0)
            else:
                submat.zeroRows(IDX_DIRICHLET, diag=0.0)

    # Create the input vector for the system
    x, labels = hopf_state(res)
    state_labels, mode_real_labels, mode_imag_labels, *_ = labels

    HOPF_LABELS = tuple(reduce(lambda a, b: a+b, labels))

    if ee is None:
        ee = x[state_labels].copy()
        # EBVEC['u'].array[0] = 1.0
        # EBVEC['u'].array[:] = 1.0
        ee.set(1.0)

    # null linearization directions are potentially needed since `dres` is used to compute
    # residuals in multiple directions
    # DSTATE_NULL, DSTATET_NULL = dres.dstate.copy(), dres.dstatet.copy()
    # DSTATE_NULL.set(0.0)
    # DSTATET_NULL.set(0.0)

    def hopf_res(x):
        """Return the Hopf system residual"""
        # Set the model state and subglottal pressure (bifurcation parameter)
        assign_hopf_system_state(x)
        omega = x['omega'][0]

        res_state = res.assem_res()

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_imag_labels])
        dres.set_dstatet(-omega*x[mode_real_labels])
        res_mode_imag = dres.assem_res()

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_real_labels])
        dres.set_dstatet(omega*x[mode_imag_labels])
        res_mode_real = dres.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(ee, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_omega['omega'][0] = bla.dot(ee, x[mode_imag_labels]) - 1.0

        ret_bvec =  bvec.concatenate_vec(
            [res_state, res_mode_real, res_mode_imag, res_psub, res_omega], labels=[HOPF_LABELS])
        apply_dirichlet_bvec(ret_bvec)
        return ret_bvec


    # Make null matrix constants
    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]]
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_STATE = bmat.BlockMat(mats, labels=(x[state_labels].labels[0], x[state_labels].labels[0]))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in [1]]
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_SCALAR = bmat.BlockMat(mats, labels=(x[state_labels].labels[0], ('1',)))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]]
        for row_size in [1]]
    NULL_MAT_SCALAR_STATE = bmat.BlockMat(mats, labels=(('1',), x[state_labels].labels[0]))

    mats = [[bmat.diag_mat(1, 0.0)]]
    NULL_MAT_SCALAR_SCALAR = bmat.BlockMat(mats, labels=(('1',), ('1',)))

    def hopf_jac(x):
        """Return the Hopf system jacobian"""
        # Set the model state and subglottal pressure (bifurcation parameter)
        assign_hopf_system_state(x)

        # build the Jacobian row by row
        dres_dstate = res.assem_dres_dstate()
        dres_dstatet = res.assem_dres_dstatet()

        # Using copys of dres_dstate is important as different dres_dstate locations
        # will require different dirichlet settings on their rows
        jac_row0 = [
            dres_dstate.copy(),
            NULL_MAT_STATE_STATE,
            NULL_MAT_STATE_STATE,
            res.assem_dres_dcontrol()[:, ['psub']],
            NULL_MAT_STATE_SCALAR]

        omega = x['omega'][0]
        # Set appropriate linearization directions
        dres.set_dstate(x[mode_imag_labels])
        dres.set_dstatet(-omega*x[mode_real_labels])
        jac_row2 = [
            dres.assem_dres_dstate(),
            -omega*dres_dstatet.copy(),
            dres_dstate.copy(),
            dres.assem_dres_dcontrol()[:, ['psub']],
            bvec.convert_bvec_to_petsc_colbmat(
                bla.mult_mat_vec(-dres_dstatet, x[mode_real_labels]))]

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_real_labels])
        dres.set_dstatet(omega*x[mode_imag_labels])
        jac_row1 = [
            dres.assem_dres_dstate(),
            dres_dstate.copy(),
            omega*dres_dstatet.copy(),
            dres.assem_dres_dcontrol()[:, ['psub']],
            bvec.convert_bvec_to_petsc_colbmat(
                bla.mult_mat_vec(dres_dstatet, x[mode_imag_labels]))]

        jac_row3 = [
            NULL_MAT_SCALAR_STATE,
            bvec.convert_bvec_to_petsc_rowbmat(ee),
            NULL_MAT_SCALAR_STATE,
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        jac_row4 = [
            NULL_MAT_SCALAR_STATE,
            NULL_MAT_SCALAR_STATE,
            bvec.convert_bvec_to_petsc_rowbmat(ee),
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        ret_mats = [jac_row0, jac_row1, jac_row2, jac_row3, jac_row4]
        ret_labels = (HOPF_LABELS, HOPF_LABELS)
        ret_bmat = bmat.concatenate_mat(ret_mats, ret_labels)

        apply_dirichlet_bmat(ret_bmat)
        return ret_bmat

    info = {
        'dirichlet_dofs': IDX_DIRICHLET
    }
    return x, hopf_res, hopf_jac, apply_dirichlet_bvec, apply_dirichlet_bmat, labels, info

def normalize_eigenvector_by_hopf_condition(evec_real, evec_imag, evec_ref):
    """
    Scales real and imaginary components of an eigenvector by a complex constant

    Let the complex eigenvector be `evec == evec_real +1j*evec_imag`. Then the
    function computes a constant `A * exp(1j * theta)` such that:
    inner(evec_ref, real(A * exp(1j * theta) * evec)) == 0
    inner(evec_ref, im(A * exp(1j * theta) * evec)) == 1

    This is the Hopf normalization.
    """
    a = bla.dot(evec_ref, evec_real)
    b = bla.dot(evec_ref, evec_imag)

    theta = np.arctan(a/b)
    amp = (a*np.sin(theta) + b*np.cos(theta))**-1

    ret_evec_real = amp*(evec_real*np.cos(theta) - evec_imag*np.sin(theta))
    ret_evec_imag = amp*(evec_real*np.sin(theta) + evec_imag*np.cos(theta))
    return ret_evec_real, ret_evec_imag

def solve_fixed_point(res, xfp_0, newton_params=None):
    """
    Solve for a fixed-point

    Parameters
    ----------
    res :
        Object representing the fixed-point residual
    xfp_0 :
        initial guess for the fixed-point state
    psub :
        the value of the bifurcation parameter (subglottal pressure)
    newton_params :
        parameters for the newton solver
    """

    ZERO_STATET = res.statet.copy()
    ZERO_STATET.set(0.0)
    res.set_statet(ZERO_STATET)

    IDX_DIRICHLET = np.array(
        list(res.solid.forms['bc.dirichlet'].get_boundary_values().keys()),
        dtype=np.int32)
    def apply_dirichlet_bvec(vec):
        """Applies the dirichlet BC to a vector"""
        for label, subvec in vec.items():
            if label in ['u', 'v']:
                subvec.setValues(IDX_DIRICHLET, np.zeros(IDX_DIRICHLET.size))

    def apply_dirichlet_bmat(mat, diag=1.0):
        """Applies the dirichlet BC to a matrix"""
        for row_label in ['u', 'v']:
            for col_label in mat.labels[1]:
                submat = mat[row_label, col_label]
                if row_label == col_label:
                    submat.zeroRows(IDX_DIRICHLET, diag=diag)
                else:
                    submat.zeroRows(IDX_DIRICHLET, diag=0.0)

    def linear_subproblem_fp(xfp_n):
        """Linear subproblem of a Newton solver"""
        res.set_state(xfp_n)

        res_n = res.assem_res()
        jac_n = res.assem_dres_dstate()
        apply_dirichlet_bvec(res_n)
        apply_dirichlet_bmat(jac_n)

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

            dx_n = xfp_n.copy()
            dx_n.set_vec(_dx_n)
            return dx_n
        return assem_res, solve

    if newton_params is None:
        newton_params = {
            'maximum_iterations': 20
        }
    xfp_n, info = nleq.newton_solve(xfp_0, linear_subproblem_fp, norm=bvec.norm, params=newton_params)
    return xfp_n, info

def solve_ls(res, xfp):
    """
    Return a set of modes for the linear stability problem (ls)

    This solves the eigenvalue problem:
    -omega df/dxt ex = df/dx ex,
    in the transformed form
    df/dxt ex = lambda df/dx ex,
    where lambda=-1/omega, and ex is a generalized eigenvector

    Parameters
    ----------
    res :
        Object representing the fixed-point residual
    xfp :
        The fixed point to solve the LS problem at
    """
    res.set_state(xfp)

    ZERO_STATET = res.statet.copy()
    ZERO_STATET.set(0.0)
    res.set_statet(ZERO_STATET)

    IDX_DIRICHLET = np.array(
        list(res.solid.forms['bc.dirichlet'].get_boundary_values().keys()),
        dtype=np.int32)

    def apply_dirichlet_bmat(mat, diag=1.0):
        """Applies the dirichlet BC to a matrix"""
        for row_label in ['u', 'v']:
            for col_label in mat.labels[1]:
                submat = mat[row_label, col_label]
                if row_label == col_label:
                    submat.zeroRows(IDX_DIRICHLET, diag=diag)
                else:
                    submat.zeroRows(IDX_DIRICHLET, diag=0.0)

    df_dx = res.assem_dres_dstate()
    df_dxt = res.assem_dres_dstatet()

    # Set dirichlet conditions for the mass matrix
    # Setting a small value for df_dxt on the diagonal ensures eigenvalues
    # associated with dirichlet DOFs will have a very small value of 1e-10
    apply_dirichlet_bmat(df_dx, diag=1.0)
    apply_dirichlet_bmat(df_dxt, diag=1.0e-10)

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
    # print("Omegas:", omegas)

    eigvecs_real = [res.state.copy() for jj in range(eps.getConverged())]
    eigvecs_imag = [res.state.copy() for jj in range(eps.getConverged())]

    for jj in range(eps.getConverged()):
        eigvec_real = _df_dx.getVecRight()
        eigvec_imag = _df_dx.getVecRight()
        eps.getEigenvector(jj, eigvec_real, eigvec_imag)

        eigvecs_real[jj].set_vec(eigvec_real)
        eigvecs_imag[jj].set_vec(eigvec_imag)

    return omegas, eigvecs_real, eigvecs_imag