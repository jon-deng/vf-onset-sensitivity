"""
Contains code to create the Hopf system

The hopf system represents the conditions satisified at a Hopf bifurcation. Consider a nonlinear
dynamical system (the `res` model) defined by
F(x_t, x; ...) ,
where x_t is the state time derivative and x is the state.

The first condition is a fixed point:
F(x_t, x; ...) = 0

The second condition is the linearized dynamics are periodic and neutrally stable. The linearized
dynamics are given by
d_x_t F delta x_t + d_x delta x = 0.
Assuming an ansatz of
delta x_t = exp(omega_r + 1j*omega_i) * zeta
and substituting in the above
will get the mode shape conditions. Note that this is a different sign convention from that
adopted by Griewank and Reddien where they assume
delta x_t = exp(omega_r - 1j*omega_i) * zeta
so the Hopf equations below are slightly different.
"""
import itertools
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

    _mode_real_vecs = res.state.copy().subtensors_flat
    _mode_real_labels = [label+'_mode_real' for label in X_state.labels[0]]
    X_mode_real = bvec.BlockVector(_mode_real_vecs, labels=[_mode_real_labels])

    _mode_imag_vecs = res.state.copy().subtensors_flat
    _mode_imag_labels = [label+'_mode_imag' for label in X_state.labels[0]]
    X_mode_imag = bvec.BlockVector(_mode_imag_vecs, labels=[_mode_imag_labels])

    # breakpoint()
    X_psub = res.control[['psub']].copy()

    _omega = X_psub['psub'].copy()
    _omega_vecs = [_omega]
    _omega_labels = [['omega']]
    X_omega = bvec.BlockVector(_omega_vecs, labels=_omega_labels)

    ret = bvec.concatenate_vec([X_state, X_mode_real, X_mode_imag, X_psub, X_omega])
    state_labels = list(X_state.labels[0])
    mode_real_labels = list(X_mode_real.labels[0])
    mode_imag_labels = list(X_mode_imag.labels[0])
    psub_labels = list(X_psub.labels[0])
    omega_labels = list(X_omega.labels[0])

    labels = [state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels]
    return ret, labels


class HopfModel:
    """
    Represents the system of equations defining a Hopf bifurcation

    This sytem of equations is given by Griewank and Reddien (1983)

    Parameters
    ----------
        res, dres: The system dynamics residual, and linearized residuals
    """

    def __init__(self, res, dres, ee=None):
        self.res = res
        self.dres = dres

        self.state, _component_labels = hopf_state(res)
        self.properties = res.properties.copy()

        # These labels represent the 5 sub-blocks in Griewank and Reddien's equations
        self.labels_hopf_components = _component_labels
        (self.labels_state,
            self.labels_mode_real,
            self.labels_mode_imag,
            self.labels_psub,
            self.labels_omega) = _component_labels

        self.IDX_DIRICHLET = np.array(
            list(res.solid.forms['bc.dirichlet'].get_boundary_values().keys()),
            dtype=np.int32)

        if ee is None:
            ee = self.state[self.labels_state].copy()
            # EBVEC['u'].array[0] = 1.0
            # EBVEC['u'].array[:] = 1.0
            ee.set(1.0)
        self.EE = ee

    def set_properties(self, props):
        self.properties[:] = props
        for model in (self.res, self.dres):
            model.set_properties(props)

    def set_state(self, xhopf):
        self.state[:] = xhopf
        for model in (self.res, self.dres):
            model.set_state(xhopf[self.labels_state])

            _control = model.control.copy()
            _control['psub'].array[0] = xhopf['psub'].array[0]
            model.set_control(_control)

    def apply_dirichlet_bvec(self, vec):
        """Zeros dirichlet associated indices on the Hopf state"""
        for label in ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']:
            # zero the rows associated with each dirichlet DOF
            subvec = vec[label]
            subvec.array[self.IDX_DIRICHLET] = 0

    def apply_dirichlet_bmat(self, mat):
        """Zeros dirichlet associated indices"""
        # Apply dirichlet BC by zeroing appropriate matrix rows
        row_labels = ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']
        col_labels = self.state.labels[0]
        for row, col in itertools.product(row_labels, col_labels):
            submat = mat[row, col]
            if row == col:
                submat.zeroRows(self.IDX_DIRICHLET, diag=1.0)
            else:
                submat.zeroRows(self.IDX_DIRICHLET, diag=0.0)

    def assem_res(self):
        """Return the Hopf system residual"""
        # Load the needed 'local variables'
        res, dres = self.res, self.dres
        mode_real_labels, mode_imag_labels = self.labels_mode_real, self.labels_mode_imag
        x = self.state
        ee = self.EE

        # Set the model state and subglottal pressure (bifurcation parameter)
        omega = x['omega'][0]

        res_state = res.assem_res()

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_real_labels])
        dres.set_dstatet(-omega*x[mode_imag_labels])
        res_mode_real = dres.assem_res()

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_imag_labels])
        dres.set_dstatet(omega*x[mode_real_labels])
        res_mode_imag = dres.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(ee, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_omega['omega'][0] = bla.dot(ee, x[mode_imag_labels]) - 1.0

        ret_bvec = bvec.concatenate_vec(
            [res_state, res_mode_real, res_mode_imag, res_psub, res_omega],
            labels=self.state.labels)
        return ret_bvec

    def assem_dres_dstate(self):
        """Return the Hopf system jacobian"""
        # Load the needed 'local variables'
        res, dres = self.res, self.dres
        state_labels = self.labels_state
        mode_real_labels, mode_imag_labels = self.labels_mode_real, self.labels_mode_imag
        x = self.state
        ee = self.EE

        # Make null matrix constants
        mats = [
            [bmat.zero_mat(row_size, col_size)
                for col_size in x[self.labels_state].bshape[0]]
            for row_size in x[state_labels].bshape[0]]
        NULL_MAT_STATE_STATE = bmat.BlockMatrix(mats, labels=(x[state_labels].labels[0], x[state_labels].labels[0]))

        mats = [
            [bmat.zero_mat(row_size, col_size) for col_size in [1]]
            for row_size in x[state_labels].bshape[0]]
        NULL_MAT_STATE_SCALAR = bmat.BlockMatrix(mats, labels=(x[state_labels].labels[0], ('1',)))

        mats = [
            [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]]
            for row_size in [1]]
        NULL_MAT_SCALAR_STATE = bmat.BlockMatrix(mats, labels=(('1',), x[state_labels].labels[0]))

        mats = [[bmat.diag_mat(1, 0.0)]]
        NULL_MAT_SCALAR_SCALAR = bmat.BlockMatrix(mats, labels=(('1',), ('1',)))

        ## Build the Jacobian row by row
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
        dres.set_dstate(x[mode_real_labels])
        dres.set_dstatet(omega*x[mode_imag_labels])
        jac_row1 = [
            dres.assem_dres_dstate(),
            dres_dstate.copy(),
            -omega*dres_dstatet.copy(),
            dres.assem_dres_dcontrol()[:, ['psub']],
            bvec.convert_bvec_to_petsc_colbmat(
                bla.mult_mat_vec(-dres_dstatet, x[mode_imag_labels]))]

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_imag_labels])
        dres.set_dstatet(-omega*x[mode_real_labels])
        jac_row2 = [
            dres.assem_dres_dstate(),
            omega*dres_dstatet.copy(),
            dres_dstate.copy(),
            dres.assem_dres_dcontrol()[:, ['psub']],
            bvec.convert_bvec_to_petsc_colbmat(
                bla.mult_mat_vec(dres_dstatet, x[mode_real_labels]))]

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
        ret_labels = self.state.labels+self.state.labels
        ret_bmat = bmat.concatenate_mat(ret_mats, labels=ret_labels)
        return ret_bmat


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

def normalize_eigenvector_amplitude(evec_real, evec_imag):
    """
    Scales real and imaginary components of an eigenvector so it has unit norm
    """
    ampl = 1/(evec_real.norm()**2 + evec_imag.norm()**2)**0.5
    return ampl*evec_real, ampl*evec_imag


def solve_petsc_lu():

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

def solve_reduced_gradient(functional, hopf: HopfModel):
    """Solve for the reduced gradient of a functional"""

    dg_dxhopf = functional.assem_dg_dstate()
    dg_dprops = functional.assem_dg_dprops()
    dg_dprops_ampl = functional.assem_dg_dprops_ampl()
    _dg_dxhopf = dg_dxhopf.to_petsc()

    dres_dstate_adj = hopf.assem_dres_dstate().tranpose()
    hopf.apply_dirichlet_bmat(dres_dstate_adj)
    _dres_dstate_adj = dres_dstate_adj.to_petsc()

    dg_dreshopf = dg_dxhopf.copy()
    _dg_dreshopf = _dres_dstate_adj.getVecRight()

    # Solve the adjoint problem for the 'adjoint state'
    ksp = PETSc.KSP().create()
    ksp.setType(ksp.Type.PREONLY)

    pc = ksp.getPC()
    pc.setType(pc.Type.LU)

    ksp.setOperators(_dres_dstate_adj)
    ksp.setUp()
    ksp.solve(_dg_dxhopf, _dg_dreshopf)

    dg_dreshopf[:] = _dg_dreshopf

    # Compute the reduced gradient
    dres_dprops = hopf.assem_dres_dprops()
    dg_dprops = blinalg.mult_mat_vec(dres_dprops, -dg_dreshopf)

    return dg_dprops

