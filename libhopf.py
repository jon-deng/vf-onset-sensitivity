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

from typing import Tuple, List, Dict, Optional
import itertools
import warnings
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import h5py

import nonlineq as nleq
from femvf.models.dynamical import base as dynbase
import blockarray.h5utils as h5utils
import blockarray.subops as gops
import blockarray.linalg as bla
from blockarray import blockvec as bvec, blockmat as bmat
# import libfunctionals as libfunc

# pylint: disable=invalid-name
ListPair = Tuple[List[float], List[float]]

def _hopf_state(res):
    """
    Return the state vector for a Hopf system
    """
    X_state = res.state.copy()

    _mode_real_vecs = res.state.copy().subarrays_flat
    _mode_real_labels = [label+'_mode_real' for label in X_state.labels[0]]
    X_mode_real = bvec.BlockVector(_mode_real_vecs, labels=[_mode_real_labels])

    _mode_imag_vecs = res.state.copy().subarrays_flat
    _mode_imag_labels = [label+'_mode_imag' for label in X_state.labels[0]]
    X_mode_imag = bvec.BlockVector(_mode_imag_vecs, labels=[_mode_imag_labels])

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
        e_mode: a normalization vector for the real and imaginary mode components
    """

    def __init__(self, res, dres, e_mode=None):
        self.res = res
        self.dres = dres

        self.state, _component_labels = _hopf_state(res)
        self.props = res.props.copy()

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

        if e_mode is None:
            e_mode = self.state[self.labels_state].copy()
            e_mode.set(1.0)
        self.E_MODE = e_mode

    def set_props(self, props):
        self.props[:] = props
        for model in (self.res, self.dres):
            model.set_props(props)

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
        ee = self.E_MODE

        # Set the model state and subglottal pressure (bifurcation parameter)
        omega = x['omega'][0]

        res_state = res.assem_res()

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_real_labels])
        dres.set_dstatet(-float(omega)*x[mode_imag_labels])
        res_mode_real = dres.assem_res()

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_imag_labels])
        dres.set_dstatet(float(omega)*x[mode_real_labels])
        res_mode_imag = dres.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(ee, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_omega['omega'][0] = bla.dot(ee, x[mode_imag_labels]) - 1.0

        ret_bvec = bvec.concatenate_vec(
            [res_state, res_mode_real, res_mode_imag, res_psub, res_omega],
            labels=self.state.labels)

        self.apply_dirichlet_bvec(ret_bvec)
        return ret_bvec

    def assem_dres_dstate(self):
        """Return the Hopf system jacobian"""
        # Load the needed 'local variables'
        res, dres = self.res, self.dres
        state_labels = self.labels_state
        mode_real_labels, mode_imag_labels = self.labels_mode_real, self.labels_mode_imag
        x = self.state
        ee = self.E_MODE

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
        dres.set_dstatet(float(omega)*x[mode_imag_labels])
        jac_row1 = [
            dres.assem_dres_dstate(),
            dres_dstate.copy(),
            -float(omega)*dres_dstatet.copy(),
            dres.assem_dres_dcontrol()[:, ['psub']],
            bvec.to_block_colmat(
                bla.mult_mat_vec(-dres_dstatet, x[mode_imag_labels]))]

        # Set appropriate linearization directions
        dres.set_dstate(x[mode_imag_labels])
        dres.set_dstatet(-float(omega)*x[mode_real_labels])
        jac_row2 = [
            dres.assem_dres_dstate(),
            float(omega)*dres_dstatet.copy(),
            dres_dstate.copy(),
            dres.assem_dres_dcontrol()[:, ['psub']],
            bvec.to_block_colmat(
                bla.mult_mat_vec(dres_dstatet, x[mode_real_labels]))]

        jac_row3 = [
            NULL_MAT_SCALAR_STATE,
            bvec.to_block_rowmat(ee),
            NULL_MAT_SCALAR_STATE,
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        jac_row4 = [
            NULL_MAT_SCALAR_STATE,
            NULL_MAT_SCALAR_STATE,
            bvec.to_block_rowmat(ee),
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        ret_mats = [jac_row0, jac_row1, jac_row2, jac_row3, jac_row4]
        ret_labels = self.state.labels+self.state.labels
        ret_bmat = bmat.concatenate_mat(ret_mats, labels=ret_labels)
        return ret_bmat

    def assem_dres_dprops(self):
        res, dres = self.res, self.dres
        (state_labels,
            mode_real_labels,
            mode_imag_labels,
            psub_labels,
            omega_labels) = self.labels_hopf_components

        # Assemble the matrix by rows
        omega = self.state['omega'][0]

        row0 = [res.assem_dres_dprops()]

        dres.set_dstate(self.state[mode_real_labels])
        dres.set_dstatet(-float(omega)*self.state[mode_imag_labels])
        row1 = [dres.assem_dres_dprops()]

        dres.set_dstate(self.state[mode_imag_labels])
        dres.set_dstatet(float(omega)*self.state[mode_real_labels])
        row2 = [dres.assem_dres_dprops()]

        _mats = [bmat.zero_mat(1, m) for m in self.props.bshape[0]]
        row3 = [
            bmat.BlockMatrix(_mats, (1, len(_mats)), (psub_labels,)+self.props.labels)
            ]
        row4 = [
            bmat.BlockMatrix(_mats, (1, len(_mats)), (omega_labels,)+self.props.labels)
            ]

        bmats = [row0, row1, row2, row3, row4]
        return bmat.concatenate_mat(
            bmats, labels=self.state.labels+self.props.labels)


def max_real_omega(model: dynbase.DynamicalSystem, psub: float) -> Tuple[float, bvec.BlockVector]:
    """
    Return the maximum real frequency component for a linearized dynamical model

    Parameters
    ----------
    model : femvf.models.dynamical.base.DynamicalSystem
    psub : float
    """
    # First set the bifurcation parameter
    control = model.control
    control['psub'][:] = psub
    model.set_control(control)

    # Solve the for the fixed point
    xfp_0 = model.state.copy()
    xfp_0.set(0)
    xfp, _info = solve_fixed_point(model)

    # Solve for linear stability around the fixed point
    omegas, eigvecs_real, eigvecs_imag = solve_linear_stability(model, xfp)

    idx_max = np.argmax(omegas.real)
    return omegas[idx_max], eigvecs_real[idx_max], eigvecs_imag[idx_max]

def bound_hopf_bifurcations(
        model: dynbase.DynamicalSystem,
        bound_pairs: ListPair,
        omega_pairs: ListPair=None,
        nsplit: int=2,
        tol: float=100.0
    ) -> Tuple[ListPair, ListPair]:
    """
    Bound the onset pressure where a Hopf bifurcation occurs

    Parameters
    ----------
    model : femvf.models.dynamical.base.DynamicalSystem
    bound_pairs : Tuple[List[float, ...], List[float, ...]]
        A list of lower/upper bound pairs, (lb, ub), of the bifurcation
        parameters (subglottal pressure). Each bound pair is checked to see if a
        Hopf bifurcation occurs between them.
    omega_pairs : Tuple[List[float, ...], List[float, ...]]
        The maximum real omega at the lower/upper bounds. If the maximum real
        omega component changes from negative to positive from the lower to
        upper bound, then a Hopf bifurcation must occur in that interval.
    nsplit : int
        The number of intervals to split a bound pair into for searching for a
        refined bifurcation parameter
    tol : float
        Tolerance on the bound pairs
    """
    # If real omega are not supplied for each bound pair, compute it here
    lbs, ubs = bound_pairs
    if omega_pairs is None:
        omega_pairs = (
            [max_real_omega(model, lb)[0].real for lb in lbs],
            [max_real_omega(model, ub)[0].real for ub in ubs],
        )

    # Filter bound pairs so only pairs that contain Hopf bifurcations are present
    has_onset = [
        lb < 0 and ub >= 0
        for lb, ub in zip(*omega_pairs)
    ]

    _lbs = [lb for lb, valid in zip(lbs, has_onset) if valid]
    _ubs = [ub for ub, valid in zip(ubs, has_onset) if valid]

    lomegas, uomegas = omega_pairs
    _lomegas = [lb for lb, valid in zip(lomegas, has_onset) if valid]
    _uomegas = [ub for ub, valid in zip(uomegas, has_onset) if valid]

    # Check if the bound pairs containing onset all satisfy the tolerance;
    # if they do return the bounds,
    # and if they don't, split the bounds into smaller segments and retry
    tols = [ub-lb for lb, ub in zip(lbs, ubs)]
    if all([_tol <= tol for _tol in tols]):
        return (_lbs, _ubs), (_lomegas, _uomegas)
    else:
        # Split the pairs into `nsplit` segments and calculate important stuff
        # at the interior points separating segments
        # This is a nested list containing, for each lb/ub pair, a list of interior
        # segments
        bounds_points = [
            list(np.linspace(lb, ub, nsplit+1)[1:-1]) for lb, ub in zip(_lbs, _ubs)]
        bounds_omegas = [
            [max_real_omega(model, psub)[0].real for psub in psubs] for psubs in bounds_points
        ]

        # Join the computed interior points for each bound into a new set of bound pairs and omega pairs
        ret_lbs = [
            x for lb, bound_points in zip(_lbs, bounds_points)
            for x in ([lb] + bound_points)
            ]
        ret_ubs = [
            x for ub, bound_points in zip(_ubs, bounds_points)
            for x in (bound_points + [ub])
            ]

        ret_lomegas = [
            x for lomega, omegas in zip(_lomegas, bounds_omegas)
            for x in ([lomega] + omegas)
            ]
        ret_uomegas = [
            x for uomega, omegas in zip(_uomegas, bounds_omegas)
            for x in (omegas + [uomega])
            ]
        return bound_hopf_bifurcations(
            model, (ret_lbs, ret_ubs), (ret_lomegas, ret_uomegas),
            nsplit=nsplit, tol=tol)

def gen_hopf_initial_guess(
        hopf: HopfModel,
        eref: bvec.BlockVector,
        bound_pairs: ListPair,
        omega_pairs: Optional[ListPair]=None,
        nsplit: int=2,
        tol: float=100.0
    ) -> bvec.BlockVector:
    """
    Generate an initial guess for a Hopf system by bounding the bifurcation point
    """
    res = hopf.res
    # Find lower/upper bounds for the Hopf bifurcation point
    (lbs, ubs), _ = bound_hopf_bifurcations(
        res, bound_pairs, omega_pairs=omega_pairs, nsplit=nsplit, tol=tol)

    if len(ubs) > 1:
        raise UserWarning("More than one Hopf bifurcation point found")
    if len(ubs) == 0:
        raise UserWarning(f"No Hopf bifurcation found between bounds {bound_pairs[0]} and {bound_pairs[1]}")

    # Use the upper bound to generate an initial guess for the bifurcation
    # First set the model subglottal pressure to the upper bound
    psub = ubs[0]
    control = res.control
    control['psub'][:] = psub
    res.set_control(control)

    # Solve for the fixed point
    x_fp0 = res.state.copy()
    x_fp0.set(0.0)
    x_fp, _info = solve_fixed_point(res)

    # Solve for linear stability around the fixed point
    omegas, eigvecs_real, eigvecs_imag = solve_linear_stability(res, x_fp)
    idx_max = np.argmax(omegas.real)

    x_mode_real = eigvecs_real[idx_max]
    x_mode_imag = eigvecs_imag[idx_max]

    x_mode_real, x_mode_imag = normalize_eigenvector_by_hopf_condition(
        x_mode_real, x_mode_imag, eref)

    x_omega = bvec.convert_subtype_to_petsc(
        bvec.BlockVector([np.array([omegas[idx_max].imag])], labels=(('omega',),))
        )

    x_psub = bvec.convert_subtype_to_petsc(
        bvec.BlockVector([np.array([psub])], labels=(('psub',),))
        )

    x_hopf = hopf.state.copy()
    for labels, subvector in zip(hopf.labels_hopf_components, [x_fp, x_mode_real, x_mode_imag, x_psub, x_omega]):
        x_hopf[labels] = subvector

    return x_hopf


## Normalize eigenvectors

def normalize_eigenvector_by_hopf_condition(
        evec_real: bvec.BlockVector,
        evec_imag: bvec.BlockVector,
        evec_ref: bvec.BlockVector
    ) -> Tuple[bvec.BlockVector, bvec.BlockVector]:
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
    amp = float((a*np.sin(theta) + b*np.cos(theta))**-1)

    ret_evec_real = amp*(evec_real*float(np.cos(theta)) - evec_imag*float(np.sin(theta)))
    ret_evec_imag = amp*(evec_real*float(np.sin(theta)) + evec_imag*float(np.cos(theta)))
    return ret_evec_real, ret_evec_imag

def normalize_eigenvector_amplitude(
        evec_real: bvec.BlockVector,
        evec_imag: bvec.BlockVector
    ) -> Tuple[bvec.BlockVector, bvec.BlockVector]:
    """
    Scales real and imaginary components of an eigenvector so it has unit norm
    """
    ampl = 1/(evec_real.norm()**2 + evec_imag.norm()**2)**0.5
    return ampl*evec_real, ampl*evec_imag


## Solve the Hopf system, fixed point, etc.

def solve_fixed_point(res: dynbase.DynamicalSystem, n_load=5):
    """
    Solve for a fixed-point

    This is high-level solver which uses intermediate loading steps with the
    Newton method to find the fixed-point for target subglottal pressure.
    """
    # The target final subglottal pressure
    psub_n = float(res.control['psub'][0])

    # Use a sequence of intermediate loading steps to generate good initial
    # guesses for the next fixed-point newton solve
    xfp_0 = res.state.copy()
    xfp_0.set(0.0)

    for psub in np.linspace(0, psub_n, n_load+1)[1:]:
        control = res.control
        control['psub'][0] = psub
        res.set_control(control)

        xfp_0, info = solve_fixed_point_newton(res, xfp_0)

    xfp_n = xfp_0
    return xfp_n, info

def solve_fixed_point_newton(
        res: dynbase.DynamicalSystem,
        xfp_0: bvec.BlockVector,
        newton_params: Optional[Dict]=None
    ) -> Tuple[bvec.BlockVector, Dict]:
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
            _rhs_n = rhs_n.to_mono_petsc()
            _jac_n = jac_n.to_mono_petsc()
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

def solve_hopf_newton(
        hopf: HopfModel,
        xhopf_0: bvec.BlockVector,
        out=None, newton_params=None
    ) -> Tuple[bvec.BlockVector, Dict]:
    """Solve the nonlinear Hopf problem using a newton method"""
    if out is None:
        out = xhopf_0.copy()

    def linear_subproblem(xhopf_n):
        """Linear subproblem of a Newton solver"""
        hopf.set_state(xhopf_n)

        res_n = hopf.assem_res()
        jac_n = hopf.assem_dres_dstate()
        hopf.apply_dirichlet_bvec(res_n)
        hopf.apply_dirichlet_bmat(jac_n)

        def assem_res():
            """Return residual"""
            return res_n

        def solve(rhs_n):
            """Return jac^-1 res"""
            _rhs_n = rhs_n.to_mono_petsc()
            _jac_n = jac_n.to_mono_petsc()
            _dx_n = _jac_n.getVecRight()

            _dx_n, _ = gops.solve_petsc_lu(_jac_n, _rhs_n, out=_dx_n)

            dx_n = xhopf_n.copy()
            dx_n.set_vec(_dx_n)
            return dx_n
        return assem_res, solve

    if newton_params is None:
        newton_params = {
            'maximum_iterations': 10
        }

    out[:], info = nleq.newton_solve(xhopf_0, linear_subproblem, norm=bvec.norm, params=newton_params)
    return out, info

def solve_linear_stability(
        res: dynbase.DynamicalSystem,
        xfp: bvec.BlockVector
    ) -> Tuple[List[float], List[bvec.BlockVector], List[bvec.BlockVector]]:
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

    _df_dx = df_dx.to_mono_petsc()
    _df_dxt = df_dxt.to_mono_petsc()

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

def solve_reduced_gradient(
        functional: 'libfunctionals.GenericFunctional',
        hopf: HopfModel
    ) -> bvec.BlockVector:
    """Solve for the reduced gradient of a functional"""

    dg_dprops = functional.assem_dg_dprops()
    dg_dx = functional.assem_dg_dstate()
    _dg_dx = dg_dx.to_mono_petsc()

    dres_dx_adj = hopf.assem_dres_dstate().transpose()
    hopf.apply_dirichlet_bmat(dres_dx_adj)
    _dres_dx_adj = dres_dx_adj.to_mono_petsc()

    dg_dres = dg_dx.copy()
    _dg_dres = _dres_dx_adj.getVecRight()

    # Solve the adjoint problem for the 'adjoint state'
    _dg_dres, _ = gops.solve_petsc_lu(_dres_dx_adj, _dg_dx, out=_dg_dres)
    dg_dres.set_vec(_dg_dres)

    # Compute the reduced gradient
    dres_dprops = hopf.assem_dres_dprops()
    return bla.mult_mat_vec(dres_dprops.transpose(), -dg_dres) + dg_dprops


class ReducedGradient:
    """
    This class handles solution of reduced gradients on the Hopf model

    Consider the functional, g(x; p, camp), where x, p, camp represent the Hopf
    state vector, parameters, and complex amplitude, respectively. Also consider
    the Hopf system defined by F(x; p) = 0. The reduced gradient is the function
    g^(p, camp)=g(x(p), p, camp) where x is implicitly constrained by the Hopf
    system.

    The reduced gradient is difficult to solve for since solution of the Hopf
    system is not straightforward without a good initial guess for a given
    parameter set. This class tries to supply reasonable initial guesses for
    solving the Hopf system so that the reduced graident can be treated as
    purely a function of parameters and plugged into an optimization
    loop.

    Parameters
    ----------
    res : HopfModel
    functional : libfunctionals.GenericFunctional
    newton_params :
        dictionary specifying newton solver parameters for solving the Hopf
        system
    """

    def __init__(self, func, res, newton_params=None):
        self.func = func
        self.res = res

        self._hist_state = [self.res.state.copy()]
        self._hist_props = [self.props.copy()]
        self._hist_camp = [self.camp.copy()]

        if newton_params is None:
            newton_params = {}
        self._newton_params = newton_params

    @property
    def camp(self):
        return self.func.camp

    @property
    def props(self):
        return self.res.props

    @property
    def hist_props(self):
        return self._hist_props

    @property
    def hist_state(self):
        return self._hist_state

    @property
    def hist_camp(self):
        return self._hist_camp

    def _update_hopf(self):
        """
        Keeps track of Hopf solution when properties are updated

        This should be called whenever the properties are set, since the
        Hopf system solution will change whenever the properties change
        """
        ## Update the hopf system by solving the Hopf bifurcation equations

        # Use the latest state in the history of Hopf states as an initial guess
        # for solving the Hopf system with the new parameters
        # max_n_iter = 2
        n_iter = 0
        xhopf_0 = self.hist_state[-1]
        while True:
            xhopf_n, info = solve_hopf_newton(
                self.res, xhopf_0, newton_params=self._newton_params)

            n_iter += 1
            if n_iter > 1:
                if info['status'] != 0:
                    warnings.warn(
                        "Hopf system could not be solved with initial guess!"
                    )
                break
            elif info['status'] != 0:
                warnings.warn(
                    "Hopf system could not be solved with initial guess."
                    " Attemping to find a better initial guess."
                )
                # Arbitratrily search over the range 0 to 1500 Pa for Hopf bifurcation
                PSUBS = 10*np.arange(0, 1500, 100)
                omegas_max = [max_real_omega(self.res.res, psub)[0].real for psub in PSUBS]
                has_transition = [
                    omega2 >= 0.0 and omega1 < 0.0
                    for omega1, omega2 in zip(omegas_max[:-1], omegas_max[1:])
                ]
                idxs_bif = np.arange(PSUBS.size-1)[has_transition]
                if idxs_bif.size == 0:
                    raise RuntimeError("No Hopf bifurcations detected")
                elif idxs_bif.size > 1:
                    warnings.warn("Found more than one Hopf bifurcation pressure; using the smallest one")

                idx_bif = idxs_bif[0]
                lbs = [PSUBS[idx_bif]]
                ubs = [PSUBS[idx_bif+1]]
                bounds = (lbs, ubs)
                omega_lbs = [omegas_max[idx_bif]]
                omega_ubs = [omegas_max[idx_bif+1]]
                omega_pairs = (omega_lbs, omega_ubs)
                xhopf_0 = gen_hopf_initial_guess(self.res, self.res.E_MODE, bounds, omega_pairs, tol=5.0)


        self.hist_state.append(xhopf_n.copy())
        self.hist_props.append(self.props.copy())
        self.hist_camp.append(self.camp.copy())

        self.res.set_state(xhopf_n)

        return xhopf_n, info

    def set_camp(self, camp):
        """Set the complex amplitude"""
        self.func.set_camp(camp)

    def set_props(self, props):
        """
        Set the model properties
        """
        self.func.set_props(props)
        self.res.set_props(props)
        hopf_state, info = self._update_hopf()
        return hopf_state, info

    def assem_g(self):
        return self.func.assem_g()

    def assem_dg_dprops(self):
        return solve_reduced_gradient(self.func, self.res)

    def assem_dg_dcamp(self):
        return self.func.assem_dg_dcamp()


class OptGradManager:
    """
    An object that provides a `grad(x)` function for black-box optimizers

    Parameters
    ----------
    redu_grad : ReducedGradient
    f : h5py.File
        An h5 file to record function eval information

    Returns
    -------
    opt_obj : Callable[[array_like], float]
    opt_grad : Callable[[array_like], array_like]
    """

    def __init__(self, redu_grad: ReducedGradient, f: h5py.Group):
        self.redu_grad = redu_grad
        self.f = f

        # Add groups to the h5 file to store optimization history
        param_labels = (redu_grad.props.labels[0]+redu_grad.camp.labels[0],)
        param_bshape = (redu_grad.props.bshape[0]+redu_grad.camp.bshape[0],)
        h5utils.create_resizable_block_vector_group(
            f.create_group('parameters'),
            param_labels,
            param_bshape)
        h5utils.create_resizable_block_vector_group(
            f.create_group('grad'),
            param_labels,
            param_bshape)
        h5utils.create_resizable_block_vector_group(
            f.create_group('hopf_state'),
            redu_grad.res.state.labels,
            redu_grad.res.state.bshape)
        f.create_dataset('objective', (0,), maxshape=(None,))

        # Newton solver convergence info for solving the Hopf bifurcation system
        f.create_dataset('hopf_newton_num_iter', (0,), maxshape=(None,))
        f.create_dataset('hopf_newton_status', (0,), maxshape=(None,))
        f.create_dataset('hopf_newton_abs_err', (0,), maxshape=(None,))
        f.create_dataset('hopf_newton_rel_err', (0,), maxshape=(None,))

    def _update_h5(self, hopf_state, info, p):
        """
        Update the Hopf model properties and solve for a Hopf bifurcation

        Parameters
        ----------
        p : bvec.BlockVector
            The parameter vector consisting of dynamical model properties +
            complex amplitude properties (size 2)
        """
        ## Record current state to h5 file
        # Record the current parameter set
        h5utils.append_block_vector_to_group(
            self.f['parameters'], p)

        hopf_state = self.redu_grad.hist_state[-1]
        h5utils.append_block_vector_to_group(
            self.f['hopf_state'], hopf_state)

        self.f['hopf_newton_num_iter'].resize(self.f['hopf_newton_num_iter'].size+1, axis=0)
        self.f['hopf_newton_num_iter'][-1] = info['num_iter']

        self.f['hopf_newton_status'].resize(self.f['hopf_newton_status'].size+1, axis=0)
        self.f['hopf_newton_status'][-1] = info['status']

        self.f['hopf_newton_rel_err'].resize(self.f['hopf_newton_rel_err'].size+1, axis=0)
        self.f['hopf_newton_rel_err'][-1] = info['rel_errs'][-1]

        self.f['hopf_newton_abs_err'].resize(self.f['hopf_newton_abs_err'].size+1, axis=0)
        self.f['hopf_newton_abs_err'][-1] = info['abs_errs'][-1]

    def set_props(self, p):
        # Set properties and complex amplitude of the ReducedGradient
        # This has to convert the monolithic input parameters to the block
        # format of the ReducedGradient object
        p_hopf = p[:-2]
        _p_hopf = self.redu_grad.props.copy()
        _p_hopf.set_vec(p_hopf)

        p_camp = p[-2:]
        _p_camp = self.redu_grad.camp.copy()
        _p_camp.set_vec(p_camp)

        # After setting `self.redu_grad` props, the Hopf system should be solved
        hopf_state, info = self.redu_grad.set_props(_p_hopf)
        self.redu_grad.set_camp(_p_camp)

        self._update_h5(hopf_state, info, bvec.concatenate_vec([_p_hopf, _p_camp]))

    def grad(self, p):
        self.set_props(p)

        # Solve the objective function value
        g = self.redu_grad.assem_g()

        # Solve the gradient of the objective function
        _dg_dp = bvec.concatenate_vec([self.redu_grad.assem_dg_dprops(), self.redu_grad.assem_dg_dcamp()])

        # Record the current objective function and gradient
        h5utils.append_block_vector_to_group(self.f['grad'], _dg_dp)
        self.f['objective'].resize(self.f['objective'].size+1, axis=0)
        self.f['objective'][-1] = g
        return g, _dg_dp.to_mono_ndarray()
