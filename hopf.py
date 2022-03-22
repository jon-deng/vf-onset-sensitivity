"""
Contains code to create the Hopf system
"""
import itertools
from functools import reduce
import numpy as np

import blocktensor.genericops as gops
import blocktensor.linalg as bla
from blocktensor import vec as bvec
from blocktensor import mat as bmat

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

    def assign_model_state(x):
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

    HOPF_LABELS = tuple(reduce(lambda a, b: a+b, labels))

    EBVEC = x[state_labels].copy()
    EBVEC['u'][0] = 1.0
    def hopf_res(x):
        """Return the Hopf system residual"""
        # Set the model state and subglottal pressure (bifurcation parameter)
        assign_model_state(x)

        res_state = res.assem_res()

        omega = x['omega'][0]
        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        res_mode_real = dres_u.assem_res() - omega*dres_ut.assem_res()

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_real_labels])
        dres_ut.set_dstate(x[mode_imag_labels])
        res_mode_imag = dres_u.assem_res() + omega*dres_ut.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_imag_labels])

        ret_bvec =  bvec.concatenate_vec(
            [res_state, res_mode_real, res_mode_imag, res_psub, res_omega], labels=HOPF_LABELS)
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
        assign_model_state(x)

        # build the Jacobian row by row
        dres_dstate = res.assem_dres_dstate()
        dres_dstatet = res.assem_dres_dstatet()

        # Using copys of dres_dstate is important as different dres_dstate locations
        # will rquire different dirichlet settings on their rows
        jac_1 = [
            dres_dstate.copy(),
            NULL_MAT_STATE_STATE, 
            NULL_MAT_STATE_STATE, 
            res.assem_dres_dcontrol()[:, ['psub']], 
            NULL_MAT_STATE_SCALAR]

        omega = x['omega'][0]
        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        jac_2 = [
            dres_u.assem_dres_dstate() - omega*dres_ut.assem_dres_dstate(), 
            -omega*dres_dstatet.copy(), 
            dres_dstate.copy(),  
            dres_u.assem_dres_dcontrol()[:, ['psub']] - omega*dres_ut.assem_dres_dcontrol()[:, ['psub']], 
            bvec.convert_bvec_to_petsc_colbmat(dres_ut.assem_res())]

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_real_labels])
        dres_ut.set_dstate(x[mode_imag_labels])
        jac_3 = [
            dres_u.assem_dres_dstate() + omega*dres_ut.assem_dres_dstate(), 
            dres_dstate.copy(), 
            omega*dres_dstatet.copy(), 
            dres_u.assem_dres_dcontrol()[:, ['psub']] + omega*dres_ut.assem_dres_dcontrol()[:, ['psub']], 
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
        ret_labels = (HOPF_LABELS, HOPF_LABELS)
        ret_bmat = bmat.concatenate_mat(ret_mats, ret_labels)

        # Apply dirichlet BC by zeroing appropriate matrix rows
        row_labels = ['u', 'v', 'u_mode_real', 'v_mode_real', 'u_mode_imag', 'v_mode_imag']
        col_labels = HOPF_LABELS
        for row, col in itertools.product(row_labels, col_labels):
            mat = ret_bmat[row, col]
            if row == col:
                mat.zeroRows(IDX_DIRICHLET, diag=1.0)
            else:
                mat.zeroRows(IDX_DIRICHLET, diag=0.0)

            # Set 1 on the diagonal (zero's one block twice but it shouldn't matter much)
            # ret_bmat[label, label].zeroRows(IDX_DIRICHLET, diag=1)

        return ret_bmat

    
    return x, hopf_res, hopf_jac, apply_dirichlet_vec, IDX_DIRICHLET, labels
