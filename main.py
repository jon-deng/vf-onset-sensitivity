"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
from petsc4py import PETSc

import blocklinalg.genericops as gops
import blocklinalg.linalg as bla
from blocklinalg.vec import BlockVec, concatenate_vec
from blocklinalg.mat import concatenate_mat, zero_mat, BlockMat

def hopf_state(res):
    """
    Return the state vector for a Hopf system
    """
    X_state = res.state.copy()

    _mode_real_vecs = res.state.copy().array
    _mode_real_labels = tuple(tuple([label+'_mode_real' for label in X_state.labels[0]]))
    X_mode_real = BlockVec(_mode_real_vecs, _model_real_labels)

    _mode_imag_vecs = res.state.copy().array
    _mode_imag_labels = tuple(tuple([label+'_mode_imag' for label in X_state.labels[0]]))
    X_mode_imag = BlockVec(_mode_imag_vecs, _mode_imag_labels)

    X_psub = res.control[['psub']].copy()

    _omega = X_psub['psub'].copy()
    _omega_vecs = tuple([_omega])
    _omega_labels = tuple(tuple(['omega']))
    X_omega = BlockVec(_omega_vec, _omega_labels)

    ret = concatenate_vec([X_state, X_mode_real, X_mode_imag, X_psub, X_omega])
    state_labels = X_state.labels[0]
    mode_real_labels = X_mode_real.labels[0]
    mode_imag_labels = X_mode_imag.labels[0]
    psub_labels = X_psub.labels[0]
    omega_labels = X_omega.labels[0]
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
    for model in (res, res_u, res_ut):
        model.set_properties(props)

    # Create the input vector for the system
    x, state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = hopf_state(res)
    labels = (state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels)

    EBVEC = x[state_labels].copy()
    EBVEC['u'][0] = 1.0
    def hopf_res(x):
        # Set the model state and subglottal pressure (bifurcation parameter)
        for model in (res, res_u, res_ut):
            model.set_state(x[state_labels])

            _control = model.control.copy()
            _control['psub'][0] = x['psub'][0]
            model.set_control(_control)

        res_state = res.assem_res()

        # Set appropriate linearization directions
        omega = x['omega'][0]
        dres_u.set_dstate(x[mode_real_labels])
        dres_ut.set_dstate(x[mode_imag_labels])
        res_mode_real = dres_u.assem_dres_u() + omega*dres_ut.assem_res()

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        res_mode_imag = dres_u.assem_dres_u() - omega*dres_ut.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_psub['omega'][0] = bla.dot(EBVEC, x[mode_imag_labels])

        return concatenate_vec([res_state, res_mode_real, res_mode_imag, res_psub, res_omega])


    # Make null matrix constants
    mats = [
        [zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]] 
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_STATE = BlockMat(mats, (x[state_labels].bshape[0], x[state_labels].bshape[0]))

    mats = [
        [zero_mat(row_size, col_size) for col_size in [1]] 
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_SCALAR = BlockMat(mats, (('1'), x[state_labels].bshape[0]))

    mats = [
        [zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]] 
        for row_size in [1]]
    NULL_MAT_OMEGA_STATE = BlockMat(mats, (('omega',), x[state_labels].bshape[0]))
    NULL_MAT_PSUB_STATE = BlockMat(mats, (('psub',), x[state_labels].bshape[0]))

    def hopf_jac(x):
        # Set the model state and subglottal pressure (bifurcation parameter)
        for model in (res, res_u, res_ut):
            model.set_state(x[state_labels])

            _control = model.control.copy()
            _control['psub'][0] = x['psub'][0]
            model.set_control(_control)

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
        jac_2 = [
            dres_u.assem_dres_dstate() + omega*dres_ut.assem_dres_dstate(), 
            dres_dstate, 
            omega*dres_dstatet, 
            dres_u.assem_dres_dcontrol()[:, ['psub']] + omega*dres_ut.assem_dres_dcontrol()[:, ['psub']], 
            dres_ut.assem_res()]
        # TODO: the last entry of the row ((dres_ut.assem_res()) must be converted to a column matrix

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        jac_3 = [
            dres_u.assem_dres_dstate() - omega*dres_ut.assem_dres_dstate(), 
            -omega*dres_dstatet, 
            dres_dstate, 
            dres_u.assem_dres_dcontrol()[:, ['psub']] - omega*dres_ut.assem_dres_dcontrol()[:, ['psub']], 
            dres_ut.assem_res()]
        # TODO: the last entry of the row (dres_ut.assem_res()) must be converted to a column matrix

        jac_4 = [
            NULL_MAT_SCALAR_SCALAR,
            EBVEC, # TODO: have to convert this to appropriate, row matrix
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        jac_5 = [
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR,
            EBVEC, # TODO: have to convert this to appropriate, row matrix
            NULL_MAT_SCALAR_SCALAR,
            NULL_MAT_SCALAR_SCALAR
            ]

        mats = [jac_1, jac_2, jac_3, jac_4, jac_5]

        return concatenate_mat(mats)

    return x, hopf_res, hopf_jac, labels