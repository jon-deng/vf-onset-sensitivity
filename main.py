"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
from os import path
from petsc4py import PETSc

from femvf.dynamicalmodels import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model

import blocklinalg.genericops as gops
import blocklinalg.linalg as bla
from blocklinalg import vec as bvec
from blocklinalg import mat as bmat

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

    breakpoint()
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

    # Create the input vector for the system
    x, state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = hopf_state(res)
    labels = (state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels)

    EBVEC = x[state_labels].copy()
    EBVEC['u'][0] = 1.0
    def hopf_res(x):
        # Set the model state and subglottal pressure (bifurcation parameter)
        for model in (res, dres_u, dres_ut):
            model.set_state(x[state_labels])

            _control = model.control.copy()
            _control['psub'][0] = x['psub'][0]
            model.set_control(_control)

        res_state = res.assem_res()

        # Set appropriate linearization directions
        omega = x['omega'][0]
        dres_u.set_dstate(x[mode_real_labels])
        dres_ut.set_dstate(x[mode_imag_labels])
        res_mode_real = dres_u.assem_res() + omega*dres_ut.assem_res()

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        res_mode_imag = dres_u.assem_res() - omega*dres_ut.assem_res()

        res_psub = x[['psub']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_real_labels])

        res_omega = x[['omega']].copy()
        res_psub['psub'][0] = bla.dot(EBVEC, x[mode_imag_labels])

        return bvec.concatenate_vec([res_state, res_mode_real, res_mode_imag, res_psub, res_omega])


    # Make null matrix constants
    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]] 
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_STATE = bmat.BlockMat(mats, (x[state_labels].bshape[0], x[state_labels].bshape[0]))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in [1]] 
        for row_size in x[state_labels].bshape[0]]
    NULL_MAT_STATE_SCALAR = bmat.BlockMat(mats, (x[state_labels].bshape[0], ('1',)))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in x[state_labels].bshape[0]] 
        for row_size in [1]]
    NULL_MAT_SCALAR_STATE = bmat.BlockMat(mats, (('1',), x[state_labels].bshape[0]))

    mats = [
        [bmat.zero_mat(row_size, col_size) for col_size in [1]] 
        for row_size in [1]]
    NULL_MAT_SCALAR_SCALAR = bmat.BlockMat(mats, (('1',), ('1',)))

    def hopf_jac(x):
        # Set the model state and subglottal pressure (bifurcation parameter)
        for model in (res, dres_u, dres_ut):
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
            bvec.convert_bvec_to_petsc_colbmat(dres_ut.assem_res())]

        # Set appropriate linearization directions
        dres_u.set_dstate(x[mode_imag_labels])
        dres_ut.set_dstate(x[mode_real_labels])
        jac_3 = [
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

        mats = [jac_1, jac_2, jac_3, jac_4, jac_5]

        return bmat.concatenate_mat(mats)

    return x, hopf_res, hopf_jac, labels

if __name__ == '__main__':
    print("heyhey yoyo")

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

    props = res.properties.copy()
    x, hopf_res, hopf_jac, labels = make_hopf_system(res, dres_u, dres_ut, props)

    g = hopf_res(x)
    dgdx = hopf_jac(x)
