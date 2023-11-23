"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
import numbers
import operator
import pytest

import numpy as np
from blockarray import linalg as bla

from libhopf import functional as libfuncs
from libhopf.setup import load_hopf_model, set_default_props

from libtest import taylor_convergence


# pylint: disable=redefined-outer-name
# pylint: disable=no-member

EBODY = 5e3 * 10
ECOV = 5e3 * 10
PSUB = 450 * 10

def _test_taylor(*args, **kwargs):
    alphas, errs, magnitudes, conv_rates = taylor_convergence(*args, **kwargs)
    assert pass_taylor_convergence_test(errs, magnitudes, conv_rates)

def pass_taylor_convergence_test(errs, magnitudes, conv_rates, relerr_tol=1e-5):
    """
    Return whether a set of errors passes the Taylor convergence test
    """
    is_taylor2 = np.all(np.isclose(conv_rates, 2, rtol=0.1, atol=0.1))
    with np.errstate(invalid='ignore'):
        rel_errs = np.where(errs == 0, 0, errs/magnitudes)
    is_relerr = np.all(np.isclose(rel_errs, 0, atol=relerr_tol))
    if is_taylor2:
        return True
    elif is_relerr:
        return True
    else:
        return False

@pytest.fixture(
    params=[
        'M5_CB_GA3_CL0.50'
    ]
)
def mesh_path(request):
    mesh_name = request.param
    mesh_dir = './mesh'
    mesh_path = path.join(mesh_dir, f'{mesh_name}.msh')

    return mesh_path

@pytest.fixture()
def setup_hopf_model(mesh_path):
    """
    Return a hopf model
    """
    hopf, *_ = load_hopf_model(mesh_path, sep_method='smoothmin', sep_vert_label='separation')
    return hopf

@pytest.fixture(
    params=[
        libfuncs.OnsetPressureFunctional,
        libfuncs.GlottalWidthErrorFunctional,
        libfuncs.StrainEnergyFunctional
    ]
)
def setup_func(setup_hopf_model, request):
    """
    Return a hopf model functional to test
    """
    FunctionalClass = request.param
    return FunctionalClass(setup_hopf_model)

@pytest.fixture()
def setup_linearization(setup_func, setup_hopf_model):
    """
    Return a linearzation point and direction
    """
    func = setup_func
    hopf = setup_hopf_model

    state0 = hopf.state.copy()
    state0['u'] = np.random.rand(*state0['u'].shape)
    hopf.apply_dirichlet_bvec(state0)
    dstate = state0.copy()
    dstate[:] = 0
    dstate['u'] = 1.0e-5
    dstate['psub'] = 1.0
    hopf.apply_dirichlet_bvec(dstate)

    props0 = hopf.prop.copy()
    set_default_props(props0, hopf.res.solid.residual.mesh())
    dprop = props0.copy()
    dprop[:] = 0
    dprop['emod'] = 1.0


    return (state0, props0), (dstate, dprop)

def set_linearization(setup_func, state, prop):
    """
    Set the linearization point for a functional
    """
    func = setup_func
    func.set_state(state)
    func.set_prop(prop)

def test_assem_dg_dstate(setup_func, setup_linearization):
    """Test the functional `state` derivative"""
    func = setup_func
    (state, prop), (dstate, dprop) = setup_linearization
    set_linearization(func, state, prop)

    def res(x):
        func.set_state(x)
        return func.assem_g()

    def jac(x, dx):
        func.set_state(x)
        return bla.dot(func.assem_dg_dstate(), dx)

    _test_taylor(state, dstate, res, jac, norm=lambda x: (x**2)**0.5)

def test_assem_dg_dprops(setup_func, setup_linearization):
    """Test the functional `prop` derivative"""
    func = setup_func
    (state, prop), (dstate, dprop) = setup_linearization
    set_linearization(func, state, prop)

    def res(x):
        func.set_prop(x)
        return func.assem_g()

    def jac(x, dx):
        func.set_prop(x)
        return bla.dot(func.assem_dg_dprops(), dx)

    _test_taylor(prop, dprop, res, jac, norm=lambda x: (x**2)**0.5)


@pytest.fixture(
    params=[
        (libfuncs.OnsetPressureFunctional, libfuncs.OnsetPressureFunctional),
        (libfuncs.GlottalWidthErrorFunctional, libfuncs.OnsetPressureFunctional),
        (libfuncs.OnsetPressureFunctional, 5)
    ]
)
def setup_funcs(setup_hopf_model, request):
    """Return functional pairs"""
    def create_func(FuncClass):
        if isinstance(FuncClass, type):
            return FuncClass(setup_hopf_model)
        else:
            value = FuncClass
            return value
    return [create_func(x) for x in request.param]

@pytest.fixture(
    params=[
        operator.add, operator.mul, operator.truediv,
        # operator.pow
    ]
)
def setup_binary_op(request):
    """Return a binary operation"""
    return request.param

def test_binary_op(setup_binary_op, setup_funcs):
    """Test binary operations on functionals"""
    # Compare the result of applying an operation on functionals with the
    # correct result

    # The correct functional value should be the operation applied on the
    # individual functional values
    g_correct = setup_binary_op(*[
        func if isinstance(func, numbers.Number) else func.assem_g()
        for func in setup_funcs
    ])

    # The tested functional is the operation applied on the functional objects
    # to create a DerivedFunctional
    g_op = setup_binary_op(*setup_funcs).assem_g()

    print(g_op, g_correct)
    assert g_op == g_correct
