"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
import numbers
import operator
import numpy as np

from blocktensor import linalg as bla

import libfunctionals as libfuncs
from setup import setup_hopf_state
from test_hopf import _test_taylor


# pylint: disable=redefined-outer-name
# pylint: disable=no-member

EBODY = 5e3 * 10
ECOV = 5e3 * 10
PSUB = 450 * 10

def test_assem_dg_dstate(func, state0, dstate):
    def res(state):
        func.set_state(state)
        return func.assem_g()

    def jac(state):
        func.set_state(state)
        return func.assem_dg_dstate()

    _test_taylor(state0, dstate, res, jac, action=bla.dot, norm=lambda x: (x**2)**0.5)

def test_assem_dg_dprops(func, props0, dprops):
    def res(props):
        func.set_props(props)
        return func.assem_g()

    def jac(props):
        func.set_props(props)
        return func.assem_dg_dprops()

    _test_taylor(props0, dprops, res, jac, action=bla.dot, norm=lambda x: (x**2)**0.5)

def test_assem_dg_dcamp(func, camp0, dcamp):
    def res(camp):
        func.set_camp(camp)
        return func.assem_g()

    def jac(camp):
        func.set_camp(camp)
        return func.assem_dg_dcamp()

    _test_taylor(camp0, dcamp, res, jac, action=bla.dot, norm=lambda x: (x**2)**0.5)

def test_op(op, *funcs):
    # Compare the result of applying an operation on functionals with the
    # correct result

    # The correct functional value should be the operation applied on the
    # individual functional values
    g_correct = op(*[
        func if isinstance(func, numbers.Number) else func.assem_g()
        for func in funcs])

    # The tested functional is the operation applied on the functional objects
    # to create a DerivedFunctional
    g_op = op(*funcs).assem_g()

    print(g_op, g_correct)

if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    hopf, xhopf, props0 = setup_hopf_state(mesh_path)
    funca = libfuncs.OnsetPressureFunctional(hopf)
    funcb = libfuncs.GlottalWidthErrorFunctional(hopf)
    # print(xhopf['psub'][:])

    func = funcb
    state0 = xhopf.copy()
    dstate = state0.copy()
    dstate.set(0)
    dstate['u'].array[:] = 1.0e-5
    # dstate.set(1.0e-5)
    hopf.apply_dirichlet_bvec(dstate)
    test_assem_dg_dstate(func, state0, dstate)

    dprops = props0.copy()
    dprops.set(0)
    dprops['emod'].array[:] = 1.0
    test_assem_dg_dprops(func, props0, dprops)

    camp0 = func.camp.copy()
    dcamp = camp0.copy()
    dcamp['amp'].set(1e-4)
    dcamp['phase'].set(np.pi*1e-5)
    test_assem_dg_dcamp(func, camp0, dcamp)

    test_op(operator.add, func, func)

    test_op(operator.mul, func, 5.0)
    test_op(operator.mul, 5.0, func)

    test_op(operator.truediv, func, func)
    test_op(operator.truediv, func, 5.0)

    test_op(operator.pow, func, 2)
