"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
import sys
from os import path
import cProfile
import warnings

sys.path.append('../')
import libhopf
from setup import setup_hopf_state

# pylint: disable=redefined-outer-name
# pylint: disable=no-member

def test_solve_hopf_newton(hopf, xhopf0):
    xhopf, info = libhopf.solve_hopf_newton(hopf, xhopf0)

if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('../mesh', mesh_name+'.xml')

    hopf, xhopf, props0 = setup_hopf_state(mesh_path)

    hopf.set_state(xhopf)
    hopf.set_props(props0)

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=UserWarning)

        # test_solve_hopf_newton(hopf, xhopf)
        cProfile.run('test_solve_hopf_newton(hopf, xhopf)')

