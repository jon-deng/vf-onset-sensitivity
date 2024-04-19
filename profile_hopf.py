"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""

from os import path
import cProfile
import pstats
import warnings
import timeit

from blockarray import subops as gops

# import libhopf
from libsetup import load_hopf_model, set_default_props

# pylint: disable=redefined-outer-name
# pylint: disable=no-member


def solve_hopf_newton_step(hopf, xhopf0):
    dres_dstate = hopf.assem_dres_dstate()
    res = hopf.assem_res()

    hopf.apply_dirichlet_bvec(res)
    hopf.apply_dirichlet_bmat(dres_dstate)

    _dres_dstate = dres_dstate.to_mono_petsc()
    _res = res.to_mono_petsc()

    _dxhopf = _dres_dstate.getVecRight()
    _dxhopf, _ = gops.solve_petsc_lu(_dres_dstate, _res, out=_dxhopf)


if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name + '.msh')
    hopf, res, dres = load_hopf_model(
        mesh_path, sep_method='smoothmin', sep_vert_label='separation-inf'
    )

    xhopf = hopf.state.copy()
    prop = hopf.prop.copy()
    set_default_props(prop, res.solid.forms['mesh.mesh'])

    # hopf.set_state(xhopf)
    hopf.set_prop(prop)
    hopf.assem_res()
    hopf.assem_dres_dstate()

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=UserWarning)

        cProfile.run('solve_hopf_newton_step(hopf, xhopf)', 'profile_hopf.prof')

        # N = 10
        # dur = timeit.timeit(lambda : solve_hopf_newton_step(hopf, xhopf), number=N)
        # print(f"{dur/N:.2f} s")

        with open('profile_hopf.stats', 'w') as output_stream:
            p = pstats.Stats('profile_hopf.prof', stream=output_stream)
            p.sort_stats('cumtime')
            p.print_stats()

        # the encoding is needed for pandas; otherwise it returns an error decoding the file
        # with open('profile_hopf.stats', 'rb') as f:
        #     df = pandas.read_csv(f, delim_whitespace=True, skiprows=6)
        # df.to_csv('profile_hopf.csv')
