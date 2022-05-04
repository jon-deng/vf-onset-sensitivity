"""
This script runs a linear stability analysis (LSA) for all the stress test cases

The stress test cases consist of all combinations of body and cover moduli where
the moduli range from 2.5 to 10 in steps of 2.5 (kPa).
"""

import os.path as path
import itertools
import warnings

import numpy as np
import h5py
from femvf import meshutils
from blockarray import h5utils

import setup
import libhopf

# pylint: disable=redefined-outer-name

# Range of psub to test for Hopf bifurcation
PSUBS = np.arange(100, 1300, 100)*10

def set_props(props, celllabel_to_dofs, emod_cov, emod_bod):
    # Set any constant properties
    props = setup.set_constant_props(props, celllabel_to_dofs, res_dyn)

    # Set cover and body layer properties
    dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
    dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
    props['emod'].array[dofs_cov] = emod_cov
    props['emod'].array[dofs_bod] = emod_bod
    return props

def run_lsa(f, res_dyn, emod_cov, emod_bod):
    # Get the cover/body layer DOFs
    _forms = res_dyn.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])
    props = set_props(res_dyn.props, celllabel_to_dofs, emod_cov, emod_bod)
    res_dyn.set_props(props)

    for group_name in ['eigvec_real', 'eigvec_imag', 'fixedpoint']:
        h5utils.create_resizable_block_vector_group(
            f.require_group(group_name), res_dyn.state.labels, res_dyn.state.bshape
        )

    eigs_info = [libhopf.max_real_omega(res_dyn, psub) for psub in PSUBS]

    omegas_real = [eiginfo[0].real for eiginfo in eigs_info]
    omegas_imag = [eiginfo[0].imag for eiginfo in eigs_info]
    eigvecs_real = [eiginfo[1] for eiginfo in eigs_info]
    eigvecs_imag = [eiginfo[2] for eiginfo in eigs_info]
    xfps = [eiginfo[3] for eiginfo in eigs_info]

    f['omega_real'] = np.array(omegas_real)
    f['omega_imag'] = np.array(omegas_imag)
    for group_name, eigvecs in zip(
            ['eigvec_real', 'eigvec_imag', 'fixedpoint'],
            [eigvecs_real, eigvecs_imag, xfps]
        ):
        for eigvec in eigvecs:
            h5utils.append_block_vector_to_group(f[group_name], eigvec)

def run_solve_hopf(f, res_hopf, emod_cov, emod_bod):
    # Set the cover/body layer properties
    _forms = res_hopf.res.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])
    props = set_props(res_hopf.props, celllabel_to_dofs, emod_cov, emod_bod)
    res_hopf.set_props(props)

    # Read the max real eigenvalue information from the LSA to determine if Hopf
    # bifurcations occur and a good starting point
    lsa_fname = f'LSA_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
    lsa_fpath = f'out/stress_test/{lsa_fname}.h5'

    with h5py.File(lsa_fpath, mode='r') as f:
        omegas_real = f['omega_real'][:]

    is_hopf_bif = [(w2 > 0 and w1 <=0) for w1, w2 in zip(omegas_real[:-1], omegas_real[1:])]
    # breakpoint()
    if is_hopf_bif.count(True) == 0:
        print(f"Case {lsa_fname} has no Hopf bifurcations")
        print(f"Real eigenvalue components are {omegas_real}")
    else:
        idx_hopf = is_hopf_bif.index(True)
        print(f"Case {lsa_fname} has a Hopf bifurcation between {PSUBS[idx_hopf]} {PSUBS[idx_hopf+1]} dPa")
        print(f"Real eigenvalue components are {omegas_real}")

        xhopf_0 = libhopf.gen_hopf_initial_guess(
            res_hopf,
            ([PSUBS[idx_hopf]], [PSUBS[idx_hopf+1]]),
            ([omegas_real[idx_hopf]], [omegas_real[idx_hopf+1]])
        )
        xhopf_n, info = libhopf.solve_hopf_newton(res_hopf, xhopf_0)

        fname = f'Hopf_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        fpath = f'out/{fname}.h5'
        with h5py.File(fpath, mode='w') as f:
            h5utils.create_resizable_block_vector_group(f, xhopf_n.labels, xhopf_n.bshape)
            h5utils.append_block_vector_to_group(f, xhopf_n)


if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')
    res_dyn, dres_dyn = setup.setup_models(mesh_path)
    res_hopf = libhopf.HopfModel(res_dyn, dres_dyn)

    # _forms = res_dyn.solid.forms
    # celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])

    EMODS = np.arange(2.5, 12.5+2.5, 2.5) * 1e3*10

    # EMODS = np.arange(5.0, 12.5+2.5, 2.5) * 1e3*10
    # print(EMODS)

    for ii, emod_bod in enumerate(EMODS):
        for emod_cov in EMODS[:ii+1]:
            fname = f'LSA_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
            fpath = f'out/stress_test/{fname}.h5'

            if not path.isfile(fpath):
                with h5py.File(fpath, mode='w') as f:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        run_lsa(f, res_dyn, emod_cov, emod_bod)
            else:
                print(f"File {fpath} already exists")

    for ii, emod_bod in enumerate(EMODS):
        for emod_cov in EMODS[:ii+1]:
            fname = f'Hopf_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
            fpath = f'out/stress_test/{fname}.h5'

            with h5py.File('out/test.h5', mode='w') as f:
                run_solve_hopf(f, res_hopf, emod_cov, emod_bod)
