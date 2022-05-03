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

def run_lsa(f, res_dyn, emod_cov, emod_bod):
    _forms = res_dyn.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])

    props = res_dyn.props

    dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
    dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
    props['emod'].array[dofs_cov] = emod_cov
    props['emod'].array[dofs_bod] = emod_bod
    res_dyn.set_props(props)

    for group_name in ['eigvec_real', 'eigvec_imag']:
        h5utils.create_resizable_block_vector_group(
            f.require_group(group_name), res_dyn.state.labels, res_dyn.state.bshape
        )

    eigs_info = [libhopf.max_real_omega(res_dyn, psub) for psub in np.arange(0, 1300, 100)*10]

    omegas_real = [eiginfo[0].real for eiginfo in eigs_info]
    omegas_imag = [eiginfo[0].imag for eiginfo in eigs_info]
    eigvecs_real = [eiginfo[1] for eiginfo in eigs_info]
    eigvecs_imag = [eiginfo[2] for eiginfo in eigs_info]

    f['omega_real'] = np.array(omegas_real)
    f['omega_imag'] = np.array(omegas_imag)
    for group_name, eigvecs in zip(['eigvec_real', 'eigvec_imag'], [eigvecs_real, eigvecs_imag]):
        for eigvec in eigvecs:
            h5utils.append_block_vector_to_group(f[group_name], eigvec)

if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')
    res_dyn, dres_dyn = setup.setup_models(mesh_path)
    # res_hopf = libhopf.HopfModel(res_dyn, dres_dyn)

    # _forms = res_dyn.solid.forms
    # celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])

    EMODS = np.arange(2.5, 12.5+2.5, 2.5) * 1e3*10
    # print(EMODS)

    for emod_cov, emod_bod in itertools.product(EMODS, EMODS):
        fname = f'LSA_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        fpath = f'out/stress_test/{fname}.h5'
        with h5py.File(fpath, mode='w') as f:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                run_lsa(f, res_dyn, emod_cov, emod_bod)

