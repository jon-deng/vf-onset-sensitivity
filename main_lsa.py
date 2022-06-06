"""
Plot results of a linear stability analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from setuptools import setup
import dolfin as dfn

import libsetup
import libhopf

dfn.set_log_level(50)
if __name__ == '__main__':
    # mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_name = 'M5_CB_GA0'

    mesh_path = f'mesh/{mesh_name}.xml'
    _, res_xml, _ = libsetup.load_hopf(
        mesh_path,
        sep_method='smoothmin',
        sep_vert_label='separation-inf'
    )
    mf_facet_xml = res_xml.solid.forms['mesh.facet_function']
    mf_cell_xml = res_xml.solid.forms['mesh.cell_function']
    ds_xml = res_xml.solid.forms['measure.ds']

    ds = ds_xml
    res = res_xml

    # Compare mesh functions between the two mesh formats
    # mesh_path = f'mesh/{mesh_name}.msh'
    # _, res_msh, _ = libsetup.load_hopf(
    #     mesh_path,
    #     sep_method='smoothmin',
    #     sep_vert_label='separation-inf'
    # )

    # mf_facet_msh = res_msh.solid.forms['mesh.facet_function']
    # mf_facet_msh.array()[mf_facet_msh.where_equal(2**64-1)] = 0
    # mf_cell_msh = res_msh.solid.forms['mesh.cell_function']
    # ds_msh = res_msh.solid.forms['measure.ds']
    # ds = ds_msh
    # res = res_msh

    breakpoint()

    # mesh_name = 'BC-dcov5.00e-02-cl1.00'
    # # mesh_name = 'M5_CB_GA0'
    # mesh_path = f'mesh/{mesh_name}.xml'

    # _, res, _ = libsetup.load_hopf(
    #     mesh_path,
    #     sep_method='smoothmin',
    #     sep_vert_label='separation-inf'
    # )
    # res, dres = libsetup.setup_models(mesh_path)

    psubs = np.arange(0, 1500, 100)*10

    least_stable_modes_info = [
        libhopf.solve_least_stable_mode(res, psub)
        for psub in psubs
    ]
    breakpoint()

    print(dfn.assemble(1*ds(3)))
    print(dfn.assemble(1*ds(4)))
    print(dfn.assemble(1*ds))

    omegas = np.array([mode_info[0] for mode_info in least_stable_modes_info])

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(psubs, omegas.real)
    axs[1].plot(psubs, omegas.imag)

    axs[1].set_xlabel("$\omega_{real}$")
    axs[1].set_xlabel("$\omega_{imag}$")
    axs[1].set_xlabel("$p_{sub}$ [Pa]")

    fig.tight_layout()
    fig.savefig('fig/lsa.png')
