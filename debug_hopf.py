"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
import sys
from os import path
import petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

from femvf.dynamicalmodels import solid as sldm, fluid as fldm
from femvf.load import load_dynamical_fsi_model
from femvf.meshutils import process_meshlabel_to_dofs
import nonlineq as nleq

import blocktensor.linalg as bla
from blocktensor import vec as bvec

from hopf import make_hopf_system
from main_hopf import set_properties

petsc4py.init(sys.argv)

if __name__ == '__main__':
    ## Load 3 residual functions needed to model the Hopf system
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

    ## Set model properties
    # get the scalar DOFs associated with the cover/body layers
    mesh = res.solid.forms['mesh.mesh']
    cell_func = res.solid.forms['mesh.cell_function']
    func_space = res.solid.forms['fspace.scalar']
    cell_label_to_id = res.solid.forms['mesh.cell_label_to_id']
    region_to_dofs = process_meshlabel_to_dofs(mesh, cell_func, func_space, cell_label_to_id)

    props = res.properties.copy()
    y_mid = set_properties(props, region_to_dofs, res)
    
    for model in (res, dres_u, dres_ut):
        model.ymid = y_mid

    ## Initialize the Hopf system
    xhopf, hopf_res, hopf_jac, apply_dirichlet_vec, idx_dirichlet, labels = make_hopf_system(res, dres_u, dres_ut, props)
    state_labels, mode_real_labels, mode_imag_labels, psub_labels, omega_labels = labels

    # Set the starting point of any iterative solutions
    xhopf_n = xhopf.copy()
    xhopf_n['psub'].array[:] = 2000.0*10
    xhopf_n['omega'].array[:] = 1.0

    idx = [1]
    res_n = hopf_res(xhopf_n)[state_labels][idx]
    jac_n = hopf_jac(xhopf_n)[state_labels, state_labels][idx, idx]

    idx = ['u', 'v']
    res_n = hopf_res(xhopf_n)[idx]
    jac_n = hopf_jac(xhopf_n)[idx, idx]

    # BUG: This doesn't work but I have no idea why!
    _res_n = res_n.to_petsc()
    _jac_n = jac_n.to_petsc()
    _dx_n = _jac_n.getVecRight()

    ksp = PETSc.KSP().create()
    ksp.setType(ksp.Type.PREONLY)
    

    pc = ksp.getPC()
    pc.setType(pc.Type.LU)

    ksp.setOperators(_jac_n)
    ksp.setUp()
    ksp.solve(_res_n, _dx_n)
