"""
This code runs transient simulations that should correspond to the simulations
in main_hopf.py
"""
import os
from os.path import isfile, isdir
import argparse
from multiprocessing import Pool
import numpy as np

from femvf import forward, load, statefile as sf
from femvf.models import solid as smd, fluid as fmd
from femvf.meshutils import process_meshlabel_to_dofs
from blocktensor import linalg

from lib_main_transient import case_config

# from main_hopf import set_properties

parser = argparse.ArgumentParser()
parser.add_argument('--zeta', type=float, default=0.0)
parser.add_argument('--r_sep', type=float, default=1.2)
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()

# PSUB = 800 * 10

# These parameters cover a broad range to try and gauge when onset happens
PSUBS = np.arange(200, 1000, 100) * 10

ETA_VISC = 5
ECOV = 5e3*10
EBODY = 15e3*10

DT = 5e-5
T_TOTAL = 0.5

mesh_name = 'BC-dcov5.00e-02-cl1.00'
mesh_path = f'mesh/{mesh_name}.xml'
model = load.load_fsi_model(mesh_path, None, SolidType=smd.KelvinVoigt, FluidType=fmd.Bernoulli, coupling='explicit')

# Get DOFs associated with layer regions
mesh = model.solid.forms['mesh.mesh']
cell_func = model.solid.forms['mesh.cell_function']
func_space = model.solid.forms['fspace.scalar']
cell_label_to_id = model.solid.forms['mesh.cell_label_to_id']
region_to_dofs = process_meshlabel_to_dofs(mesh, cell_func, func_space, cell_label_to_id)

## Set model properties to nominal values
props = model.get_properties_vec()

# constant ones
props['rho'][:] = 1.0
props['nu'][:] = 0.45
props['eta'][:] = 5.0

# geometric properties related to the symmetry/contact planes
y_gap = 0.1
y_contact_offset = 1/10 * y_gap
props['y_midline'][:] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap/2
props['ycontact'][:] = props['y_midline'] - y_contact_offset 
props['kcontact'][:] = 1e16

# Fluid properties
ZETA = 1e-4
R_SEP = 1.0
props['r_sep'][:] = R_SEP
props['ygap_lb'][:] = y_contact_offset
props['zeta_lb'][:] = 1e-6
props['zeta_amin'][:] = ZETA
props['zeta_sep'][:] = ZETA
props['zeta_ainv'][:] = ZETA

# Create the output directory
out_dir = f'out/zeta{ZETA:.2e}_rsep{R_SEP:.1f}'
if not isdir(out_dir):
    os.makedirs(out_dir)

dofs_cover = region_to_dofs['cover']
dofs_body = region_to_dofs['body']

# Set the constant body layer modulus
props['emod'][dofs_cover] = ECOV
props['emod'][dofs_body] = EBODY


def run(psub):
    # Set the initial state/properties/control needed to integrate in time
    ini_state = model.get_state_vec()
    ini_state.set(0)

    _control = model.get_control_vec()
    _control['psub'][:] = psub
    controls = [_control]

    _times= DT*np.arange(int(round(T_TOTAL/DT))+1)
    times = linalg.BlockVec((_times,), ('times',))

    # Set the file and write the simulation results to it
    file_name = case_config(mesh_name, psub, ECOV, EBODY)
    file_path = f'{out_dir}/{file_name}.h5'

    if not isfile(file_path):
        with sf.StateFile(model, file_path, mode='w') as f:  
            forward.integrate(model, f, ini_state, controls, props, times, use_tqdm=True)

if __name__ == '__main__':
    print("Running Psub variations")
    ecovs = [ECOV]
    with Pool(processes=args.num_processes) as pool:
        pool.map(run, PSUBS)

    # for loop version for easier debugging
    # for psub, ecov in product(PSUBS, ecovs): 
    #     run(psub, ecov)
