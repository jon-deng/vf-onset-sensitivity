"""
This code runs transient simulations that should correspond to the simulations
in main_hopf.py
"""
import os
from os.path import isfile, isdir
import argparse
# import warnings
from multiprocessing import Pool
import numpy as np

from femvf import forward, load, statefile as sf
from femvf.models.transient import solid as smd, fluid as fmd
from femvf.meshutils import process_celllabel_to_dofs_from_forms
from femvf.static import static_configuration_coupled_picard
from blocktensor import linalg

from lib_main_transient import case_config
# from main_hopf import set_props
from setup import set_props, setup_transient_model
# warnings.filterwarnings('error')

parser = argparse.ArgumentParser()
parser.add_argument('--zeta', type=float, default=0.0)
parser.add_argument('--r_sep', type=float, default=1.2)
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()

# PSUB = 800 * 10

INIT_STATE_TYPE = 'zero'
INIT_STATE_TYPE = 'static'

# These parameters cover a broad range to try and gauge when onset happens
# PSUBS = np.concatenate([np.arange(200, 300, 10), np.arange(300, 1000, 100)]) * 10
PSUBS = np.arange(550, 650, 10) * 10
# PSUBS = np.arange(200, 300, 10)* 10
# PSUBS = np.arange(300, 1000, 100) * 10

ETA_VISC = 5
ECOV = 5e3*10
EBODY = 5e3*10

R_SEP = 1.0
ZETA = 1e-4

DT = 5e-5
T_TOTAL = 0.6

mesh_name = 'BC-dcov5.00e-02-cl1.00'
mesh_path = f'mesh/{mesh_name}.xml'
model = setup_transient_model(mesh_path)

# Get DOFs associated with layer regions

region_to_dofs = process_celllabel_to_dofs_from_forms(
    model.solid.forms, model.solid.forms['fspace.scalar']
    )

## Set model properties to nominal values
props = model.get_properties_vec()
props = set_props(props, region_to_dofs, model)
model.set_props(props)

# # geometric properties related to the symmetry/contact planes
y_gap = 0.01

# Create the output directory
OUT_DIR = f'out/zeta{ZETA:.2e}_rsep{R_SEP:.1f}_ygap{y_gap:.2e}_init{INIT_STATE_TYPE}_fixed_rsep'
if not isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

def run(psub):
    # Set the initial state/properties/control needed to integrate in time
    ini_state = model.get_state_vec()
    ini_state.set(0)

    _control = model.get_control_vec()
    _control['psub'][:] = psub
    controls = [_control]

    _times = DT*np.arange(int(round(T_TOTAL/DT))+1)
    times = linalg.BlockVector((_times,), labels=[('times',)])

    # Compute the static configuration for the initial state if needed
    if INIT_STATE_TYPE == 'static':
        model.set_control(controls[0])
        model.set_props(props)
        x_static, info = static_configuration_coupled_picard(model)
        print(f"Solved for equilibrium state: {info}")
        ini_state['u'][:] = x_static['u']
        ini_state['q'][:] = x_static['q']
        ini_state['p'][:] = x_static['p']

        _control1 = model.get_control_vec()
        _control1['psub'][:] = psub + 500*10

        _control2 = model.get_control_vec()
        _control2['psub'][:] = psub + 500*10

        _control3 = model.get_control_vec()
        _control3['psub'][:] = psub
        controls = [_control1, _control2, _control3]

    # Set the file and write the simulation results to it
    file_name = case_config(mesh_name, psub, ECOV, EBODY)
    file_path = f'{OUT_DIR}/{file_name}.h5'

    if not isfile(file_path):
        with sf.StateFile(model, file_path, mode='w') as f:
            forward.integrate(model, f, ini_state, controls, props, times, use_tqdm=True)
    else:
        print(f"Skipped existing simulation file {file_path}")

if __name__ == '__main__':
    print("Running Psub variations")
    with Pool(processes=args.num_processes) as pool:
        pool.map(run, PSUBS)

    # for loop version for easier debugging
    # for psub in PSUBS:
    #     run(psub)
