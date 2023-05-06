"""
This code runs transient simulations that should correspond to the simulations
in main_hopf.py
"""
import os
from os.path import isfile, isdir
import argparse
from multiprocessing import Pool
import numpy as np

from femvf import forward, statefile as sf
from femvf.meshutils import process_celllabel_to_dofs_from_forms
from femvf.static import static_coupled_configuration_picard
from blockarray import blockvec as bv

from lib_main_transient import case_config
import libsetup
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
mesh_path = f'mesh/{mesh_name}.msh'
model = libsetup.load_transient_model(mesh_path, sep_method='smoothmin')

# Get DOFs associated with layer regions

region_to_dofs = process_celllabel_to_dofs_from_forms(
    model.solid.forms, model.solid.forms['fspace.scalar']
)

## Set model properties to nominal values
prop = model.prop.copy()
prop = libsetup.set_default_props(prop, model.solid.forms['mesh.mesh'])
model.set_prop(prop)

# # geometric properties related to the symmetry/contact planes
y_gap = 0.01

# Create the output directory
OUT_DIR = f'out/zeta{ZETA:.2e}_rsep{R_SEP:.1f}_ygap{y_gap:.2e}_init{INIT_STATE_TYPE}_fixed_rsep'
if not isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

def run(psub):
    # Set the initial state/properties/control needed to integrate in time
    ini_state = model.state0.copy()
    ini_state[:] = 0

    _control = model.control.copy()
    _control['psub'] = psub
    controls = [_control]

    times = DT*np.arange(int(round(T_TOTAL/DT))+1)

    # Compute the static configuration for the initial state if needed
    if INIT_STATE_TYPE == 'static':
        model.set_control(controls[0])
        x_static, info = static_coupled_configuration_picard(model, controls[0], model.prop)
        print(f"Solved for equilibrium state: {info}")
        ini_state[['u', 'q', 'p']] = x_static[['u', 'q', 'p']]

        _control1 = model.control.copy()
        _control1[:] = 0
        _control1['psub'] = psub + 500*10

        _control2 = _control1.copy()
        _control2['psub'] = psub + 500*10

        _control3 = _control1.copy()
        _control3['psub'] = psub
        controls = [_control1, _control2, _control3]

    # Set the file and write the simulation results to it
    file_name = case_config(mesh_name, psub, ECOV, EBODY)
    file_path = f'{OUT_DIR}/{file_name}.h5'

    if not isfile(file_path):
        with sf.StateFile(model, file_path, mode='w') as f:
            forward.integrate(model, f, ini_state, controls, prop, times, use_tqdm=True)
    else:
        print(f"Skipped existing simulation file {file_path}")

if __name__ == '__main__':
    print("Running Psub variations")
    if args.num_processes > 1:
        with Pool(processes=args.num_processes) as pool:
            pool.map(run, PSUBS)
    else:
        # This loop version is for easier debugging
        for psub in PSUBS:
            run(psub)
