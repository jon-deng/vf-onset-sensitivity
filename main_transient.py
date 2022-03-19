"""
This code runs transient simulations that should correspond to the simulations
in main_hopf.py
"""
import os
from os.path import isfile, isdir
import argparse
from multiprocessing import Pool
from itertools import product
import numpy as np

from femvf import forward, load, statefile as sf
from femvf.models import solid as smd, fluid as fmd
from blocklinalg import linalg

from libmaintransient import case_config

parser = argparse.ArgumentParser()
parser.add_argument('--zeta', type=float, default=0.0)
parser.add_argument('--r_sep', type=float, default=1.2)
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()

PSUB = 800 * 10

# These parameters cover a broad range of finite oscillations to see if they are self-similar
PSUBS = np.arange(800, 1600, 200) * 10

# for zeta=1e-3, rsep=1.2:
# These parameters cover a range from damped oscillation to finite oscillation
PSUBS = np.arange(800, 700, -10) * 10

# These parameters cover a more detailed range where onset seems to occur
PSUBS = np.arange(740, 780, 1) * 10

ETA_VISC = 5
ECOV = 5e3*10
ECOVS = ECOV * np.linspace(0.8, 1.2, 21)
ECOVS = [ECOV]
EBODY = 15e3*10

DT = 5e-5
T_TOTAL = 0.5

mesh_name = 'BC-dcov5.00e-02-cl1.00'
mesh_path = f'mesh/{mesh_name}.xml'
model = load.load_fsi_model(mesh_path, None, SolidType=smd.KelvinVoigt, FluidType=fmd.Bernoulli, coupling='explicit')

## Set model properties to nominal values
props = model.get_properties_vec()

# constant ones
props['rho'][:] = 1.0
props['nu'][:] = 0.45
props['eta'][:] = 5.0

# geometric properties related to the symmetry/contact planes
pre_gap = 0.01
ygap_lb = 1e-5
props['y_midline'][:] = np.max(model.solid.mesh.coordinates()[..., 1]) + pre_gap/2
props['ycontact'][:] = props['y_midline'] - ygap_lb 
props['kcontact'][:] = 1e16

# Fluid properties
ZETA = args.zeta
R_SEP = args.r_sep
props['r_sep'][:] = R_SEP
props['ygap_lb'][:] = ygap_lb
props['zeta_lb'][:] = 1e-6
props['zeta_amin'][:] = ZETA
props['zeta_sep'][:] = ZETA
props['zeta_ainv'][:] = ZETA

# Create the output directory
out_dir = f'out/zeta{ZETA:.2e}_rsep{R_SEP:.1f}'
if not isdir(out_dir):
    os.makedirs(out_dir)

# Get DOFs associated with layer regions
from femvf.meshutils import process_meshlabel_to_dofs
solid = model.solid
region_to_dofs = process_meshlabel_to_dofs(solid.mesh, solid.cell_func, solid.scalar_fspace, solid.cell_label_to_id)

dofs_cover = region_to_dofs['cover']
dofs_body = region_to_dofs['body']

# Set the constant body layer modulus
props['emod'][dofs_body] = EBODY


def run(psub, ecov):
    # Set the initial state/properties/control needed to integrate in time
    ini_state = model.get_state_vec()
    ini_state.set(0)

    _control = model.get_control_vec()
    _control['psub'][:] = psub
    controls = [_control]

    props['emod'][dofs_cover] = ecov

    _times= DT*np.arange(int(round(T_TOTAL/DT))+1)
    times = linalg.BlockVec((_times,), ('times',))

    # Set the file and write the simulation results to it
    file_name = case_config(mesh_name, psub, ecov, EBODY)
    file_path = f'{out_dir}/{file_name}.h5'

    if not isfile(file_path):
        with sf.StateFile(model, file_path, mode='w') as f:  
            forward.integrate(model, f, ini_state, controls, props, times, use_tqdm=True)

# if __name__ == '__main__':
    # Investigate how well oscillations at onset correlate with oscillations past onset
    # print("Running Psub variations")
    # ecovs = [ECOV]
    # with Pool(processes=args.num_processes) as pool:
    #     pool.starmap(run, product(PSUBS, ecovs))

    # # for loop version for easier debugging
    # # for psub, ecov in product(PSUBS, ecovs): 
    # #     run(psub, ecov)

    # # Investigate the shape of an measurement-fit type objective function with VF  
    # # modulus changes
    # print("Running Emod variations")
    # psubs = [PSUB]
    # ecovs = [ECOV] # don't want to run this one
    # with Pool(processes=args.num_processes) as pool:
    #     pool.starmap(run, product(psubs, ecovs))
