# Sensitivity of phonation onset pressure to vocal fold stiffness distribution

This package accompanies the paper '--' and performs the onset pressure experiment described there.

## Installation

To run this package, you will need to install the package here (https://github.com/jon-deng/vf-fem or https://github.com/UWFluidFlowPhysicsGroup/vf-fem) which contains the definitions for different vocal fold models.
You will also need some common python packages such as 'matplotlib' and 'jupyter'.

## Running the script

After installing the required packages you can run the scripts 'main_onsetpressure.py' and 'main_lsa.py'.
To run the main script which generates data in the paper, use a terminal to run the command
`python main_onsetpressure --study-name main_sensitivity --output-dir out`
The file 'fig.ipynb' generates is a notebook that processes the results and generates figures.
To open it, use a terminal to run the command
`jupyter notebook fig.ipynb`

The script, 'main_lsa.py', shows the stability of fixed points as subglottal pressure increases.
