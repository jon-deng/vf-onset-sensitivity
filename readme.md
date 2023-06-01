# Sensitivity of phonation onset pressure to vocal fold stiffness distribution

This package accompanies the paper '--' and performs the onset pressure experiment described there.

## Installation

To run this package, you will have to install the packages:

- <https://github.com/jon-deng/vf-fem> or <https://github.com/UWFluidFlowPhysicsGroup/vf-fem> (contains different vocal fold model definitions)
- <https://github.com/jon-deng/block-array> (utilities for working with block matrices/vector)
- <https://github.com/jon-deng/nonlineq> (utilities for solving nonlinear equations)
- <https://github.com/jon-denf/vf-exputils> (miscellaneous utilities for running the experiment)

You will also need some common python packages such as 'matplotlib' and 'jupyter'.

## Running the script

After installing the required packages you can run the scripts 'main_onsetpressure.py' and 'main_lsa.py'.
To run the main script which generates data in the paper, use a terminal to run the command
`python main_onsetpressure.py --study-name main_sensitivity --output-dir out`
The file 'fig.ipynb' generates is a notebook that processes the results and generates figures.
To open it, use a terminal to run the command
`jupyter notebook fig.ipynb`

The script, 'main_lsa.py', shows the stability of fixed points as subglottal pressure increases.
To run it, use a terminal to run the command
`python main_lsa.py`
