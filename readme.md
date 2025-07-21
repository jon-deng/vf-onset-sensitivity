# Sensitivity of phonation onset pressure to vocal fold stiffness distribution

This package accompanies the paper '--' and performs the onset pressure experiment described there.

## Project structure

The project is organized across directories as follows:

- `src/libhopf`: a python package for solving for Hopf bifurcations
- `tests`: tests for correctness of the `libhopf` package
- `mesh`: meshes used in the study
- `fig`, `out`: directories for output figures and results

The root folder contains scripts `main_lsa.py`, `main_onsetpressure.py`, `fig.ipynb`
which are used to run/analyze the study.

## Installation

To run this package, you will have to install the packages:

- <https://github.com/jon-deng/vf-fem> or <https://github.com/UWFluidFlowPhysicsGroup/vf-fem> (contains different vocal fold model definitions)
  - Install the version with tag `vf-onset-sensitivity`
  - To do this, checkout the appropriate commit using `git checkout vf-onset-sensitivity`
- <https://github.com/jon-deng/block-array> (utilities for working with block matrices/vector)
- <https://github.com/jon-deng/nonlinear-equation> (utilities for solving nonlinear equations)
- <https://github.com/jon-denf/vf-exputils> (miscellaneous utilities for running the experiment)

You will also need some common python packages such as 'matplotlib' and 'jupyter'.

Finally, install the `libhopf` package itself using the command below.

```bash
pip install -e .
```

This will make the `libhopf` package available from your python installation.
Alternatively, you can add `src` to your python path.

## Running tests

Tests are written using `pytest` and can be run using the command below from the project directory.

```bash
pytest tests
```

The tests also demonstrate some basic usage of the `libhopf` package.

## Running the study

After installing the required packages you can run the scripts 'main_onsetpressure.py' and 'main_lsa.py'.
To run the main script which generates data in the paper, use a terminal to run the command

```bash
python main_onsetpressure.py --study-name main_sensitivity --output-dir out
```

You can also run shorter experiments to see what the code is doing using other study names, for example

```bash
python main_onsetpressure.py --study-name test --output-dir out
# or
python main_onsetpressure.py --study-name test_3d --output-dir out
```

The script, 'main_lsa.py', shows the stability of fixed points as subglottal pressure increases.
To run it, use a terminal to run the command

```bash
python main_lsa.py
```

The file `fig.ipynb` generates is a notebook that processes the results and generates figures.
To open it, use a terminal to run the command

```bash
jupyter notebook fig.ipynb
```
