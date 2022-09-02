## Onset sensitivity

### Motivation

- Phonation onset pressure is an important metric of speech function as it is one way to characterize the 'ease' with which phonation can be initiated.
- Some studies have shown that phonation onset pressure is elevated in patients with vocal disorders; this suggests that onset pressure can also be an indicator of vocal health.

- Many past studies have investigated phonation onset conditions (pressure, frequency, etc) and the primary factors that influence it.
  - For example: [Titze1988], [Lucero1998], [Zhang2007], etc. (a number of other studies have also been conducted by Zhang and Zhang's group)

- Earlier studies gave useful insights but were limited by usage of simplified models (for example [Titze1988])
- More recent studies have used continuum models to investigate the onset conditions with linear stability analysis
- These studies have provided useful insights into phonation onset but are limited by the finite number of parametric variations explored

- To address these limitations, this study aims to investigate phonation onset conditions using sensitivity analysis and optimization
- We investigate sensitivity of measures computed from phonation onset with respect to continuous variations in parameters, specifically:
  - elastic moduli,
  - shape,
  - (maybe viscosity?) (I have a feeling that the trend for viscosity will simply be to set it to zero, but this doesn't seem meaningful to me)

- We could present sensitivity results of:
  - phonation onset pressure
  - phonation onset frequency

- and present parameters that minimize a measure of phonation efficiency
  - (not super-straightforward to define since the 'output' sound at phonation onset is zero)
  - Potential measure of phonation efficiency:
    - product of phonation onset pressure and damage measure in VF interior at the fixed point
    - product of phonation onset pressure and 'compliance' at the fixed point
      - The compliance is a common object minimized in structural problems and is simply the total strain energy of the structure (have to find the source where I saw this again)
    - product of phonation onset pressure and mean leakage flow
    - phonation onset pressure by itself
      - this is likely not a bounded functional so will result in negative stiffnesses

### Methods

- Use a simplified 2D model of the VFs coupled with an ad-hoc bernoulli flow
  - Choose a nominal geometry of the VFs (M5)
  - Choose base conditions for the VF parameters
  - compute sensitivies for each (or single) base condition/nominal geometry

- Sensitivity results:
  - Compute sensitivties of phonation onset pressure and frequency for:
    - A set of 1 layer stiffness VFs at increasing stiffness factors
    - A set of 2 layer stiffness VFs at increasing stiffness factors
      - Choose the 2 layers to have a fixed 'reasonable' stiffness ratio

- Optimization results:
  - Optimize the model for one (or a few) of the proposed phonation efficiency measures
  - For the chosen efficiency measure(s)
    - use starting points consisting of scaled multiples of a base stiffness (same as for the sensitivity study)
    - for each starting point, constrain the objective function to have the same initial frequency as the frequency at the starting point
  - May have to randomly perturb starting points for one case to judge if there are multiple local minima

**Misc**
For the onset pressure minimization type study, the plan to design a study is:

- Read literature on phonation onset
    - Titze 1988, Lucero 1998, Zhang 2007 (+ additional ones by him)

- Read dynamical systems literature for insights on Hopf bifurcations
    - How do people analyze 'why' Hopf bifurcations occur in dynamical systems?
