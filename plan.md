## Onset sensitivity

### Motivation

- Phonation onset pressure is an important metric of speech function as it is one way to characterize the 'ease' with which phonation can be initiated.
- Some studies have shown that phonation onset pressure is elevated in patients with vocal disorders suggesting that vocal disorders could be a cause of the increase.

- Many past studies have investigated phonation onset conditions (pressure, frequency, etc) and the primary factors that influence it.
- For example: [Titze1988], [Lucero1998], [Zhang2007], etc. (see other studies conducted by Zhang's group)

- Earlier studies were gave useful insights but were limited by usage of simplified models
- More recent studies have used continuum models to investigate the onset conditions with linear stability analysis
- These studies have provided useful insights into phonation onset but are limited by the finite number of parametric variations explored

- To address these limitations, this study aims to investigate phonation onset conditions using sensitivity analysis
- We investigate sensitivity of measures and phonation onset with respect to continuous variations in parameters, specifically:
  - elastic moduli,
  - shape,
  - (maybe viscosity?) (I have a feeling that the trend for viscosity will simply be to set it to zero, but this doesn't seem meaningful to me)
- The measures at phonation onset are:
  - phonation onset pressure
  - phonation onset frequency
  - (a measure of phonation efficiency?)
    - (I'm not sure how phonation efficiency could be defined at phonation onset since the 'output' at phonation onset is zero)
    - (potential case: ratio of phonation onset pressure to damage measure in VF interior at the fixed point?)

### Methods

- Use a simplified 2D model of the VFs coupled with an ad-hoc bernoulli flow
  - Choose a nominal geometry of the VFs (M5)
  - Choose base conditions for the VF parameters
  - compute sensitivies for each (or single) base condition/nominal geometry

- Optimize the simplified 2D model for the measure of phonation efficiency for a sequence of target fundamental frequencies

**Misc**
For the onset pressure minimization type study, the plan to design a study is:

- Read literature on phonation onset
    - Titze 1988, Lucero 1998, Zhang 2007 (+ additional ones by him)

- Read dynamical systems literature for insights on Hopf bifurcations
    - How do people analyze 'why' Hopf bifurcations occur in dynamical systems?
