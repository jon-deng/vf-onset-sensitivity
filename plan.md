## Onset sensitivity

### Motivation

- Phonation onset pressure is an important metric of speech function as it is one way to characterize the 'ease' with which phonation can be initiated.
- For example, some studies have shown that phonation onset pressure is elevated in patients with vocal disorders; this suggests that onset pressure can also be an indicator of vocal health.

- Many past studies have investigated phonation onset conditions (pressure, frequency, etc) and the primary factors that influence it.
  - For example: [Titze1988], [Lucero1998], [Zhang2007], etc.
  - These studies are implictly based on the body-cover assumption

- Recent work has suggested the VF interior is not a strictly layered structure but has hetereneous variations
- As a result, it is interesting to investigate how non-layered distributions affect phonation onset
- Since the exact form of the heterogenous variations is unknown, this study aims to study how sensitive phonation onset is to arbitrary stiffness variations
  - The sensitivities will be related to observed variations in VF stiffness in literature

#### Aside:
- Could investigate sensitivity of measures computed from phonation onset with respect to continuous variations in parameters, specifically:
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

#### Study and Analysis Ideas
- Show a second order approximation for onset quantities (pressure and frequency)
  - i.e. $p_\mathrm{onset}(E_0+\Delta{E}) = p_\mathrm{onset}(E_0) + \frac{dp}{dE} \Delta{E} + \Delta{E}^T \frac{d^2p}{dE^2} \Delta{E}$
    - The hessian component might have only a few principal directions that would relate to experimental observations of VF stiffness/layers

- Compute the minimum onset pressure with the approximate second order model
  - i.e. solve $\Delta{E}_\mathrm{min} = \frac{d^2p}{dE^2}^{-1} (-\frac{dp}{dE})$
  - compare this with the minimum from purely non-linear function
  - What is the meaning of principal directions w/ negative eigenvalue?
    - These are negative definite 'directions'

- Analyze what the principal directions and gradients do by 'travelling' along those directions
  - plot structural modes/eigenvalues as you travel along the direction?
    - could hypothesize what the parameter perturbations do in terms of changing the structural modes
      - suspect that the principal directions + gradient tend to make two structural modes have similar frequencies?
      - modal coupling type effect?
    - Literature seems to suggest that instability is related to the distribution of structural modes

- Theoretical analysis of linearized dynamics sensitivity
  - Split linearized dynamics into natural structural modes + state coupled forces modes (due to fluid-coupling)
    - i.e. $\dot{y} = M^{-1}K y + F_\mathrm{ext}(y)$
  - Analyze the sensitivity of natural structural modes to matrix entries (i.e. the state coupled forces)
  - Consider how the eigenvalue of a natural structural mode might have real part grow more positive (less stable) due to changes in the state coupled forces

**Misc**
For the onset pressure minimization type study, the plan to design a study is:

- Read literature on phonation onset
    - Titze 1988, Lucero 1998, Zhang 2007 (+ additional ones by him)

- Read dynamical systems literature for insights on Hopf bifurcations
    - How do people analyze 'why' Hopf bifurcations occur in dynamical systems?
