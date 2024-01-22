# Project plan

## Introduction

VFs have highly variable shapes, for example, between individuals as well between different postures for a single individual [cite].
The shape of the VFs is believed to have a significant impact on their self-oscillation.
    A simple example of this is the length and size of the vocal folds; longer, larger vocal folds vibrate at lower frequencies and conversely, shorter smaller vocal folds vibrate at higher ones.
VF shape variations can be in the form of complicated variations in the medial profile [cite] (some studies investigate different profiles from castings of human VFs).

Although VFs have highly variable shapes, explorations of the effect of VF shape on VF dynamics typically only consider variations in idealized VF geometries, such as changes in the angle of the medial surface.
    While these studies show valuable results on the effects of VF shape on (....), to name a few, changes in idealized shapes cannot capture the variability in organic shapes of real VFs.
    This makes it unclear if these results are missing effects of other important shape variations, not captured by idealized VF shapes.

Parameterizing VF shape in a way such that organic VF shape variations can be captured is difficult.
Options include the usage of splines or idealized geometries with variable geometric parameters (for example, M5).
The greater the number of shape variations possible, however, the more shape related parameters that have to explored in a parametric study which would such studies difficult to conduct.

To avoid the problem of parameterizing shape, we instead treat shape as variable on an element level in a FE model.
This results in a high dimensional problem where the shape of a FE model can be varied by moving the points of mesh nodes.
To avoid conducting a parametric study, we use a sensitivity analysis of the model with respect to onset pressure wherein we identify the shape variations responsible for the biggest variations in onset pressure.
While the space of possible shape variations is high, the sensitivity analysis shows that only a few key shape changes are responsible for most of the variation in onset pressure.
We consider onset pressure due to it's importance in speech as a potential gauge of vocal effort; however, future studies could also consider other quantities.

## Methodology

Conduct a sensitivity analysis of onset pressure with respect to arbitrary shape changes to determine the shape changes that have the biggest impact on onset pressure.

Conduct the analysis over a variety of base shapes and stiffness combinations to see if this sensitivity is strongly affected by these parameters.
    Obviously if this is true, then you can't interpret the sensitivity results as being always true at all conditions.

Determine if there is an optimal shape that minimizes onset pressure.
