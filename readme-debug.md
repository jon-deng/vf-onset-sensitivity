This branch was set up for a small change in the sensitivity results.
Before I modified the fluid model a little bit, the first eigenvalue I got for the 'test' case was about 10; however, after modifying the fluid model the first eigenvalue is about 8.5.
Checking the model behaviour, it seems everything is nearly identical within floating point errorsso it's unclear why this happens.

The file debugalias.sh sets up two aliases for checking out before/after changes that illustrate the weird bug.
