I conceived theoretically a few phenomena which are related to learning:


1. The first layer is responsible for partitioning the parameter space. Each feature vector in the first-layer matrix partitions the space into two half-spaces, the location and direction of the separating hyperplane determined by the direction of the $w$ vector and the corresponding bias term
1. The magnitude of the $w$ vector generally does not affect classification. it has a few "uses":
    1. It might affect classification if a point is close to a few borders, and it's in the wrong side of a few of them, then the one with the larger magnitude might tip the classification.
    1. With $\tanh$ and sigmoid, a greated magnitude induces stability in the magnitude, since the size of the step (proportional to the derivative at the point) diminishes as the magnitude of the vector increases
    1. With both nonlinear and relu, a greater magnitude induces stability in direction, since the effect of applying perpendicular change with some constant magnitude on the direction would decrease as the magnitude increases.
1. The difference between $Wx+b$ and $W(x+b)$:
    1. It seems that with the first formulation, there might be cases where $b$ and $W$ would compete, making the direction of change in $W$ less coherent. This is an odd result given the fact that it doesn't seem to require any nonlinearity. More analytical/ empirical study of this situation is needed

#### Analysis of "flip distance"
The flip distance is defined as the number of steps until a single example, currently labeled incorrectly, becomes labeled correctly, by iterations of SGD.

Assume a 1D input sample $x$ with a first layer parameter $a$, its current classification is $\tanh(ax)$. let's assume $a\gt 1, x\gt 1$ and the backprop feedback term is a negative constant. The activation would apply  