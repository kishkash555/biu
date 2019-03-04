
* MLPs and their feature interpretation
* a model of how the network hanldes correlated input (need to build it!)
* why binary network seems to work: information is not encoded in scale, "corner" projections enough - actual networks don't use higher resolution than that.

suggested design for experiment:
5 iid variables t0~t4 (probably uniform)
3x's that are based on unitary combinations of the variables (+ nonlinearity)?

8 classes based on the values of 3 variables (the other two serve as noise)
construct an optimal solution analytically
train and see how close the solution that is obtained.

general directions for "thinning" an existing NN: 
* 
---------------
* scale is important in order to use the nonlinearities i.e. 
    * squash unnecessary detail in the extermeties, by scaling out and centering until the extremeties are in the saturation zone
    * create non-monotonous functions of an input variable by distributing a single variable to several features then combining these features in a different order.

a non-monotonous variable can be dealt with in two layers, provided it is distributed to enough features, each with different cutoff (controlled by the scale and center). if non-monotonous behavior is expected, the training can possibly be assisted by either splitting the range in advance (i.e. breaking a single variable into several soft ranges by indicator variables a periodic transformation such as fourier).

the features themselves need a high bit depth during training, in order to allow a large number of training updates to amount to meaningful change in a quantity (although it is interesting to ask how such small increments come to being in the first place and can the training be made more efficient by sacling up the important ones and eliminating the unimportant ones).

During prediction, a much lower bit rate (corresponding to features representing fixed point in space) is sufficient, so a network can and should be trained in a way that limits the bit depth of the resulting coefficients.

if we limit each layer to low-order interactions, are we hurting the expresivness of the network? low-order interactions can be combined to form higher-order interactions, and the scale is geometric (each layer increases the order of the interactions by a factor K). maybe we can "rearrange" the network, so that each layer describes only low-order interactions of its features, and in this way create a network that is both more organized, more readable, and more efficient to represent and calculate

---

## Binary networks Soudary et al. 2016
* It is possible to train networks based (almost) completely on binary weights (+/-1) and activations.
* Almost:
    * The inputs are scaled and centered 
    * The final activations are not binarized, argmax
Binarization is achieved through the sign function
* Gradient descent works by “straight-through” approach (similar to '60s perceptrons)
* A modified version of _Batch Norm_ is used (in all layers?) 
* Best-in-class MLPs and convolutional networks were replicated with same number of layers and similar number of nodes in each layer [^nodes_in_layer]
* potential savings (memory, evaluation time, energy) are very signficant (100x)
* Interesting research around this on some fundamental notions of network calculation


[^nodes_in_layer]: or up to 3 times as many nodes.

---

## Batch norm
Earlier (2013)
### Motivation: 
* uncorrelated, normalized variables should allow larger training steps and faster convergence.
* all layers (except input layer) experience _internal covariate shift_ (drift of its inputs) as we train previous layers' coefficients, requiring them to constantly adapt and slowing their convergence.
* decorrelating the training set is computationally expensive, and not easily transferrable to the testing phase.
---
## Batch norm (cont)
### Approach:
* a BN layer can be added anywhere in the architecture.
* The BN layer first normalizes and centers all the inputs based on the minibatch statistics, then scales and shifts the entire population based on learnable parameters.
* significant learning speedups were achieved.

---

## Choosing features
* an M by N matrix does not necessarily need to have MxN free parameters: it can be expressed as U*V, where U is MxK and V is KxN.
* The minimum K is 1, leading to just M+N parameters
* The approach in this paper is to fix U rather than train U and V simultaneously. U can be fixed for all layers, or change from layer to layer
* The interpretation is that U contains a dictionary of man-made _features_ (combinations of inputs that are believed to be imporant) and V is calculated to makes the most use of the feature in this particular layer for this particular problem.


---
## Hashing
* This is similar to using features in that it constrains the transformation matrices to have less than MxN free parameters
* The authors use arbitrary repititions of values in different positions in a transformation matrix to realize savings in memory and computations.
* Significant savings are reported


