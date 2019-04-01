# Neural Networks:
## More about learning

---

#### SECTION 1
## INVESTIGATE LEARNING PROCESS OF A TOY CLASSIFIER

---

### Motivation

* Demonstrate the claim that cross-entropy training with "one hot" 0/1 labels will lead to "a race to infinity".
* Investigate the learning process of an over-parameterized model.

---

### Toy MLP Setup
* $y = U(\tanh(Wx+b_0))+b_1$:
    * $x \in \mathbb{R}^2$
    * $W\in  \mathbb{R}^{2\times2}$
    * $U\in  \mathbb{R}^{2\times2}$
    * $b_0,\ b_1 \in \mathbb{R}^{2}$

* Initialization:
    * $W,\ U$: Glorot uniform
    * $b_0,\ b_1:\ \mathbf{\vec{0}}$ 

Let  $\Theta = \left\{W_,U, b_0, b_1 \right\}$ denote the trainable parameters.
$\Theta_0$ represents parameters initial values.

--- 

### Visualization
* The value of each output neuron depends on the two input neurons: $f_i:(x_1,x_2) \to y_i$ 
* Confound our interest to the region $(x_1,x_2) \in [-1,1]\times[-1,1]$
* Sample many points in this region (on an evenly spaced grid).
* Color each image pixels according to the output value at that coordinate.

---

### Experiment

* Assume the (random) initialization is a _trained classifier with 100% training-set accuracy: 
* Generate a 2000-point training set as follows:
$X_i  \sim U\left(  [-1,1]\times[-1,1] \right)\\  
Y_i = \begin{cases} 
0,\ y^{(i)}_1 \lt y^{(i)}_2\\
1,\ else\\
\end{cases}$
for $y_i = f_{\Theta_0}(X_i)$ the function represented by the 

* Train with Cross Entropy Loss for 5 epochs.



Research question: Focus on **What

