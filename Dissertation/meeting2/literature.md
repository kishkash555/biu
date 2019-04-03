# Neural Networks:
## compression
## and constrained learning

---

### Neural Network Compression 
> A procedure which reduces the size of a network with an acceptable impact on its test accuracy

---

### The _Efficient Frontier_ analogy
#### The EF in finance
![efficient frontier](meeting2/Efficient-Frontier.gif)

---

#### Feasible Neural Networks
![efficient frontier1](meeting2/efficient_frontier1.png)


---

#### Frontier of Feasible Neural Networks

![efficient frontier1](meeting2/efficient_frontier2.png)


---

![efficient frontier1](meeting2/efficient_frontier3.png)

Effective compression should allow us to attain performance otherwise attainable only with larger networks.

---


## MLP CLASSIFIERS

---

### MLP Layer

* $z = xW+b$
* $a = g(z)$ nonlinearity, e.g. $a=\tanh(z)$
* dims:
    * $z \in \mathbb{R}^{1 \times n}$
    * $a \in \mathbb{R}^{1 \times m}$
    * $W \in \mathbb{R}^{m\times n}$
    * $b \in \mathbb{R}^{1 \times n}$

---

### Single activation - intuition
Let $\vec{W_j}$ denote a column of $W$.

Then $z_i = \vec{W_j} \cdot \vec{x}+b_i$
* The _direction_ of $\vec{W_j}$ is the "feature"
* The _magnitude_ of $\vec{W_j}$ determines the "thickness" of the transition from negative to positive saturation.
* **cross-entropy loss** pushes useful features towards **larger and larger magnitudes**.
* The _bias_ term $b_i$ determines _where_ the separation between positive and negative values occurs.

---

## Compression approaches

* ***Indirect***
    * Binarized weights
    * Structure (i.e. repetitions) in $W$
    * Normalize input layer/ intermediate layer activations relative to minibatch
* ***Direct***
    * Post-training weight/ activation elimination
    * Post-training bit depth reduction

---


## Binarized networks 
#### (Courbariaux _et al._ 2016)
**Concept:**  Develop an MLP with all connections weights restricted to +1 and -1

![binary paper header](meeting2/binary_paper_header.png)

---

## Binarized networks <!-- .element: align="left" -->
#### <!-- .element: style="color: #808080" align="left" --> (Courbariaux _et al._ 2016)  
#### Overview  <!-- .element: align="left" -->
* Developed a "BNN" (binarized neural network), an MLP with all connections weights restricted to +1 and -1 <!-- .element: class="fragment" -->
* Successfully trained BNNs on image classification tasks <!-- .element: class="fragment" -->
    * Achieved same accuracy <!-- .element: class="fragment" -->
    * Used same number of neurons per layer <!-- .element: class="fragment" -->
* Demonstrated 7x feed-forward time reduction, through custom CUDA kernel implementation. <!-- .element: class="fragment" -->


---

#### Binarized networks (cont.)  <!-- .element: align="left" -->
### Theoretical speedup factor <!-- .element: align="left" -->
* 1-bit multiplication is _600 more_ efficient than 32-bit floating-point multiplication
* Further potential speedup by exploiting repeating columns.

---

#### Binarized networks (cont.)  <!-- .element: align="left" -->
### Limitations  <!-- .element: align="left" -->
* Realizing speedup depends on either custom hardware or custom kernel. Otherwise no speedup.
* Training is more complex (and may take more cycles).
* First and final layer are not binarized.
* Weights during training are not binarized.
* Proven only for image classification problems.

---

### Binarized networks analysis  <!-- .element: align="left" -->
* The spatial _magnitude_ of each feature vector is restricted to $\sqrt{d}$
* The spatial _angles_ of each feature vector are restricted to the corners of a cube:

`$\hat{\phi_i} \cdot \hat{\phi_j} = \cos \theta_{ij} \in \left\{\pm(1-\frac{2k}{d}) | k \in \left[1,\ldots,d/2 \right] \right\}$`

> The effect on the "expression power" is negligible <!-- .element: class="fragment" -->

---

## Compression approaches

* ***Indirect***
    * ~~Binarized weights~~
    * <b>Structure (i.e. repetitions) in $W$</b>
<u style="color:#A0A0A0;text-decoration:none">
    * Normalize input layer/ intermediate layer activations relative to minibatch
<u style="color:#A0A0A0;text-decoration:none">
* ***Direct*** 
    * Post-training weight/ activation elimination 
    * Post-training bit depth reduction 
</u>

---

</u>

## Structured and low-rank matrices
#### Concept <!-- .element: align="left" -->
* retain the dimensions of $W$ but optimize less than $m\cdot n$ parameters.


#### Approaches <!-- .element: align="left" -->
* Repeat values in arbitrary positions within each $W$
* Express $W$ as a multiplication of two low-rank matrices
* Systematically repeating entries e.g. along secondary diagonals ([toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix))

Each approach requires its own mathematical derivation and methods.

---

## Compressing Neural Networks with the Hashing Trick
<span class="fragment" data-fragment-index="1"> ![hashing_trick](meeting2/hashing_trick_header.png) <!-- .element: class="fragment shrink" data-fragment-index="2" --> </span>

<span class="fragment" data-fragment-index="2">  **Concept**: Save memory and multiplications, by arbitrarily constraining different entries witn a matrix to the same value </span> 

---

### Repeating elements in a connection matrix
![hashing_trick](meeting2/Hashing_trick_illustration.png)


---

### Repeating elements in a connection matrix
* Decide on `$K^{(\mathcal{l})}$`, free parameters per layer, `$K^{(\mathcal{l})} \ll M^{(\mathcal{l})} \times N^{(\mathcal{l})}$`
* Create a hash function `$h: [M] \times [N] \to [K]$` 
* Set `$V_{ij} = w_{h(i,j)}$`
    * $V_{ij}$ is the (virtual) connection matrix
    * $w_{(\cdot)}$ is a vector of $K$ parameters

---

### Repeating elements - Analysis
* Simple and fairly generic (CNNs, RNNs, ...) <!-- .element: class="fragment" -->
* Adjustable compression factor which exceeds binarization  <!-- .element: class="fragment" -->
    * 1:64  &xhArr; &frac12; bit per entry! 
* outperforms other methods(?) <!-- .element: class="fragment" -->
<span class="fragment">
![hashing trick performance](meeting2/hashing_trick_performance.png) 
![hashing trick performance](meeting2/hashing_trick_performance_legend.png)
</span>

---

### Connection hashing vs. feature hashing

For $z_i$ (the layer outputs pre-nonlinearity):

`$z_i = \sum\limits_{j=1}^{m} W_{ij}x_j\ \ ;\ \ W_{ij} = w_{h(i,j)}$`

Where $h(i,j)$ maps from index in $W$ to index in $w$

Equivalently 
`$z_i = \mathbf{w}^T\phi_i(\mathbf{x})$`
Where

`$$[\phi_i(\mathbf{x})]_k = \sum\limits_{j:h(i,j)=k}x_j$$`

Which means we can equivalenty view this as hashing the _inputs_ to the layer. 


---

### Connection hashing - analysis
* Very easy to implement
* Good control of compression factor upto 1:64 (and beyond)
* Not clear why (and if) this works better than reducing the layer size

---

## Factorization
![predicting parameters header](meeting2/predicting_parameters_header.png)

![need_deep networks header](meeting2/need_deep_networks_header.png)

---

### Factorization
$W = UV$

$W \in \mathbb{R}^{m \times n},\ U \in \mathbb{R}^{m \times k},\ V \in \mathbb{R}^{k \times n}$

* Reduces from $mn$ to $(m+n)k$ parameters
* $k$ controls the compression factor

---

### Factorization
* Method was used succesfully in imitating a deep net with a shallow one
* Other authors argue against training $U$ and $V$ together. 
* Instead, They determine $U$ before learning,
    * Either via domain expertise,
    * Or by first learning $U$ separately in a simplified problem.

---

### Factorization (cont.)

* In image processing, $U$ can be designed to span all important smooth filters.
* The benchmark in _Hashing Trick_ ranked this method worst 
    * but they were learning $U$ and $V$ concurrently <!-- .element: class="fragment" -->


---

### Low displacement-rank matrices
![structured transforms](meeting2/structured_transforms.png)

---

### Low displacement-rank matrices
* The naive implementation allows some memory saving, but no speedup.
* The authors develop a more elaborate system that takes advantage of Fast Fourier Transform to speed up both inference and backpropagation, with a controllable compression factor.
* On a CPU, the minimal layer size to achieve wall-clock speedup is 2048 neurons (wider layers &rArr; more speedup).
* Another paper proves the universality of such NNs.


---


 

| Imposed Constraint | Interpretation | Benefit |
| --- | --- | --- |
| Low bit depth | Coarser search-grid | up to 600x faster |
| "Fix together" arbitrary connection elements  | Impose correlations between columns | 4x reduction in number of parameters |
| Matrix separated into 2 low-rank matrices | Columns constrained to a subspace | ?x compression | 
| Structured matrices | - | $O(nlog^2n)$ multiplication |


---

## Compression approaches

* ***Indirect***
    * ~~Binarized weights~~
    * ~~Structure (i.e. repetitions) in _W_~~
    * <b>Normalize input layer/ intermediate layer activations relative to minibatch</b>
<u style="color:#A0A0A0;text-decoration:none">
    * 
<u style="color:#A0A0A0;text-decoration:none">
* ***Direct*** 
    * Post-training weight/ activation elimination 
    * Post-training bit depth reduction 
</u>


---

</u>

## Batch Normalization
![batch normalization header](meeting2/batch_normalization_header.png)

---

### Batch Normalization

<h5 align="left"> Concept (cont.): </h5>

* Speed up training by maintaining direct control over the scale and bias of inputs to a layer

* Replace the implicit scale and bias of the input population with explicit, learnable scale and bias 

<h5 align="left"> Accomplished: </h5>

* Achieved faster training and surpassed state-of-the-art performance in image processing
* Broad adoption

Is this a "Weaker" form of factorization?

---


### Direct methods
![data free param pruning](meeting2/data_free_param_pruning_header.png)

---

### Parameter pruning
* Choose a distance metric between $\vec{W_j}$ e.g. L<sub>2</sub>
* compute the metric for every feature pair:
    * disregarding magnitude differences
    * Giving the bias term more weight in the distance formula
* Starting with the closest pair, pick one member to remove.
* "Reroute" the inputs and outputs of the deleted neuron.
* Stop before "knee" in error curve
* Tested with ReLU activations where $\forall\alpha\ge0,\ g(\alpha \vec{x})=\alpha g(\vec{x})$ so scale can always be delegated to next layer.

---

### Parameter pruning

![pruning-saliency](meeting2/pruning_saliency.png)

---

### Parameter pruning

![owl](meeting2/owl.jpg)

---

### Summary
* Compression, and improved learning, are interwined.
* Methods can be mapped to a "spectrum" from most inutitive/heuristic to more mathematically involved.
* Research area still wide open 

---

### Next steps
* Implement:
    * The hashing trick
    * Binarized networks 
    * Parameter pruning
* Ideas to develop

---

### Ideas to develop
* Stochastic neurons
* Layer genetic evolution
* Correlations between activations
* Scoring layer reduction techniques


---

## Compression approaches

### Not encountered in literature
* Concatenate output of a (deep) layer with outputs from a previous layer
* Detect correlation between neurons (in same layer or different layers)
* Stochastic neuron activation

---

## SUMMARY 
* Approaches "overlap"
* No theoretical framework
* One successful application is not enough to understand approach
* Smaller networks which "disguise themselves" as larger ones
* Looks like everyone is trying to fool SGD...


---

## NEXT STEPS

#### Further Reading
* Sequence models - training and compression
* Architecture evolution
* Training on "soft" scores
* SGD analysis


---

## NEXT STEPS

#### Reasearch
* Implement ordinary vs. hashed MLP and look into training process
* Design a problem with a known solution and check the solutions reached by SGD

#### Ideas to further develop
* Decision-tree-like splits
* Correlations and linear dependence between activations


---

#  THANK YOU!

---


---

### Constraining network structure
* Reduce bit depth
* Reduce the number of free parameters in $W$:
    * hard structure
    * soft structure
    * unstructured

### Smarter training
* _distillation_: Train a "student" network on outputs of "teacher" network
* Batch Normalization

### Post training
* Remove very similar features occurring in same layer 
