# Neural Network compression

---

## (My) Definition
> A procedure which results in an ANN that can perform a task similarly to an existing ANN, consuming considerably less computational resources per processed sample, and typically having a more compact representation.

---

## Main Takeaway from reading so far
There are many approaches, and all of them work exceptionally well!
 
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->


---


| Imposed Constraint | Interpretation | Benefit |
| --- | --- | --- |
| Low bit depth | Coarser search-grid | up to 100x faster |
| "Fix together" arbitrary connection elements  | Impose correlations between columns | 4x reduction in number of parameters |
| Matrix separated into 2 low-rank matrices | Columns constrained to a subspace | ?x compression | 

---

### Section 1 
# Some general observations

---

## Typical ANN layer
* $z^{(\mathcal{l})} = xW^{(\mathcal{l})}+b^{(\mathcal{l})}$
* $a^{(\mathcal{l})} = g(z^{(\mathcal{l})})$, we'll assume $g=\tanh$
* $x^{(\mathcal{l+1})} \equiv a^{(\mathcal{l})}$ i.e. $a$ is the input for next layer

---

## What does a connection matrix "do"?

* Rotate &rarr; Stretch&reflect &rarr; Rotate <!-- .element: style="color: #7070FF" -->
    * The entire matrix is an <!-- .element: class="fragment" data-fragment-index="1" -->  **affine tranformation** in hyperspace <!-- .element: class="fragment" data-fragment-index="1" -->
    * Result may be in a lower- or higher- dimension space <!-- .element: class="fragment" data-fragment-index="1" -->
    * This is the matrix's Singular Value Decomposition. <!-- .element: class="fragment" data-fragment-index="1" -->
* Columns as Features <!-- .element: style="color: #40B040" -->

---

## What does a connection matrix "do"?

* Columns as Features <!-- .element: style="color: #40B040" -->
    * The features are non-linearly combined layer by layer... <!-- .element: class="fragment" data-fragment-index="2" -->
        * ... or in the same layer (two layers are enough) <!-- .element: class="fragment" data-fragment-index="3" -->
    * Last transformation: each <!-- .element: class="fragment" data-fragment-index="4" --> **column** represents the scores of a class 
        * The predicted class is decided by highest score <!-- .element: class="fragment" data-fragment-index="5" -->
        * Probabilstic interpretation: I doubt it  <!-- .element: class="fragment" data-fragment-index="5" -->
    * <!-- .element: class="fragment" data-fragment-index="6" --> **Classification is a "competition" between features** 
        * Scores tend to only go up (we'll see why)

---

## What does a _nonlinearity_ "do"?
* "Squash" together values (beyond a threshold) <!-- .element: class="fragment fade-in-then-semi-out" -->
* Reduces information.. <!-- .element: class="fragment fade-in-then-semi-out" -->
    * but makes things more interesting <!-- .element: class="fragment fade-in-then-semi-out" -->
* Example: <!-- .element: class="fragment" -->
    * <!-- .element: class="fragment fade-in-then-semi-out" --> `$z=4x_1 + 2x_2 + 3x_3$` 
       * changes in values of $x_{(\cdot)}$ have the same effect regardless of $z$
    * <!-- .element: class="fragment fade-in-then-semi-out" --> `$a=\tanh(z + 0.5)$`  
       * for {z | z &gt; 1.3 &or; z &lt; 2.3}, small changes in $x_{(\cdot)}$ have almost no effect on $a$
       * $a$ now has a _qualitative_ rather than _quantitative_ interpretation

---

## What does a nonlinearity "do"? (2)
* <!-- .element: class="fragment" -->  _Saturated_ activations correspond to _decision planes_
* Layer by layer (or in the same layer), planes combine into <!-- .element: class="fragment" --> (non-convex) _regions_
* Therefore, (saturated) activations in intermediate layers are <!-- .element: class="fragment" --> _region indicators_ 
* Regions <!-- .element: class="fragment" -->
    * have soft boundaries,
    * and may overlap.

---

## How do the nonlinearities affect training?
* After reaching an _approximate_ fit, futher epochs are expected to "harden" region boundaries:
    * linear &rarr;&rarr;&rarr; saturated update is faster than &larr;&larr;&larr;. <!-- .element: class="fragment" -->
    * <!-- .element: class="fragment", style="color:#FFB0B0" -->
 **"neg-log-softmax" error term pushes score of correct class to infinity** 

---

### Section 2
# Survey of selected papers

---

## Binarized networks 
#### (Courbariaux _et al._ 2016)
**Concept:**  Develop an MLP with all connections weights restricted to +1 and -1

![binary paper header](Presentations/binary_paper_header.png)

---

## Binarized networks
#### (Courbariaux _et al._ 2016)
**Concept:**  Develop an MLP with all connections weights restricted to +1 and -1
#### Overview  <!-- .element: align="left" -->
* Successfully trained a "BNN" (binarized neural network) on an image classification task <!-- .element: class="fragment" -->
* Reached same performance as reference network  <!-- .element: class="fragment" -->
* with just a modest increase in number of nodes per layer  <!-- .element: class="fragment" -->

---

## Binarized networks (2)
#### Method's benefits  <!-- .element: align="left" -->
* <!-- .element: class="fragment" data-fragment-index="1" --> Demonstrated **7x** time reduction (through custom CUDA kernel) 
* computations should reduce by ~6*10&sup2;: <!-- .element: class="fragment" data-fragment-index="2" -->

| Number represenation | mult 2 numbers - ops | Comments |
| ----: | :-----: | ---- |
| 32 bit (FP)   | ~600 | |
| 1 bit  | 1 | XNOR gate
 <!-- .element: class="fragment" data-fragment-index="2" --> 


---

### Binarization - Limitations
* Custom hardware and compiler optimizations are required. <!-- .element: class="fragment" -->
* Performance outside image classification: not tested <!-- .element: class="fragment" -->
* Cumbersome 'hybrid' compuational model still required:  <!-- .element: class="fragment" -->
    * Inputs: floating point <!-- .element: class="fragment" -->
    * Intermediate layers: binary <!-- .element: class="fragment" -->
    * Class scores: integers <!-- .element: class="fragment" -->

---

### Binarization - Analysis
* _Any_ arbitrary vector in hyperspace can be represented as:
    * A length $\\mathcal{l} \\in \\mathbb{R}^+$, and angles `$\phi _1, \phi_2, ... \phi_{d-1}$`
    * `$\phi_i \in (0, 2 \pi)$`
* ANN matrix columns &xhArr; arbitrary vectors <!-- .element: class="fragment" -->
* BNN matrix columns &xhArr; Constrained vectors: <!-- .element: class="fragment" -->
    * `$\mathcal{l} = \sqrt{d}$`
    * `$\hat{\phi_i} \cdot \hat{\phi_j} = \cos \theta_{ij} \in \pm(1-\frac{2k}{d})$` where `$k = 1,\ldots,d/2$`

**Yet they succeed!**  <!-- .element: class="fragment" -->
    

---


### Binarization - Conclusions
* Length is not necessary to represent "states" 
    * All activations are saturatd
* Good solutions do not require "infinite" resolution in input space.
    * XNOR gates span the complete functional space.

---


## Repeating elements in a connection matrix
<span class="fragment" data-fragment-index="1"> ![hashing_trick](Presentations/hashing_trick_header.png) <!-- .element: class="fragment shrink" data-fragment-index="2" --> </span>

<span class="fragment" data-fragment-index="2">  **Concept**: Save memory and multiplications, by arbitrarily constraining different entries to the same value </span> 

---

### Repeating elements in a connection matrix
![hashing_trick](Presentations/Hashing_trick_illustration.png)


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
![hashing trick performance](Presentations/hashing_trick_performance.png) 
![hashing trick performance](Presentations/hashing_trick_performance_legend.png)
</span>

---

### Connection hasing vs. feature hashing

For $z_i$, layer outputs (pre-nonlinearity):
 
 `$z_i = \sum\limits_{j=1}^{m} V_{ij}a_j$`

Equivalently 
`$z_i = \mathbf{w}^T\phi_i(\mathbf{a})$`
Where

`$$[\phi_i(\mathbf{a})]_k = \sum\limits_{j:h(i,j)=k}a_j$$`

Which means that each $z_i$ depends on a sum of an arbitrary subset of the previous layer's activations $a_1,\ldots,a_m$

---

<!-- .element: class="fragment" --> Mathematically equivalent to _feature_ hashing 
* Heuristic <!-- .element: class="fragment" -->
 I believe that understanding this method's success is of fundamental importance <!-- .element: style="color: aqua" class="fragment" -->


---

### Factorization
* Concept: Train $UV$ rather than $W$, where $U$ and $V$ are low rank.
* Not all authors agree on the effectiveness: 
    * one paper argues that U has to be predetermined
    * A "smooth" $U$ works well in image processing context.
* Factorization performed very poorly in the benchmark conducted by the _Hashing Trick_ authors.

---

### Batch Normalization
* Not a compression method
* Concept: 
    * Speed up training by deliberately eliminating the scale and bias of inputs to a layer
    * Replace the implicit scale and bias of the input population with explicit, learnable scale and bias 
* Faster training led to out-performing state-of-the-art of the time
* Is this "Weak" factorization?


---


### Summary 
* Overlap between approaches
    * But a broader, more systematic framework is needed
* Train-from-scratch vs. rely on existing network: 
    * Train from scratch received more emphasis.
    * Maybe becuase it's easier when training data is not too big and openly available.
    * Training on soft outputs is straightforward
    * Did anyone try a method that compares hidden activations?


---

### Ideas to further develop
* Interpretations for "action" of a layer
    * Rotate &rarr; stretch/reflect &rarr; rotate (the SVD approach)  <!-- .element: class="fragment" -->
        * &rarr; then "squash" (nonlinearity) <!-- .element: class="fragment" -->
    * "features" or "projections" / "voting" system <!-- .element: class="fragment" -->
* Avoiding the heavy parameterization of the final layer <!-- .element: class="fragment" -->
    * having all possible classes compete "head to head" seems redundant <!-- .element: class="fragment" -->
* 

All above approaches train a new model from scratch, using the same training data as the original network



---
