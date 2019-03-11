# Neural Network compression

---

## (My) Definition
> A procedure which results in an ANN that can perform a task similarly to an existing ANN, consuming considerably less computational resources per processed sample, and typically having a more compact representation.

---

## Work done so far
* Reading and summarizing ~5 papers
* Writing this presentation

---

## Main Takeaway
There are many approaches, and all of them work exceptionally well!
 
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->
*  <!-- .element: style="color: #202020;" -->


---


| Approach | Benefit |
| --- | --- |
| Coarsening the "grid" of potential vectors  | up 100x faster |
| Constraining different entries of a connection matrix to the same value | 4x reduction in number of parameters |
| Limiting the search space of connection matrix vectors |  ?x compression | 


---


## Binarization
**Concept:**  Develop an MLP with all connections weights restricted to +1 and -1
* Successfully trained a binarized MLP for a classification task <!-- .element: class="fragment" -->
* Moderate increase in number of nodes per layer  <!-- .element: class="fragment" -->
* computations can be reduced by ~6*10&sup2;: <!-- .element: class="fragment" -->

| bits | bit ops to mult 2 numbers | 
| ---- | ----- |
| 32   | 600 |
| 1    | 1 (XNOR) |
 <!-- .element: class="fragment" --> 


---

### Binarization - Limitations
* Requires special kernel/ hardware - otherwise no gain <!-- .element: class="fragment" -->
* Not fully binarized:  <!-- .element: class="fragment" -->
    * Inputs are floating point numbers <!-- .element: class="fragment" -->
    * Class scores are integers <!-- .element: class="fragment" -->
* So needs both special- and general-purpose HW <!-- .element: class="fragment" -->

---

### Binarization - Analysis
* Any vectors can be represented as a length $\\mathcal{l} \\in \\mathbb{R}^+$ and directions `$\phi _1, \phi_2, ... \phi_{d-1} \in (0, 2 \pi)$`
* Binary Networks work despite that:
    * Length is constant for all vectors
    * Connections represent coarse directions (only `$2^d$` directions possible)
* Yet they do well 
    * at least in classification tasks

---

### Repeating elements in a connection matrix
![hashing_trick](Hashing_trick_illustration.png)

All above approaches train a new model from scratch, using the same training data as the original network



* Network compression is an active research topic
* Different groups are using different approaches
* A systematic look at the "cross-section" of current research seems valuable in its own right.

---

## Summary
* Many approaches have been suggested...
* ...But most literature is in the area of image recorginition and convolutional networks
* Still a lot to cover, more reading can help.
* How do we utilize this for sequences? 
