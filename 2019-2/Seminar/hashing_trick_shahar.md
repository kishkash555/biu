# Compressing Neural Networks With the Hashing Trick

---

![hashing_trick](Seminar/hashing_trick_header.png)

---

### Section 1.1
# Introduction

---

## Short Quiz

![Mystery persona](Seminar/Geoff-Hinton.jpg)

Who is this guy?


---

### Deep learning concise timeline
* <!-- .element: class="fragment" --> 1957 - **Invention of the perceptron**
* 1970s - "ANN winter" <!-- .element: class="fragment" -->
* <!-- .element: class="fragment" --> 1982 - **Backpropagation applied to Multi-Layer Perceptrons** 
* 1990s - Active research, small models <!-- .element: class="fragment" -->
* 2000s - Second ANN winter <!-- .element: class="fragment" -->
* <!-- .element: class="fragment" --> 2012 - **ImageNet (ILSVRC) Won by AlexNet (5 Layers, 60 Million parameters)** 
* 2012-present - Deep learning boom <!-- .element: class="fragment" -->

---

<div style="width: 100%; overflow: hidden;">
<div style="width: 600px; float: left;">

<h3> It works!
<ul>
<li> Major impact:
<ul>
    <li> image recognition
    <li> speech to text
    <li> natural language processing
    <li> Robotics and automation
    <li> ...
</ul></ul>
</div>
<div style="float: right;">

![tech-landscape](Seminar/ai-landscape.jpg)

</div>
</div>


---

### Section 1.2
# Advancing Performance of Neural Networks

---

### How are Performance Improvements Achieved?
* Parameter tuning <!-- .element: class="fragment" data-fragment-index="1" -->
* More data + Larger networks <!-- .element: class="fragment" data-fragment-index="2" -->

![why-deep-learning](Seminar/Why-Deep-Learning1.png) <!-- .element: class="fragment" data-fragment-index="2" -->

---

### The _size_ superiority assumption

![network-year-size](Seminar/parameter-number-growth-by-year.jpg)


---

### AlphaGo - 2017
* Machine built and trained by Google defeats the best Go player in the world

![alphago](Seminar/alphago.jpeg) <!-- .element: class="fragment" data-fragment-index="1" -->

<p class="fragment" data-fragment-index="1" data-markdown> 
It took **&dollar;30 Million** <br> to achieve this...
</p>

---

### Section 2
# Compression

---


### Compression - motivation
* Reduce operational costs
* <font color ="#80F080"> Less energy </font>
* Run on smaller platforms &#128241; <!-- smartphone symbol --> 
* Accelerate development

---

### Compression - idea
* Keep number of neurons
* Reduce learnable parameters in transition matrices
* Use simple maths or algos
* Everything goes (you get points on effort) <!-- .element: class="fragment" -->

---

### Compression - methods
1. Optimal brain damage
1. Matrix decomposition
1. Hashing  


---

### _Optimal Brain Damage_
* **Remove** connections with minimal (absolute) weight

and/or 

* **Join** similar neurons

![pruning-saliency](Seminar/pruning_saliency.png)

---

### _Optimal Brain Damage_
* Pros: intuitive, easy
* Cons:
    * Does not reduce matrix dimension
    * Limited compression potential

---

### Matrix decomposition 
* Layer $\mathcal{l}$ has $m$ neurons
* Layer $\mathcal{l}+1$ has $n$ neurons
* How many multiplications operations will we carry out?

$W \in \mathbb{R}^{n \times m}$ <!-- .element: class="fragment" -->

Calculating $Wx$ requires $m^2\cdot n$ operations <!-- .element: class="fragment" -->

---

`$W_{n\times m} \equiv U_{n \times r} V_{r \times m}$`

`$r \ll \min(m,n)$`

if $r=1$ we require just $m+n$ operations <!-- .element: class="fragment" -->

---

### Matrix decomposition
* Pros:
    * Truely reduces number of operation
    * "Scalable" compression (with $r=1,2,\ldots$)
    * Hardware already optimized
* Cons:
    * hard to train

---

### Section 3
# The Hashing Trick

---

<section style="text-align: left;">

### Hash Functions 

$\mathcal{H}:\ \mathbb{N}^d \to \\{1, \ldots k\\}$ <!-- .element: align:left -->

A hash-function should gives all the outputs "equal chances"
<p style="margin-bottom:1cm;"></p>

### Hash Generators <!-- .element: class="fragment" data-fragment-index="1" -->

`$\mathcal{G}: \mathbb{N} \to \{ \mathbb{N}^d \to \{1, \ldots k\} \}$` <!-- .element: class="fragment" data-fragment-index="1" -->
* Create as many $\mathcal{H}$'s as we need <!-- .element: class="fragment" data-fragment-index="1" -->
* <!-- .element: class="fragment" data-fragment-index="1" --> The outputs of any two $\mathcal{H}$'s are _statistically independent_. 


---

### Reducing layer size

* by _reusing_ elements within a connection matrix 

<div style="width: 100%; display: table;">
    <div style="width: 100%; overflow: hidden;">
        <div style="width: 600px; float: left;">
<p style="margin-bottom:1cm;"></p>

`$V_{i,j} := w_{\mathcal{H}(i,j)}$`

<p style="margin-bottom:0.7cm;"></p>

<ul>
<li> Memory: `$n \times m \rightarrow k$`

<li> $V$ - _Virtual_ connection matrix </li>
<li> $\mathbf{w}$ - actual vector in memory </li>
</ul>


</div>

<div style="margin-left: 620px;">
![hashing_trick](Seminar/Hashing_trick_illustration.png) 
        </div>
    </div>

---

### Forward pass

How to calculate `$\mathbf{z}_{n \times 1} = V_{n\times m} \cdot \mathbf{a}_{m \times 1}$`

<u>Direct</u>
* calculate `$V_{i,j} = w_{\mathcal{H}(i,j)}$`
* $z= Va$ 


---


### Forward pass

How to calculate `$\mathbf{z}_{n \times 1} = V_{n\times m} \cdot \mathbf{a}_{m \times 1}$`

<u>Feature-hashing</u>

Collect $a_j$ into $n$ vectors $\vec{\phi_1} \ldots \vec{\phi_n}$:
* `$S_{ik} = \{j | h(i,j) = k \}$`
* <!-- .element: style="color: #202020" -->
* `$\vec{\phi_i}$` = [ `$\sum\limits_{j \in S_{i1}} a_j$` | `$\sum\limits_{j \in S_{i2}} a_j$` | ... | `$\sum\limits_{{j \in S_{ik}}} a_j$`  ]
* $z_i = \vec{w} \cdot \vec{\phi_i}$ 

---

### Backward pass

* The gradients "originate" from elements of $V$ but need to be accumulated per element of $\mathbf{w^T}$
* This requires the backward mapping `$S_{ik} = \{j | h(i,j) = k \}$` to be kept in memory
* Backward calculations are memory demanding 
* Memory access is inefficient (multiple single-cell access)

---

### Recreation of results
* Used pytorch to implement
* Couldn't read authors codes (~7000 lines of code in _Lua_ lang)
* Needed to try a few "tactics" for backpropagation
* Training is about 200 times slower (2 minutes &rarr; 10 hours)
* No benefit running on GPU
* Was only able to reach 11% accuracy
* Did not work with $\tanh$ activation

---

### Experimental setup and comparison
* Feed-forward Networks of 3 and 5 layers
* The _Virtual_ size remains constant, $k$ varies
* compressions factors of &half; &frac14; &frac18; ... $\frac{1}{64}$
* Compared to other compression approaches from literature

---

### Datasets
![mnist](Seminar/MnistExamples.png)
![conv](Seminar/convex-dataset.png)

![mnist-bg](Seminar/mnist-back-random.png)
![mnist-rot](Seminar/mnist-rot.png)
![rect](Seminar/rect-example.png)

---

### Results - MNIST

![mnist-sbs](Seminar/results-mnist-3layers-5layers-sbs.png)
Note:
* Nothing happening until 1/8
* 5 layers is an overkill for this problem
* The black (plain NN) and RER tell an interesting story

---

### Results - MNIST ROT

![mnist-sbs](Seminar/results-rot-3layers-5layers-sbs.png)

---

### Thoughts - why it works
* Redundancies in feed-forward networks:
    * "Dead paths" / duplicate paths
    * Columns can be shuffled
    * magnitued of vectors is not important
    * Direction can be "jiggled"
* Conclusion: "Weak coupling" between vectors does not interfere with SGD learning
 


---


### sources
* why deep learning image: https://machinelearningmastery.com/what-is-deep-learning/
* SGD image: https://www.hackerearth.com/blog/machine-learning/3-types-gradient-descent-algorithms-small-large-data-sets/
* Learning rate too high or low: https://www.hackerearth.com/blog/machine-learning/3-types-gradient-descent-algorithms-small-large-data-sets/

* AI tech landscape: https://dzone.com/articles/ai-and-machine-learning-trends-for-2018-what-to-ex
* Parameter number by year: https://www.nature.com/articles/s41928-018-0059-3/figures/1
* AlphaGo player: https://www.telegraph.co.uk/science/2017/10/18/alphago-zero-google-deepmind-supercomputer-learns-3000-years/
![hashing_trick](Seminar/Hashing_trick_illustration.png)

* MNIST image By Josef Steppan - Own work, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=64810040">Link</a>


* MNIST Rot image: https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits

* Convex and Rect images: Larochelle, H. et al. An empirical evaluation of deep architectures on problems with many factors of variation. In ICML, pp. 473-480, 2007. http://www.dmi.usherb.ca/~larocheh/publications/deep-nets-icml-07.pdf    

---

# Thank You!

---

### Summary
* Technological evolution towards more sophisticated networks
* More efficient networks "are out there"
* Demand for methods that will either: 
    * Train compact networks more efficiently
    * Compress networks post-training

