# Compressing Neural Networks With the Hashing Trick

---

![hashing_trick](Seminar/hashing_trick_header.png)

---

### Section 1
# Background

---

## Short Quiz

![Mystery persona](Seminar/Geoff-Hinton.jpg)

Who is this guy?


---

### Deep learning concise timeline
* <!-- .element: class="fragment" --> 1957 - Invention of the perceptron
* <!-- .element: class="fragment" --> 1969 - Seymour Papert, _Perceptron_ 
* 1970s - "ANN winter" <!-- .element: class="fragment" -->
* 1982 - Backpropagation applied to Multi-Layer Perceptrons <!-- .element: class="fragment" -->
* 1990s - Active research, small models <!-- .element: class="fragment" -->
* 2000s - Second ANN winter <!-- .element: class="fragment" -->
* 2012 - ImageNet (ILSVRC) Won by AlexNet (5 Layers, 60 Million parameters) <!-- .element: class="fragment" -->
* 2012-present - Deep learning boom <!-- .element: class="fragment" -->

---

![tech-landscape](Seminar/ai-landscape.jpg)

---

### The motivation to use NN

![why-deep-learning](Seminar/Why-Deep-Learning.png)


---

### It works!
* Major impact:
    * image recognition
    * speech to text
    * natural language processing
    * ...

### It grows!
* New architectures and topologies spring constantly
* New fields of application

---

### How are Performance Improvements Achieved?
* Additional training examples <!-- .element: class="fragment" data-fragment-index="1" -->
* Deeper, wider networks <!-- .element: class="fragment" data-fragment-index="1" -->
* Combining various submodules into a single architecture <!-- .element: class="fragment" data-fragment-index="1" -->

&dArr; <!-- .element: class="fragment" data-fragment-index="2" -->

**Constant increase in network sizes** <!-- .element: class="fragment" data-fragment-index="2" -->

---

![network-year-size](Seminar/parameter-number-growth-by-year.jpg)

---

### AlphaGo - 2017
![alphago](Seminar/alphago.jpeg)

He must be thinking about the $25 Million in hardware cost...

---

### Can we do the same with less?
If we can, we will...
* Reduce operational costs
* Reduce energy consumption
* Open the way for wider use
* Accelerate development of new solutions

---

### After intese theoretical analysis...
![yok](Seminar/mr-bean-scratching.jpg) <!-- .element: class="fragment" -->

---

### Summary
* Technological evolution towards more sophisticated networks
* More efficient networks "are out there"
* Demand for methods that will either: 
    * Train compact networks more efficiently
    * Compress networks post-training

---

### Section 2
# A few compression techniques

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

### Matrix dimension trick
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

### Matrix dimension trick
* Pros:
    * Truely reduces number of operation
    * "Scalable" compression (with $r=1,2,\ldots$)
    * Hardware already optimized
* Cons:
    * hard to train

---

### Bit-depth reduction
* Take the _k_ most signficant bits (MSBs) of each weight 

---

* Pros:
    * Virtually no loss of accuracy
* Cons:
    * Need custom kernel to enjoy computational benefits

---

### Section 3
# The Hashing Trick

---

### What is a Hash?

$\mathcal{H}:\ \mathbb{N}^d \to \\{1, \ldots k\\}$ 

* Outputs are uniformly distirbuted

---

### Principle
* Replace $W_{n \times m}$ with $V_k$

`$$W_{i,j} = V_{\mathcal{H}(i,j)}$$`

---

### Benefits
* Fast to compute
* Storage free (?)

![hashing_trick](Seminar/Hashing_trick_illustration.png)

---

### Datasets
![mnist](Seminar/MnistExamples.png)
![conv](Seminar/convex-dataset.png)

![mnist-bg](Seminar/mnist-back-random.png)
![mnist-rot](Seminar/mnist-rot.png)
![rect](Seminar/rect-example.png)

---

### Comparison to other methods



---

### Compression &hArr; boosting SGD performance
![EF1](Seminar/efficient-frontier1.png)

---

### Compression &hArr; boosting SGD performance
![EF2](Seminar/efficient-frontier2.png)

---

### Section 2
# Method



---

### Standard ANN layer (fully-connected)
* $z = Wx+b$
* $a = g(z)$ nonlinearity, e.g. $a=\tanh(z)$
* dims:
    * $x \in \mathbb{R}^{m \times 1}$
    * $z,a \in \mathbb{R}^{n \times 1}$
    * $V \in \mathbb{R}^{m \times n}$
    * $b \in \mathbb{R}^{n \times 1}$

---

### Hashed ANN layer

* `$w \in \mathbb{R}^{1 \times K}$`
<p style="margin-bottom:1cm;"></p>
* `$h:\ [1 \ldots m ] \times [1 \ldots n] \to [1 \ldots K]$`
<p style="margin-bottom:1cm;"></p>
* `$W_{ij} = w_{h(i,j)}$`
<p style="margin-bottom:1cm;"></p>
* <!-- .element: class="fragment" --> `$\frac{K}{m \cdot n}$` is the compression factor 
* Different hash for each layer gets a different hash

---

![hashing trick illustration](Seminar/Hasing_trick_illustration.png)

---

### Hashing as matrix multiplication
* Is there an $H$ such that `$W = Hw$`?

`$W = H^{(1)}w \odot H^{(2)}w \odot \ldots H^{(n)}w$`

`$H^{(j)} \in \mathbb{R}^{m \times K}$`

`$H^{(j)}_{iq}=\cases{1 & h(i,j)=q\\
0 & \text{otherwise}} $`

---


### sources
* why deep learning image: https://machinelearningmastery.com/what-is-deep-learning/
* SGD image: https://www.hackerearth.com/blog/machine-learning/3-types-gradient-descent-algorithms-small-large-data-sets/
* Learning rate too high or low: https://www.hackerearth.com/blog/machine-learning/3-types-gradient-descent-algorithms-small-large-data-sets/

* AI tech landscape: https://dzone.com/articles/ai-and-machine-learning-trends-for-2018-what-to-ex
* Parameter number by year: https://www.nature.com/articles/s41928-018-0059-3/figures/1
* AlphaGo player: https://www.telegraph.co.uk/science/2017/10/18/alphago-zero-google-deepmind-supercomputer-learns-3000-years/
![hashing_trick](Seminar/Hashing_trick_illustration.png)

* Mr. bean image https://indianexpress.com/article/entertainment/television/rowan-atkinson-says-mr-bean-return-doubtful-5391451/

* MNIST image By Josef Steppan - Own work, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=64810040">Link</a>


* MNIST Rot image: https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits

* Convex and Rect images: Larochelle, H. et al. An empirical evaluation of deep architectures on problems with many factors of variation. In ICML, pp. 473-480, 2007. http://www.dmi.usherb.ca/~larocheh/publications/deep-nets-icml-07.pdf    

---

---

^Note:
### Stochastic Gradient Descent
* <font color ="#A0F0A0"> Intuitive </font>
* <font color ="#A0F0A0"> Generic </font>
* <font color ="#F08080"> Depends on initial conditions </font>
* <font color ="#F08080"> Depends on learning rate </font>


![funky-path-SGD](Seminar/SGD-path.png)

---


^Note:
### Theoretical gaps - examples
* Theory lagging far behind practice
    * Fitness for a particular problem? 
    * Optimal architecture for a particular problem?
    * Optimal hyperparameters? (e.g. learning rate)

![learning-rate-pitfalls](Seminar/learning-rate-too-high-or-low.png)

