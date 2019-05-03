# Compressing Neural Networks With the Hashing Trick

![hashing_trick](Seminar/hashing_trick_header.png)

---

### Section 1
# Introduction

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
* Revolutionized image processing and autonomous driving
* Benchmarks in other fields seeing constant imporvements
* 

### It grows!
* New architectures and topologies spring constantly
* New fields of application

---

### ...But
* Networks are becoming larger and larger
    * High operational costs
    * Long development
    * High memory requirements
    * High energy requirements
* No theoretical guidelines

---

### Theoretical gaps - examples
* Theory lagging far behind practice
    * Fitness for a particular problem? 
    * Optimal architecture for a particular problem?
    * Optimal hyperparameters? (e.g. learning rate)

![learning-rate-pitfalls](Seminar/learning-rate-too-high-or-low.png)

---

### Stochastic Gradient Descent
* <font color ="#A0F0A0"> Intuitive </font>
* <font color ="#A0F0A0"> Generic </font>
* <font color ="#F08080"> Depends on initial conditions </font>
* <font color ="#F08080"> Depends on learning rate </font>


![funky-path-SGD](Seminar/SGD-path.png)

---

### Motivations for reducing size of network
* Reduce operational costs
* Reduce memory and energy requirements
* Shorten response latency - improve user experience
* Democratize AI (lower overhead costs)

---

### Network compression - Some approaches

* "Optimal Brain Damage" <!-- .element: class="fragment" -->
    * Remove/ join connections
* Low-Rank decomposition <!-- .element: class="fragment" -->
    * Use two "thin" matrices to replace one "fat"
* Decreasing bit depth <!-- .element: class="fragment" -->
* Distillation/ "Dark-knowledge" <!-- .element: class="fragment" -->
    * Use "smooth" scores from big networks during training


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

![hashing_trick](Seminar/Hashing_trick_illustration.png)
