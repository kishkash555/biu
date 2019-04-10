### Question 1.1
Algorithm $\mathcal{A}$ is not ERM because there are inputs for which it does not output the minimum-risk classifier. Consider an input space of two dimensions i.e. $x=(x_1,x_2)$, and let our sample $\mathcal{S}$ contains just these three positive examples:
$\{x=(-1,0),\ y=1\}$ 
$\{x=(0,1),\ y=1\}$ 
$\{x=(1,0),\ y=1\}$ 

The first two points are spaced 2, so the minimum diameter of circle containing both is 2. it is easy to verify that of all the circles with diameter 2, the only one containing all three positive examples above is centered at $(c1,c2)=(0,0)$.

However, we have not yet considered the positions of the negative examples. Let's assume that we have a negative example at $x=(0,-0.9)$, and that all other negative examples are likewise on the $x_2$-axis and below it ($x_2\lt-0.9$). The circle found by our algorithm $\mathcal{A}$ will include at least one negative example which means it will have a _positive_ empirical loss. This means that it is not ERM if it is possible to find a lower-risk hypothesis from the same class. And indeed, it is possible: A circle centered at $x=(0,1)$ with diamaeter 1.5 contains all three positive points, does not contain any negative point (since its bottom point is at $x=(0,-0.5)$, above the topmost negative point), and therefore has a risk of 0.

To summarize, there exists an $h^*$ for which the total empirical risk is zero, but our algorithm  $\mathcal{A}$ will converge to a subpotimal $h'$.

#### Empirical risk minimizer in the plane
Can we define an Empirical Risk Minimizer in the plane? It will have to be defined in a way that guarantees each training example will be classified to its specified class. Such an algorithm is _1-Nearest Neighbor_. This algorithm classifies each training example to the class it belongs, since a training example is always its own nearest neighbor. However, this algorithm is not considered a good generalizer, since it may create a large number of very small regions where the classification is different that the surrounding regions.

### Question 1.2
$\mathcal{S} = \{x_i | i=1 \ldots m\}$
$\mathbb{E}\left(\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{I_{\{h(x_i)\ne f(x_i)\}}} \right)=$ [Linearity]

$\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{E}\left(\mathbb{I_{\{h(x_i)\ne f(x_i)\}}} \right)=$ [Definition of indicator]

$\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{P}\left[h(x_i)\ne f(x_i) \right]=$$\frac{m}{m}\mathbb{P}_{x\sim D}\left[h(x)\ne f(x) \right]=$

$$\mathbb{P}_{x\sim D}\left[h(x)\ne f(x) \right]$$ Q.E.D

