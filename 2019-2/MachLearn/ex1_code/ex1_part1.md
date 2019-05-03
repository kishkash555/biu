### Question 1.1
Algorithm $\mathcal{A}$ is not ERM because there are inputs for which it does not output the minimum-risk classifier. Consider an input space of two dimensions i.e. $x=(x_1,x_2)$, and let our sample $\mathcal{S}$ contains just these three positive examples:
$\{x=(-1,0),\ y=1\}$ 
$\{x=(0,1),\ y=1\}$ 
$\{x=(1,0),\ y=1\}$ 

The first two points are spaced 2, so the minimum-diameter circle containing both has to have a diameter 2. it is easy to verify that of all the circles with diameter 2, the only one containing all three positive examples above is centered at $(c1,c2)=(0,0)$.

However, we have not yet considered the positions of the negative examples. Let's assume that we have a negative example at $x=(0,-0.9)$, and that all other negative examples are likewise on the $x_2$ -axis and below it ($x_2 \lt -0.9$). The circle found by our algorithm $\mathcal{A}$ will include at least one negative example which means it will have a _positive_ empirical loss. This means that it is not ERM if it is possible to find a lower-risk hypothesis from the same class. And indeed, it is possible: A circle centered at $x=(0,1)$ with diamaeter 1.5 contains all three positive points, does not contain any negative point (since its bottom point is at $x=(0,-0.5)$, above the topmost negative example), and therefore has a risk of 0.

To summarize, there exists an $h^*$ for which the total empirical risk is zero, but our algorithm  $\mathcal{A}$ will converge to a subpotimal $h'$.

#### Empirical risk minimizer in the plane
Can we define an Empirical Risk Minimizer in the plane in terms of a minimal shape? If we restrict the hypothesis class so that the center of the circle is fixed, then larger circles will contain smaller ones. In that case, the smallest circle containing all positive points is an ERM since any circle larger may contain more negative points, leading to higher loss, but not contain more positive points (since by definition they were all included in the algo's output). This approach would work with any shape (oval, rectangle,...) as long as the hypothesis class is restricted to shapes where shapes with larger "radius" fully contain shapes with smaller radius, and as long as $h^*$ has necessarily zero error.

### Question 1.2
$\mathcal{S} = \{x_i | i=1 \ldots m\}$
$\mathbb{E}\left(\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{I_{\{h(x_i)\ne f(x_i)\}}} \right)=$ [Linearity]

$\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{E}\left(\mathbb{I_{\{h(x_i)\ne f(x_i)\}}} \right)=$ [Definition of indicator]

$\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{P}\left[h(x_i)\ne f(x_i) \right]=$$\frac{m}{m}\mathbb{P}_{x\sim D}\left[h(x)\ne f(x) \right]=$

$$\mathbb{P}_{x\sim D}\left[h(x)\ne f(x) \right]$$ Q.E.D

