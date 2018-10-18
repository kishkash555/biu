# הסתברות - מבוא 
## יסודות מרחב מדגם בדיד

- $\Omega$: The group of _outcomes_ (i.e. an entire configuration).
- $\mathcal{F}$: $2^\Omega$

 will be called an _event_, and it is the group of all subgroups of $\Omega$.

Because $\Omega$ is countable, we can define an __atomic probability function__:

$$
    p: \Omega \rightarrow [0,1],\\
    \sum_{\omega \in \Omega} p(\omega)=1
$$

From which a definition of $\mathbb{P}: \mathcal{F} \rightarrow [0,1]$ follows intuitively:

$$\mathbb{P}(A \in \mathcal{F})=\sum_{\omega \in A}p(\omega)$$

we then say that $\mathbb{P}$ _should_ be a __probability function__, which is defined as follows:

$$
{P}: \mathcal{F} \rightarrow [0,1]\\
\mathbb{P}(\Omega)=1\\
\mathbb{P}(\dot{\bigcup} A_i) = \sum_{i \in \mathbb{N}}\mathbb{P}(A_i)
$$


### inclusion-exclusion principle

![inc-ex](a1.png)

