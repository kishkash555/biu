# Homework Assignment 9
### Algorithms 1 Summer 2019
### ID: 011862141
#### Aug 13th, 2019
---



## Question 1

the numbers $[i,j]$ can be represented as $i+t,\ t \in [0, j-1]$.
The binary representation of t includes $\lceil \log(j-i) \rceil =k$ bits which can be generated through k calls to $random(0,1)$.
Therefore the runtime for this algorithm is $O(\lceil \log(j-i) \rceil)$

## Question 2
#### Description
To get an equally-distributed output of 0 or 1, my algorithm will call $biased-rand()$ 2k times. It will compare the binary number produced by the k "odd" calls (i.e. the outputs of the 1st, 3rd, 5th calls and so on) to the output of the "even" calls (2nd, 4th, 6th calls and so on). It will output 0 if $\sum \limits_{i=1}^k b_{2i-1} * 2^i \lt \sum \limits_{i=1}^k b_{2i} * 2^i$. and 1 if $\gt$.
Note $b_i$ represents the output of $biased-rand()$ in the $i$th time it is called.
If the numbers are equal, it will increase $k$ by one i.e. draw 2 more numbers.

#### pseudocode
Summing this up in pseudocode:

    random01_from_biased_rand():
        do
            a = biased_rand()
            b = biased_rand()
        while (a=b)
        if a < b return 0
        else return 1

#### Correctness
For two consecutive calls to $biased\_rand()$  with $\mathbb{P}(1)=p$, the probabilities are as follows:

case no. | output sequence | probability
---|:--:|---
1 | 00 | $(1-p)^2$
2 | 01 | $p(1-p)$
3 | 10 | $p(1-p)$
4 | 11 | $p^2$

The probability of getting a 0 as a result of a single pass through the loop is the probability of case 2, $p(1-p)$. The probability of gettting a 1 is the probability of case 3, again $p(1-p)$. The probability of getting a 1 or a 0 is independent of results of previous runs since the biased-rand function has no memory (no correlations in the sequence of outputs). So each time the loop runs, if it exits, it will output 0 or 1 with equal probabilities, and if it doesn't exit, we can apply the argument to the next loop.

#### run time
Each loop obviously runs in $O(1)$ time so we are interested in the number of times the loop will be executed. 
The probability of executing the loop again is the probability of cases 1 or 4 which is $P_1+P_4=2p^2-2p+1\equiv w$. Denote event $A_k$: A bernulli trial successful for exactly $k$ consecutive times. If the probability of success of the bernulli trial is $w$, then $\mathbb{P}(A_k)=w^k(1-w)$. To find out the mean, i.e. $\sum \limits_{k=1}^\infty kP(k)$, we note that $k \sim Geom(1-w)$ where Geom denotes the geometric distribution. The mean of this distribution is $\frac{1}{1-w}=\frac{1}{1-(2p^2-2p+1)}=\frac{1}{2p(1-p)}$.


So the mean time complexity is $O(\frac{1}{p(1-p)})$.

## Question 3

#### Lower bound on probability of termination using Markov Inequality
We know the LV's algorithm runtime R is a positive random variable with $\mathbb{E}(R(n))=T(n)$ (we can drop the $n$ in the notation since it doesn't change).


From Markov Inequality, $\mathbb{P}(R \gt a) = \frac{\mathbb{E}R}{a}=\frac{T}{a}$ and so
$\mathbb{P}(0 \lt R \le a) = 1- \frac{\mathbb{E}R}{a}=1-\frac{T}{a}$

If want the probability of termination to be $1-\frac{1}{\alpha}$, we can set
$1-\frac{1}{\alpha} = 1-\frac{T}{a}\\
a=\alpha T$

So we conclude that if an LV algorithm is allowed to run for $\alpha T(n)$ it will return with an answer with probability $1-\frac{1}{\alpha}$.

#### Description of LV &rarr; MC 
1. Set a timeout of $\alpha T(n)$
1. Call the LV algorithm A
1. If timeout expires before algorithm finishes:
    Return arbitrarily "Yes" or "No"
1. Else, return A's response.


#### Explanation
Since A's response is always correct, if we reach line 4, the algorithm gives the correct answer. As we showed, we reach line 4 with probability $1-\frac{1}{\alpha}$ which means the chance of error is bound from above by $\frac{1}{\alpha}$.

## Question 4

No. The algorithm does not generate all possible mutations with equal probability. 
Since there are $n!$ permutations, the probability of each permutation under condition of equality should be $1/n!$. Suppose by way of contradiction that each permuation has a probability of $1/n!$ under the algorithm.

We can see that A[1] will be swapped with an entry A[2]... A[n] in the first iteration. After that, A[1] will not be accessed again in the remaining iterations.

Therefore, the identity permutation (and any other permutation where A[1] remains unchanged) is not possible in this algo, the probability of all such permutations is 0. This is in contradiction to the assumption that they each have a probability of $1/n!$.







