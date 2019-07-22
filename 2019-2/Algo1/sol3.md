## Question 1
#### Naive algorithm
Try every possible path from (1,1) to (n,n), sum the score along each path separately, then pick the path with the highest score.


#### Recursive algorithm
$M[i,j] = \begin{cases}
A[1,1] & i=j=1\\
A[i,j]+\max \begin{pmatrix}M[i-1,j-1],\\M[i-1,j-1],\\M[i,j-1]\end{pmatrix} & otherwise
\end{cases}$

The idea is that a path that goes through (i,j), is only maximal if the path from (1,1) to (i,j) was maximal. Therefore $M[i,j]$ keeps track of the maximum score obtainable from (1,1) to (i,j), with $M[n,n]$ being the maximal overall score.


##### Proof
- A[1,1] is trivially the maximal up to (1,1)
- Assume by induction that $M[i-1,j-1],\ M[i-1,j-1],\ M[i,j-1]$ were maximal
- since these are the only locations from where a step to $M[i,j]$ can originate, picking the maximal among them ensures that there is no other path with higher score going through (i,j) that we have missed.

#### Dynamic programming
We will use a 2D n x n array for $M[i,j]$

As follows from the formula, we start out by setting $M[1,1]=A[1,1]$. Then we start filling the 1st row, $M[1,:]$ left to right, and continue row by row. The solution can be read from $M[n,n]$.

##### Pseudocode

```
MaxScoreGame(A):
    initialize M, n x n array
    for j = 1 to n:
        for i = 1 to n:
            if i = 1 and j = 1:
                M[i,j] = A[i,j]
            else if i=1:
                M[i,j] = A[i,j] + M[i,j-1]
            else if j=1:
                M[i,j] = A[i,j] + M[i-1,j]
            else:
                M[i,j] = A[i,j] + max(M[i-1,j-1],M[i,j-1],M[i-1,j])
            end if
        end for
    end for
    return M[n,n]
```

##### Asymptotic complexity analysis
- Time: the time complexity is $O(n^2)$. We have two nested loops each over $n$ values. The work in each inner loop is $O(1)$
- Space: The n by n array requires $O(n^2)$ space. However, we only need the last two rows, or even more strictly, the last row + $O(1)$ elements if we use a ring buffer and keep track of the index offset. In order to recover the full path, we need the full array.

##### Path recovery
The path is recovered starting at (n,n). By substracting $M[n,n]-A[n,n]$ we get the score at the previous step, which is equal to the number appearing in one of $M[n-1,n],\ M[n-1,n-1],\ M[n, n-1]$. When we find the spot containing that number (the maximal among the three), we mark that spot as the previous step in the path and continue iteratively until we reach (1,1).



## Question 2
#### Naive algorithm
- Enumerate all possible "parse trees" by creating all possible binary trees where the leaf order is according to the order of the numbers in the input.
- Calculate the arithmetic expression resulting from each tree, and store its result.
- Choose the (or one of the) arithmetic expression(s) whose result was highest.

#### Recursive


The input is:
$A_i$: The i'th number in the sequence, $i=1\ldots n$
$S_i$: The i'th operation in the sequence, +1 for a "+" and -1 for a "-"

<!--
The output is:
$T = (C_L,C_R)$ A binary tree node where each child can be either a tree or an $A_i$ input.
-->

We define $B(i,j)$ which is the optimal tree for $A_i \ldots A_j$
$B(i,j) = \begin{cases}
A_i & i=j,\\
\max \limits_{i\le k\le j}(B(i,k)+ S_k \times B(k+1,j)) &otherwise
\end{cases}
$

The recursion checks all possible positions of the current split, with each side computes the optimum for the subexpression via recursion.



#### Proof
##### Correctness
If B(i,k) and B(k+1,j) are correct (i.e. they represent valid results of calculating on i..k and k+1...j respectively), then combining these expressions with the proper sign (represented by $S_k$) is a valid expression using $A_i\ldots A_j$

##### Optimality
Lemma: _For any pair of integers $1 \le i \le j \le n$, the function B(i,j) returns the maximum evaluation of the sequence of numbers $A_i \ldots A_j$._

Proof: By induction on j-i.
Base case: j=i: $A_i$ is optimal (since it is the only option).
Step: Assume lemma is true for $\forall(j-i) \lt l$ and prove for $j=i=l$
Denote b* the optimal value for B(i,j). Assume B(i,j) < b* (i.e not the optimal). 
Denote by k* the index between i and j-1 that is used last in the optimal ordering. So 
$$\begin{aligned}b^* =b^*_{i,k^*} + S_{k^*} \times b^*_{k^*+1,j}  \gt &B(i,j)\\
= & \max \limits_{i\le k\le j}(B(i,k)+ S_k \times B(k+1,j)\\
\ge & B(i,k^*) + S_{k^*} \times B(k^*+1,j)\\
\Darr&\\
b^*_{i,k^*} + S_{k^*} \times b^*_{k^*+1,j} \ge &B(i,k^*) + S_{k^*} \times B(k^*+1,j)\end{aligned}$$



The induction assumption can be applied to $B(i,k^*)$ and $B(k^*+1,j)$, so $B(i,k^*) \ge b^*_{i,k^*},\  S_{k^*} \times B(k^*+1,j) \ge S_{k^*} \times b^*_{k^*=1,j}$. Summing the two expressions and comparing to the non-optimality assumption, we arrive at a contradiction.

## Question 3
#### Naive solution
For every possible parsing, test if each word is valid in Hebrew. If a parse is found such that all words are valid, return True. otherwise return False.

#### Recursive

$V(i,j)=\begin{cases}
True & i=j\\
\bigvee\limits_{i\le k \lt j}(V(i,k) \wedge V(k+1,j)) & otherwise
\end{cases}$

For a given sequnece of letters, it may be valid as a single word, as a split into two words, three, etc. So the recursive approach is to try a split at any position, and use disjunction (_OR_) between them since it is sufficient for one of the splits to be valid for the entire sequence to be valid. _Within_ the split, we demand conjunction (_AND_) since the entire sequnce is valid only if after a split, both parts can be resolved to a valid Hebrew word.


#### Dynamic programming
We store V[i,j] in an n x n array (note we only use the elements from the diagonal to the right, since V[i,j] is defined for $i \le j$)

We first assign $V[k,k]=True,\ \forall 1 \le k \le n$.

We then proceed along the secondary diagonal, calculating V[1,2], V[2,3], ..., V[n-1,n].

We continue to the third, fourth and rest of the diagonals.

The result is read from V[1,n] which is the top right entry of V.







