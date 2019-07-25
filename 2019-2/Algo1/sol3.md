# Homework Assignment 3
### Algorithms 1 Summer 2019
###ID: 011862141
---

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
There are $2^{n-1}$ possible parses, corresponding to whether we place a word-boundary or not in each of the $n-1$ intra-letter spaces.

#### Recursive formula

$V(i,j)=\begin{cases}
True & i=j\\
T(S,i,j) \vee \bigvee\limits_{i\lt k \lt j}(V(i,k) \wedge V(k,j)) & otherwise
\end{cases}$

Base case: T(S,i,i) is the validity of an empty string and we define it to be True. 
(note I intereprted T(S,i,j) to refer to the sequence S[i]...S[j-1], which is empty if i=j).

Recursion: For a given sequnece of letters, it may be valid as a single word, as a split into two words, three, etc. So the recursive approach is to first check its validity as a single word, then try a split at any position. note $k \in [i+1 ,j-1]$ to avoid a circular definition. The recursion will take care of the case of a split of more than two. I use disjunction (_OR_) between the different splits since it is sufficient for one of the splits to be valid for the entire sequence to be valid. _Within_ the split, we demand conjunction (_AND_) since the entire sequnce is valid only if after a split, both parts can be resolved to a valid Hebrew word.


#### Dynamic programming
We store V[i,j] in an (n+1) x (n+1) array (note we only use the elements from the diagonal to the right, since V[i,j] is defined for $i \le j$)

##### Illustration of calculation order
Example of calculation order for n=4. The arrow indicates dependence, i.e. the cell on the left depends on the pairs on the right. 
* Fill the main diagonal (no dependencies): (1,1), (2,2), (3,3), (4,4), (5,5)
* Fill the second diagonal (no dependencies): (1,2), (2,3), (3,4), (4,5)
* Fill the third diagonal:
(1,2) + (2,3) &rarr; (1,3)
(2,3) + (3,4) &rarr; (2,4)
(3,4) + (4,5) &rarr; (3,5)
* 4th diagonal:
(1,2) + (2,4) &or; (1,3) + (3,4) &rarr; (1,4)
(2,3) + (3,5) &or; (2,4) + (4,5) &rarr; (2,5)
* Final:
(1,2) + (2,5) &or; (1,3) + (3,5) &or; (1,4) + (4,5) &rarr; (1,5)


The result is read from V[1,n] which is the top right entry of V. 

Once (1,k) was calculated, the rest of the column (2,k)~(n+1,k) is no longer needed, but I don't see a trivial way to enjoy memory savings based on this.


##### Path recovery
In this case the path is the sequence of valid words. We can find them by backtracking on the inference order. i.e. if V[1,n+1]=True it means that either there exists a _k_ such that V[1,k]=True and V[k,n+1]=True or T[S,1,n+1]=True. If such a _k_ is found, then the two positions can be likewise resolved to find splits, or they resolve to a single word themselves.

##### Pseudo code
```
Valid_word(S):
    n = |S|
    Initilaize V, an (n+1) x (n+1) array
    for d=0 to n:
        for k = 1 to n+1-d:
            i = k
            j = k + d
            if j=i:
                V[i,j] = True
            else if T[S,i,j]:
                V[i,j] = True
            else:
                x = False
                for w = 2 to j:
                    if V[1,w] AND V[w,j]:
                        x = True
                        break
                V[i,j] = x
    return V[1,n+1]    
```

##### Asymptotic complexity
There are 3 nested loops, all bound by size _n_. Therefore the overall time complexity is $O(n^3)$. The space complexity is derived from the 2d array, hence $O(n^2)$.

## Question 4

Throughout the answer to this question, we will assume, without loss of generality, $B_1 \ge B_2$.

### Why combining doesn't yield a valid/optimal solution
This problem is **not** the same as when having a single knapsack with capacity $B_1+B_2$. Counter-example: suppose the item list includes a single item with weight $w_1 = B_1 + 1$. If we assume a single knapsack, we can carry this item, but with two separate knapsacks, the true answer is 0 since we cannot carry this item in either.

### Why incremental solution is not optimal
The incremental solution is **not** the optimal solution, (where incremental refers to filling one knapsack then the other). Counter-example: 
Suppose 
$B_1=11,\ B_2=6$
$w=[4, 6, 4, 3]$.
$v=[100, 6, 1, 4]$

Suppose we start with $B_1$. Item 1 is very valuable, so surely we will place it in $B_1$. we then have three additional items, with weights 6, 4, and 3. With a remaining capcity of just 7 in $B_1$, we choose item2 (with weight 6), since it is the most valuable. There are no more items we can place here, so we move on to $B_2$. Again we need to choose either item3 with weight 4 or item4 with weight 3, so one of these items will remain behind.

If we solved the same problem greedily, but started with $B_2$, we wouldn't fair better. With a capacity of 6 we can only fit one of the items. Item1 with weight 4 is the most valuable, so we pick item1. The problem now goes to $B_1$, which doesn't have capacity for everything since $ 11 \lt 6+4+3$.

However, the optimal solution is to fit item1 with weight 6 in $B_2$ and carry the rest of the items in $B_1$. This solution is clearly better than both greedy solutions since we carry everything.

#### Recursive formula
We can extend the recursive formula we saw in class, maintaining the overall structure:

$f(j,b_1,b_2)=\begin{cases}
f(j-1,b_1,b_2) & w_j \gt b1 \ge b2\\
\\
\max \begin{bmatrix}
f(j-1,b_1,b_2),\\
f(j-1,b_1-w_j,b_2)+v_j\end{bmatrix} &  b1 \ge w_j \gt b2\\
\\ 
\max \begin{bmatrix}
f(j-1,b_1,b_2),\\
f(j-1,b_1-w_j,b_2)+v_j,\\
f(j-1,b_1,b_2-w_j)+v_j\end{bmatrix} &  b1 \ge b2 \ge w_j 
\end{cases}$



#### Dynamic programming
The dynamic programming solution will require a 3-dimensional array $n \times B_1 \times B_2$. It will keep in cell (j, b1, b2) the optimal solution for a problem with number of items=j, remaining capacity of $B_1$=b1, remaining capacity of $B_2$=b2.



