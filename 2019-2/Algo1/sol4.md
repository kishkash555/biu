# Homework Assignment 4
### Algorithms 1 Summer 2019
###ID: 011862141
---

## Question 1 - Edit Distance between strings
### Recursive formulation

We think of the problem as follows:

    String A    x   x   (x)   x ...   x
                        A[i]

    String B    x   x   x   (x) ...   x   x   x
                            B[j]


* We keep two indexes, _i_ and _j_ 
* our recursive function ED(A,B,i,j) answers the question: what is the minimum Edit Distance for the substrings A[1..i] and B[1..j]. 
* The initial call is to ED(A,B,m,n) and the base condition for the recursion is ED(A,B,0,0) = 0.
* If A[i] = B[j] then the (minimum) price ED(A,B,i,j) equals ED(A,B,i-1,j-1). We could also try not to match A[i] to B[j] but being greedy is optimal. This is since a problem smaller by one charcter can be at most better by 1 in terms of edit distance, so at worst matching is equivalent to not matching.
* If A[i] &ne; B[j] Then we advance the recursion by trying the best option of:
    * Dropping one letter from A at a cost of one ("deletion")
    * Dropping one letter from B at a cost of one ("insertion")
    * Dropping one letter from both at a cost of one ("edit")
    

$ED(A,B,i,j)=\begin{cases}
0 & i=j=0\\
ED(A,B,i-1,j-1) & A[i]=B[j]\\
\min\begin{pmatrix} 
    ED(A,B,i-1,j-1) +1,\\
    ED(A,B,i-1,j)+1,\\
    ED(A,B,i,j-1)+1,\\
\end{pmatrix} & otherwise 
\end{cases}
$


### Dynamic algorithm
The dynamic algorithm uses a 2D array D[i,j] for storing ED(A,B,i,j). The prerequisite for populating cell (i,j) is for cells (i-1,j), (i,j-1) and (i-1,j-1) to be populated so first we populate the first row and the first column, then we continue row by row (or column by column).

Populating one cell takes $O(1)$ so the algorithm's space and time complexity are $O(n^2)$. If Recovery of the editing path is not required, we can keep just the last two rows (or the last n+1 items) which reduces. memory complexity to $O(n)$

## Question 2 - subgroup with exact weight
### Recursive formulation

* The naive complexity is $O(2^n)$ - the number of potential subgroups.
* Our recursive function SEW(A,i,w) answers the question: does the problem have a solution for a group composed of elements A[1]..A[i] and weight of _w_?
* base conditions:
    * If i=0 but w &ne; 0 the answer is False.
    * If i=0, w=0 the anser is True.
* The recursion step looks for a True response when we decide to take the current one (if it is not too heavy), or not take it:

$SEW(A,i,w)=\begin{cases}
w=0 & i=0\\
SEW(A,i-1,w) & w \lt A[i]\\
 SEW(A,i-1,w) \vee  SEW(A,i-1,w-A[i]) & otherwise\\
\end{cases}
$


### Dynamic programming
We rely on $w\in\mathbb{Z}$. We have B, a 2D array n x k. B[i,w] holds a single bit: 1 (True) if weight _w_ can be reached with a subgroup of A[1..i], else 0.

We start by setting B[1,A[1]]. In each step we first copy the array from the previous step, and for every set ("turn on") bit B[i-1,t] we set the bit B[i,t+A[i]], if it was not already turned on. If at any point we set B[*,w], we return True. If we exhaust A without setting the w'th column, we return False.

If there's no need to recover the path (i.e. the items that make up the subgroup), then we can use a 1D array of size n.

Overall we have $O(n\cdot k)$ time complexity. the memory complexity is the same for the more straightforward approach, but can be reduced to $O(k)$.

## Question 3 - visit each workstation in one of two assembly lines
### Recursive formulation

Our recursive function C(i,j) answers the question: what is the minimal cost for moving a product from workstations 1 to _i_ so that the last workstation visit is in line _j_?

* For i=0, C(0,j)=0
* The recursion step looks for the minimum between staying on the same line and switching to the other line. we will define $\bar j=\begin{cases} 1 & j=2\\ 2 & otherwise\end{cases}$ 


$C(i,j)=\begin{cases}
0 & i=0\\
\min(a_{ij}+C(i-1,j), a_{ij}+t_i+C(i-1,\bar j)) & otherwise\\
\end{cases}$

### Dynamic programming

We maintain _C_, a 2 x n array which keeps the minimum cost. We fill the columns from left to right. This is $O(n)$ both in terms of memory and time.


### Question 4 - Ants and polling stations
We'll refer to polling stations as "houses".

There's an analogy between this and the Student Problem. The decision we try to make in each recursive step is - How many ants to skip (zero or more) when placing the next house.
The overall cost may increase if the decision to place the next house _s_ ants away, resulted in one of the ants between the houses needing to walk a greater distance.

We will use an auxilary function M(i,j), defined as follows: M(i,j) is the index of the leftmost ants that walks right, if there are houses in A[i] and A[j] (and no houses in A[i+1]..A[j-1]).

The longest distance an ant in the range i..j needs to walk if the only houses in the range are in A[i] and A[j].

We will start by solving the auxilary problem recursively, then dynamically.

$M(i,j)=\begin{cases}
0 & j=i \vee j=i+1 \\
M(i,j-1) & D(M(i,j-1),j) \le D(M(i,j),i)\\
\min\limits_{M(i,j-1)\lt s\lt j}\{s| D(i,s) \gt D(s,j)\} & otherwise
\end{cases}$

In the definition of $M$ we used: $D(i,j)\equiv |A[i]-A[j]|$, The distance matrix between ants' locations.

_M_ can be filled from the diagonal to the right, line by line. Since _s_ is monotonically increasing in each line, the "scan" over its values is no more than $O(n)$ per line of M. Therefore _M_ can be calculated in $O(n^2)$.

With _M_ calculated, we can solve the main problem. F(i,j) is the best (minimum) cost for a problem defined over ants A[1]..A[i] with j houses available.


$F(i,j)=\begin{cases}
0 & i \le j\\
\min\limits_{0\lt s \lt i-j}(\max\begin{pmatrix} 
    D[i-s,M[i-s,i]-1]),\\
    D[M[i-s,i],i],\\
    F(i-s-1,j-1)\\
    \end{pmatrix}) &o therwise
\end{cases}$

Explanation:
* Base: If we have enough houses to build a house for every ant, then the max distance is 0.
* if there's a house at A[i] and we build the next house at A[i-s], then the maximum cost can come from one of three places:
    * It may be from the leftmost ant that walks right, from the ants in the range _i-s_ to _i_.
    * It may be from the rightmost ant that walks left, from the ants in the range _i-s_ to _i_.
    * It may be from an ant in the range $1\ldots i-s-1$, depending on the optimal solution for that range.

There's another subtlety: For the recursive steps we assume a house at A[i], but for the first recursive call, F(n,k), there's no house at A[n]. We can solve this by adding a fictitious entry A[n+1] and initiating the recursion with F(n+1,k). A[n+1] is far enough, so although the algorithm "thinks" there's a house at A[n+1], ant A[n] will always prefer to go left (even if the first house is at A[k]).


### Dynamic programming
I have already described how _M_ is calculated using dynamic programming.
_F_ requires a 2D array n x k which can be populated in $O(n^2\cdot k)$ time.


#### Summary of time and memory complexity
* Calculating D - $O(n^2)$ time and memory
* Calcualting M - $O(n^2)$ time and memory
* Calculating F - $O(n\cdot k)$ memory and $O(n^2\cdot k)$ time.


## Question 5 - Count of binary trees of N nodes.
### Recursive formulation
T(n) will denote the number of trees of size n.
Reducing the problem:
* We make a root, then allocate $s$ and $n-1-s$ nodes respectively to the left and right subtrees. $s$ can be any number in the range. $0 \lt s \lt n-3$.
* Every combination of the subtrees is a different tree, so we'll multiply their number of configurations.

$T(n)=\begin{cases}
1 & n \le 1\\
\sum\limits_{0 \lt s \lt n-1}T(n-1-s)\cdot T(s) & otherwise
\end{cases}$

### Dynamic programming
The dynamic programming can be implemented using an array of size n, filled from left to right. filling each entry takes $O(n)$ due to the sum in the recursive formula, so the time complexity is $O(n^2)$.




