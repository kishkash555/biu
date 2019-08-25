# Homework Assignment 12
### Algorithms 1 Summer 2019
### ID: 011862141
#### Aug 24th, 2019
---

## Question 1
### Part A
#### Modification to support &#x25c7;'s in the pattern
We use a modification of Algorithm 7 from Session 8.
Here's the algorithm, with the modification (added line) highlighted
1. Create M, size n-m+1
1. Create C, size $|\Sigma|$
1. Create I, size $|\Sigma|$
1. for i= 1 to m:
1. &emsp;C[P[i]] = C[P[i]]+1
1. &emsp;Add i to the linked list in I[P[i]]
1. for $\sigma \in \Sigma$ s.t. $C[\sigma] \gt c$:
1. &emsp; perform count of $\sigma$ using FFT
1. for j= 1 to n:
1. &emsp;if C[T[j]] < c:
1. &emsp;&emsp;for i in I[T[j]]:
1. &emsp;&emsp;&emsp;M[j-i+1] = M[j-i+1] + 1
1. **for =1 to m: M[i] = M[i] + C[&#x25c7;]** #_(add the number of joker characters in the pattern uniformly to all offsets)_
1. return M


#### Reasoning
Note that in line 7, when looping over "heavy" characters, we ignore &#x25c7; characters (even if they are "heavy"). Therefore, locations of &#x25c7; in the pattern will not be matched in the FFT, and since they don't appear in the text, they will not be matched in lines 9-12 either. 
But in fact, they should match with _any_ offset, regardless of the character appearing in the text in that offset. So modification 2 adds C[&#x25c7;] i.e. the number of times the joker character appears in the pattern, to all entries of M, making sure that if for an offset _i_, all non-joker characters match, M[i] = m as it should.

#### Running time
The loop in line 13 is $O(m)$ and doesn't affect the algorithm's overall asymptotic running time: $O(n\sqrt{m\log m})$

### Part B
#### Modification to support &#x25c7;'s in the pattern and text
1. Create M, size n-m+1
1. Create C, size $|\Sigma|$
1. Create I, size $|\Sigma|$
1. for i= 1 to m:
1. &emsp;C[P[i]] = C[P[i]]+1
1. &emsp;Add i to the linked list in I[P[i]]
1. for $\sigma \in \Sigma$ s.t. $C[\sigma] \gt c$:
1. &emsp; perform count of $\sigma$ using FFT
1. **r= count of "&#x25c7;" in T[1..m-1]**
1. for j= 1 to n:
1. &emsp;**if T[j+m-1]="&#x25c7;": r=r+1**
1. &emsp;**if T[j-1]="&#x25c7;": r=r-1**
1. &emsp;**M[j]=M[j]+r**
1. &emsp;if C[T[j]] < c:
1. &emsp;&emsp;for i in I[T[j]]:
1. &emsp;&emsp;&emsp;M[j-i+1] = M[j-i+1] + 1
1. for =1 to m: M[i] = M[i] + C[&#x25c7;]
1. return M

#### Reasoning
When a joker character is encountered in the text, it should match any character in the pattern. For the light characters, implementing a loop over M's entries is to costly. So instead we note that the number of unaccounted-matches at offset _i_ is the number of &#x25c7; characters that appear in the segment of text corresponding to that offset: T[i],T[i+1],...,T[m+i-1].
lines 11 and 12 maintain the correct count of &#x25c7; by incrementing _r_ when a joker character enters the text segment and decrementing it when a joker character leaves the text segment. Line 13 updates M to account for these additional implied matches.

#### Running time
The running time is not affected by the modifications. Line 9 is $O(m)$ (outside of any loop) and lines 11-13 are $O(1)$.

## Question 2
### Solution approach
#### Definitions
We define $M$, 
- $M=\{S_i : |Si| \ge c\}$ ($c$ is a constant to be determined later)
So _M_ is a set which contains the sets with size above $c$.
- Denote $|M|=m$

#### Preparation stage
1. Sort each group _(from now we assume all groups are stored in sorted array)._
1. D =Algorithm1(M) _(see below for details)_

#### Query stage
Given indices _i, j_:
1. if both $S_i,\ S_j \in M$: 
1. &emsp; return $D_{m_i,m_j}$ _(retrieve the result from the corresponding entry in D)_
1. else:   (_at least one of $S_i,\ S_j \notin M$:_)
1. &emsp; return Algorithm2($S_i,S_j$)

### Algorithm1 - find if _m_ sets are pairwise-disjoint
#### Description
For each group, we traverese the elements of the rest of the groups, whenever a matching element is found, we update a 2-D array $D$.

#### Pseudocode
Algo1($S_1,S_2,\ldots,S_m$):
1. initialize array E containing the elements of the $m$ groups (with duplications) $|E|=|S_1|+|S_2|+\ldots+|S_m| \le n$
1. initialize array I s.t. I[i] = source group of E[i]
1. Sort both arrays by E's sorting order
1. Initialize D: m-by-m binary array, set all TRUE.
1. for i=1 to m:
1. &emsp;c=1
1. &emsp;for j=1 to n:
1. &emsp;&emsp;if $E[j] \gt S_i[c]$: c=c+1
1. &emsp;&emsp;if $I[j]\neq i$ and $S_i[c]=E[j]$: 
1. &emsp;&emsp;&emsp;$D[i,I[j]]=False$
1. return D



#### Run time analysis:
1. Initializations: $O(n)$ for lines 1, 2, $O(n\log n)$ for line 3, $O(m^2)$ for line 4
1. Loop in line 5: executed $m$ times
1. Loop in lines 7-10: the loop is executed $n$ times. Each operation in lines 8-10 is $O(1)$

Algorithm total: $O(nm + n\log n + m^2)$

### Algorithm 2: find if two sets are disjoint when one is known to be "small"
#### Description
This algorithm uses binary search to find, for each element of set A (the smaller set), if it also exists in B

#### Pseudocode
1. Algo2($S_A,S_B$):
1. \# without loss of generality we assume $|S_A| \le |S_B|$
1. for h in $S_A$:
1. &emsp; if $h \in S_B$: # using binary search
1. &emsp;&emsp; return FALSE
1. return TRUE

#### Runtime analysis
$O(|S_A| \log(|S_B|) \le O(c\log(n))$ Since $S_A \notin M$ and therefore $|S_A| \le c$.

### Determining _m_ and total time analysis
We note that since the total number of elements is $n$, $m \le n /c$. we set: $c = \sqrt{n}$ and so $m \le \sqrt{n}$.
#### Preparation
- The runtime of sort is at most $O(n\log n)= \tilde{O}(n)$
- The runtime of algorithm 1 is $O(nm + n\log n + m^2) = O(n(\sqrt{n} + \log n + 1)) = \tilde{O}(n^{1.5})$


#### Query time
- Querying _D_ is $O(1)$.
- The runtime of algorithm 2 is $O(c\log(n))=O(\sqrt{n}\log(n)) = \tilde{O}(n^{0.5})$


Which conform to the required run times.

## Question 3

#### B: $f(X,Y)=-f(Y,X)$
$f(X,Y) =$
$= \sum\limits_{x\in X}\sum\limits_{y\in Y} f(x,y)$ by definition.
$= \sum\limits_{x\in X}\sum\limits_{y\in Y} -f(y,x)$ by law of symmetry of flow.
$= \sum\limits_{y\in Y}\sum\limits_{x\in X} -f(y,x)$ switch summation order (finite sums)
$= -\sum\limits_{y\in Y}\sum\limits_{x\in X} f(y,x)$  
$= -f(Y,X)$ by definition  


#### A: $f(X,X)=0$
$f(X,X) =-f(X,X)$ Using B above
$\Rightarrow f(X,X)=0$

#### C: $f(X \cup Y,Z)$

$f(X \cup Y,Z)=$
$= \sum\limits_{x\in X \cup Y}\sum\limits_{z\in Z} f(x,z)$ by definition.
$= \sum\limits_{x\in X}\sum\limits_{z\in Z} f(x,z) + \sum\limits_{x\in Y}\sum\limits_{z\in Z}f(x,z)$ by definition of disjunction and provided $X \cap Y = \empty$
$=f(X,Z)+f(Y,Z)$ by definition of $f$.


multiplying both sides by -1 we get
$-f(X \cup Y,Z)=-f(X,Z)-f(Y,Z)$
$f(Z,X \cup Y)=f(Z,X)+f(Z,Y)$


## Question 4: multi-source multi-target flow problem
#### Explanation
The idea here is that the difference between a source/target node and a "through" node is that a source node is "allowed" to break the conservation law. 
If we add an edge from the source into a specific node _v_, and we set the capicity of this edge conceptually to infinity, node _v_'s outgoing flow can increase without restriction (except for the normal outgoing capacity constraint), just as with a source node. Ignoring the new edge, it "looks like" this node breaks the conservation law.
The same discussion can be applied to target nodes.

#### Construction of G'=(V',E')
$V' = V \cup \{s,t\}$
$E' = E \cup \{(s,s_i)| i=1,\ldots,k\} \cup  \{(t_i,t)| i=1,\ldots,\ell\}$
$c'(u,v) = \begin{cases}
c(u,v) & (u,v) \in E\\
\sum\limits_{v\in V} 100\cdot c(s_i,v) & v \in \{s_1,\ldots s_k\} \vee v \in \{t_1,\ldots t_\ell\}\\
\end{cases}$

In words: the capacity is matched for edges that were copied from _G_. A large enough capacity is assigned to each edge going into a source node, or out from a target node. The capacity assigned to _each edge_ is larger than the capacity of the cut $(\{s_1,\ldots s_k\}, V \setminus \{s_1,\ldots s_k\})$, and therfore the maximum flow in the graph.

**Claim 1**: There exists a one-to-one mapping between a flow $f$ in $G$ and a flow $f'$ in $G'$.
**Proof**: we will use the following claim to prove claim 1.

**Claim 2**: If $f$ is a valid flow in $G$, then $f'$, defined as
$f'(u,v) = \begin{cases}
f(u,v) & (u,v) \in E\\
\sum\limits_{z\in V} f(v,z) + f(z,u) & otherwise
\end{cases}$

is a valid flow in $G'$.

**Proof**: 

1. capacity: For edges $(u,v) \in E$: Since both the capacity and the flow are equal to those in $G$, iff the capcity constraint was observed in $G$ it is observed in $G'$. (The case $(u,v) \notin E$ will be discussed shortly).
1. Symmetry: For existing edges, symmetry is trivial since both $u,v \in V$ and so $f(u,v)=-f(v,u) \Leftrightarrow f'(u,v)=-f'(v,u)$. The case $(u,v) \in E' \setminus E$ will be discussed shortly.
1. Conservation: In the case that both $u,v \in (V \setminus \{s_1,\ldots s_k\}) \setminus  \{t_1,\ldots t_\ell\}$, $u$ and $v$ have exactly the same incoming and outgoing edges in $G'$ as they did in $G$ and therefore $\sum\limits_{u\in V'} f'(u,v) = \sum\limits_{u\in V} f(u,v) =0$.

We now move to discuss the cases that were left out for each rule.
1. Capacity: The _added_ edges were constructed with sufficiently large capacity, a capacity which is larger than the capacity of a cut in the graph, therefore the flow on them as defined above does not exceed their individual capacities.
1. Flow conservation: in $G$, $s_i$ were allowed to violate the conservation rule since they were sources. in $G'$ they are not considered sources. each $s_i$ has, in $G'$ the same outgoing edges it had in $G'$, and by construction these edges have the same flow values as in $G$. The conservation law is satisfied by setting $f'(s,s_i) = -f'(s_i,s) = \sum\limits_{v \in V} f'(s_i,v)$. This flow does not violate capacity constraints since the capacity $c'(s,s_i)$ is large enough and it does not affect the conservation of any other node (except $s$, which is "exempt" from the conservation rule).  
1. Symmetry: the symmetry for each node is implied in the construction. Whereever I defined $f'(u,v)$ I implied $f'(u,v)=-f'(v,u)$.

This concludes the proof of claim 2.
To prove Claim 1, it remains to show that the construction is unique. Looking at the added edges and the values of the flow, they are completely determined by information that is in $G$ and therefore they are unique for a given flow graph $(G,f,c)$.

I am not detailing the proof with regards to $\{t_1,\ldots t_\ell\}$ since it uses exactly the same arguments.



