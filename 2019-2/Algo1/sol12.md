# Homework Assignment 12
### Algorithms 1 Summer 2019
### ID: 011862141
#### Aug 24th, 2019
---



## Question 3

#### B: $f(X,Y)=-f(Y,X)$
$f(X,Y) =$
$= \sum\limits_{x\in X}\sum\limits_{y\in Y} f(x,y)$ by definition.
$= \sum\limits_{x\in X}\sum\limits_{y\in Y} -f(y,x)$ by law of symmetry of flow.
$= \sum\limits_{y\in Y}\sum\limits_{x\in X} -f(y,x)$ finite sums may be reordered
$= -\sum\limits_{y\in Y}\sum\limits_{x\in X} f(y,x)$  
$= -f(Y,X)$ by definition  


#### A: $f(X,X)=0$
$f(X,X) =-f(X,X)$ Shown in B above
$\Rightarrow f(X,X)=0$

#### C: $f(X \cup Y,Z)$

$f(X \cup Y,Z)=$
$= \sum\limits_{x\in X \cup Y}\sum\limits_{z\in Z} f(x,z)$ by definition.
$= \sum\limits_{x\in X}\sum\limits_{z\in Z} f(x,z) + \sum\limits_{x\in Y}\sum\limits_{z\in Z}f(x,z)$ by definition of disjunction and provided $X \cap Y = \empty$
$=f(X,Z)+f(Y,Z)$ by definition of $f$.


multiplying both sides by -1 we get
$-f(X \cup Y,Z)=-f(X,Z)-f(Y,Z)$
$f(Z,X \cup Y)=f(Z,X)+f(Z,Y)$


## Question 4: multi-source multi-targe flow problem
#### Explanation
The idea here is that the difference between a source/target node and a "through" node is that a source node is "allowed" to break the conservation law. 
On the other hand, if we add an edge from _s_ into a specific node, with conceptually infinity capacity, we allow the sum of outgoing flow from that node to increase without restriction (except for the normal capacity constraint).
The same discussion can be applied to target nodes.

#### Construction of G'=(V',E')
$V' = V \cup \{s,t\}$
$E' = E \cup \{(s,s_i)| i=1,\ldots,k\} \cup  \{(t_i,t)| i=1,\ldots,\ell\}$
$c'(u,v) = \begin{cases}
c(u,v) & (u,v) \in E\\
\sum\limits_{v\in V} 100\cdot c(s_i,v) & v \in \{s_1,\ldots s_k\} \vee v \in \{t_1,\ldots t_\ell\}\\
\end{cases}$

In words: the capacity is copied for edges that were copied from _G_. A large enough capacity is assigned to each edge going into a source node, or out from a target node. The capacity assigned to _each edge_ is larger than the capacity of the cut $(\{s_1,\ldots s_k\}, V \setminus \{s_1,\ldots s_k\})$.

**Claim 1**: There exists a one-to-one mapping between a flow $f$ in $G$ and a flow $f'$ in $G'$.
**Proof**: we will use the following claims to prove claim 1.

**Claim 2**: If $f$ is a flow in $G$, then $f'$, defined as
$f'(u,v) = \begin{cases}
f(u,v) & (u,v) \in E\\
\sum\limits_{z\in V} f(v,z) + f(z,u) & otherwise
\end{cases}$

is a valid flow in $G'$.

**Proof**: 

1. capacity: For edges $(u,v) \in E$: Since both the capacity and the flow are equal to $G$, iff the capcity constraint was observed in $G$ it is observed in $G'$. (The other case will be discussed shortly).
1. Symmetry: By construction $(u,v) \in E \Leftrightarrow u,v \in V$. Therefore $f(u,v)=-f(v,u) \Leftrightarrow f'(u,v)=-f'(v,u)$. The case $(u,v) \notin E$ will be discussed shortly.
1. Conservation: In the case that both $u,v \in (V \setminus \{s_1,\ldots s_k\}) \setminus  \{t_1,\ldots t_\ell\}$, $u$ and $v$ have exactly the same incoming and outgoing edges in $G'$ as they did in $G$ and therefore $\sum\limits_{u\in V'} f'(u,v) = \sum\limits_{u\in V} f(u,v) =0$.

We now move to discuss the added edges.
1. Capacity: The added edges were constructed with sufficiently large capacity, a capacity which is larger than the capacity of a cut in the graph, therefore the flow cannot exceed that capacity.
1. Flow conservation: in $G$, $s_i$ were allowed to violate the conservation rule since they were sources. in $G'$ they are not considered sources. each $s_i$ has, in $G'$ the same outgoing edges it had in $G'$, and by construction they have the same flow value as in $G$. The conservation law is satisfied by setting $f'(s,s_i) = -f'(s_i,s) = \sum\limits_{v \in V} f'(s_i,v)$. This flow does not violate capicity constraints since the capacity $c'(s,s_i)$ is large enough and it does not affect the conservation of any other node (except $s$, which is "exempt" from symmetry). This 
1. Symmetry: the symmetry for each node is implied in the construction. Whereever I defined $f'(u,v)$ I implied $f'(u,v)=-f'(v,u)$.

This concludes the proof of claim 2.
To prove Claim 1, it remains to show that the construction is unique. Looking at the added edges and the values of the flow, they are completely determined by information that is in $G$ and therefore they are unique for a given flow graph $(G,f,c)$.

I am not detailing the proof with regards to $\{t_1,\ldots t_\ell\}$ since it using exactly the same arguments.



