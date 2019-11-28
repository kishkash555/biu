# Homework Assignment 7
### Algorithms 1 Summer 2019
### ID: 011862141
#### Aug 7th, 2019
---

## Question 1 half-connected graph

#### Suggested algorithm
1. Compute $G^{SCC}$, The DAG representing strongly connected components of $G$.
2. Sort $G^{SCC}$'s nodes by topological order
3. starting from $v_1$, the first node by topological order, if there are edges between each consecutive pair of nodes  $(v_i,v_{i+1})$ then return True. else return False


#### Correctness
**Claim1 :** If U, V are two different SCCs of G, then for all $u\in U,\ v\in V$ there is a path $u \leadsto v$ iff there is a path $U \leadsto V$ in $G^{SCC}$
**Proof  &lArr;:** Denote the path from U to V in $G^{SCC}$ as $P=(U=U_1,U_2,\ldots,U_k=V)$.
By definition of $G^{SCC}$:
$\exists u_1^{out} \in U_1, u_2^{in} \in U_2\ s.t.\ (u_1^{out},u_2^{in}) \in E$ 
$\exists u_2^{out} \in U_2, u_3^{in} \in U_3\ s.t.\ (u_2^{out},u_3^{in}) \in E$ 

And similarly for the rest of the path.

By definition of GCC, there exist paths in G, between any two nodes within the same SCC i.e. for  any $u\in U_i$, there exist paths $u_i^{in} \leadsto u$ and $u \leadsto u_i^{out}$. This holds for all _i_ from 1 to k i.e for all SCCs.

Combining these paths, we get the path from $u\in U_1$ to $v \in V$.

**Proof  &rArr;:** Assume there is a path $P=(u_1=u,u_2,\ldots u_k=v)$ in G from $u\in U$ to $v \in V$. Then denote $u_{k'} \in U'$ the first node along the path _P_ that is not in U:
 $u_{k'-1} \in U \wedge (u_{k'-1},u_{k'})\in E \wedge u_{k'} \in U'$. 
 Then by definition, $(U,U')\in E^{SCC}$. 
 We now look for the first node $u_{k''} \in U'' \neq U'$ and use the same argument to deduce $(U',U'')\in E^{SCC}$.
 We continue the construction until we establish a path in $G^{SCC}$ from _U_ to _V_.

 **Claim 2:** In a topological sort of a DAG, if two consecutive nodes $(v_i,v_{i+1})$ have no edge between them, then there is no path in the DAG $v_i \leadsto v_{i+1}$ or $v_{i+1} \leadsto v_i$.
 **Proof:** in a topological sort all edge point rightward therefore any path starting in $v_i$ will go to $v_j,\ j \gt i+1$ and will never come back to $v_{i+1}$.
 a path starting from $v_{i+1}$ similarly must go right and cannot involve $v_i$.

 Combine these claims we reach the following conclusions:
 - In $G^{SCC}$, if there are edges between two consecutive nodes, that means there is a path between any two edges (in one direction), which means there is a path (in one direction) between any two edges in _G_.
 - In $G^{SCC}$, if there is no edge between two consecutive nodes in topological order, then the second of these has no path to the nodes before it (in either direction) and therefore the nodes in _G_ it represents have no path to the nodes in the SCCs before them.

#### Running time
- Computing  $G^{SCC}$ and sorting it - linear in the size of G
- Checking for existence of edges between at most |V| pairs of nodes - linear

The algorithm run time is linear in |V|+|E|.


## Question 2

#### Suggested algorithm
First, let's denote the required output per node as F(v): $F(v) = \min_{v \leadsto u} f(u)$
In the rest of this answer, We use F[] to denote the (updatable) value of F for a node. At the end of a run, F[v]=F(v) i.e. the function stores the true value.

**min_reachable_value(G=(V,E)):**
1. Build $G^{SCC}$ and sort it topologically.
1. for each $W \in V^{SCC}$:
1. &emsp; $F[W] =  \min_{u \in W} f(u)$ # find the minimum value of _f_ for the SCC
1. for each $W \in V^{SCC}$, going in reverse topological order:
1. &emsp; for each $(U,W)\in E^{SCC}$:
1. &emsp;&emsp; if f[U] > f[W]: update f[U] = f[W]
1. for each node $v \in V$ # now "propagate" the values of F back from $G^{SCC}$ to _G_.
1. &emsp;set F[v]  = F[W] where $W \in V^{SCC}$ and $v \in W$
1. return F

#### Description and correctness

Since reachability is transitive within an SCC, we know that for v, u in the same SCC, F[v]=F[u]. Due to F being the minimum of the reachable f's, F is bounded from above by the minimum of f in the SCC. Lines 2-3 initialize F as the minimum of f within its SCC.

Call the current SCC, W. If there are other SCCs reachable from W, the value of F within W will be the minimum among the values of the SCCs reachable from W. So we need to go over the SCC's reachable from W, and make sure we update F[W] whenever we encounter a reachable SCC U that has F[U] < F[W]. However we can use the optimal substructure of min, i.e. we can store the lowest value for a node in that node and update only the node touching it directly. The minimum value will propagate to all the nodes from which it is reachable.

In Lines 4-6 we traverse $G^{SCC}$ starting from the last node by topological order, call it _W_. There are no other SCCs reachable from W, so F(W)=F[W] i.e. the value stored in F[W] is final. For each SCC connected to it, $(U,W)\in E^{SCC}$ the value of F(U) is bounded from above by min(F[U],F[W]), so we update F[U] accordingly.

After making the updates for nodes touching W, we move to the next node of $G^{SCC}$ by reverse topological order. Since all the nodes that are reachable from it are on its right, and they were all finalized i.e. their values propagated left, it now has the correct value and its values can be propagated left as well.


#### Running time
- Line 1: Building $G^{SCC}$ and sorting it takes linear time.
- Line 2-3: Finding the minimum _f_ of the SCC can be done in linear time (in the number of nodes in the SCC). Then updating the nodes to equal the minimum is again linear in the number of nodes. The total number of nodes in all the SCCs is |V| therefore this loop completes in linear time
- Line 4-6: We process each node and each edge in $G^{SCC}$, so again this is bounded by linear time (on the original graph)
- Line 7: Loop over nodes, linear time

The total run time of this algorithm is linear.

## Question 3A - Hamiltonian path in a DAG
#### Suggested algorithm
1. sort the graph topologically
1. starting from the first node in the topological sort, look for an edge to the next node
1. if there is no edge $(v_i,v_{i+1})$ for some _i_, return NONE
1. if the last node is reached, then report $P=(v_1,\ldots,v_n)$ as the hamiltonian path.

#### Correctness
As discussed in question 1, in a topological sort of a DAG, if edge $(v_i,v_{i+1})$ is _not_ in _E_, then there is no path that includes both $v_i$ and $v_{i+1}$ since once $v_{i+1}$ was skipped, no edge can point backwards to it

#### Running time
The run time of this algorithm is:
- Topo sort - linear
- traversing the nodes - linear
Total - linear time

## Question 3b - collect maximum score along edges of a DAG
### Obersvations
Assume the DAG is sorted topologically with node labels $v_1,v_2,\ldots,v_i,\ldots,v_n$
1. $v_1=s$ since it is the only node that has no incoming edge, i.e. it is the only node that can be places without a node on its left. 

1. $v_n=t$ since it is the only node such that has no outgoing edges, the only node that does not need a node on its right.

1. Since all paths in the graph are left to right, and no node except _t_ terminates a path (since all nodes has outgoing edges), all paths can be extended until they include _t_.

1. When looking for a maximum we can use the fact that a part of a maximal path is maximal up to that node.

### Suggested algorithm
We use dynamic programming. Define $M(v_i)$: the total edge weight for arriving from node _s_ to node $v_i$.

#### recurisve formula
$M(v_j)=\begin{cases}
0 & v_i=s\\
\max_{(v_i,v_j)\in E} \{w((v_i,v_j))+M(v_i)\} & else\\
\end{cases}$

The terminal condition is that trivially, the path from _s_ to _s_ contains no edges and has zero sum of edge weights.
The recursive step involves "choosing" an incoming edge from the "current" node $v_j$, adding its weight to the sum, then recursive solving the problem for the previous edge along that path.
M(t) represents the solution i.e. the highest-score path from _s_ to _t_.

#### Dynamic programming
We notice the quantity $M(v_i)$ is accessed every time an edge from $v_i$ is processed, and we maintain an array for M.

Pseudocode:
**max_weight_path(G=(V,E),w):**
1. Sort the DAG topologically and label the nodes $v_1$ to $v_n$
1. Initialize $M[v_1]=0$
1. For i=1 to n:
1. &emsp;for each edge $(v_i,v_j)$:
1. &emsp;&emsp;$M[v_j]=\max(M[v_j],w((v_i,v_j))+M[v_i])$
1. return M[t]

#### Running time
- Topo sort: linear in vertices and edges.
- The loop over the edges Lines 3-5: Linear in number of edges
- Other operations (Lines 2,6): Linear

Overall: Linear


## Question 4A probability
The first child: n options
Second child: n-1 options
...
Last child: 1 option
Total: n! options


## Question 4b probability
We note that in 4A, if the table is round, every ordering has multiplicity of _n_ since the children can be rotated up to _n_ times (with all configurations considered equivalent). therefore n!/n = (n-1)!

## Question 5 Union Bound
$Pr[A_1 \cup A_2 \cup \ldots \cup A_k]=\sum Pr(A_i) -  \sum\limits_{m=2\ldots k} (-1)^m Pr(\bigcap \limits_{V \subseteq\{1 \ldots k\},|V|=m  } A_V) = - \sum\limits_{m=1\ldots k} (-1)^m Pr(\bigcap \limits_{V \subseteq\{1 \ldots k\},|V|=m  } A_V)$
In words: the Probability of the union of the events is equal the sum of the probabilities of the individual events minus the probabilities of all the intersections between pairs of events, plus the probabilities of all triplets, minus all quadruplets etc.

Every level is more qualified than the previous i.e. has smaller probability:
$Pr(A_q \cap A_r)\le \sum_m Pr(A_q \cap A_r \cap A_m)$

Where equality exists iff $Pr(\bigcup_m A_m)=1$.


So we have a series starting with a positive value, alternating in sign, and diminishing absolute value of elements. Therefore the sum of the series is bound from above by its first member.
