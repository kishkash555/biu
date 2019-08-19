# Homework Assignment 11
### Algorithms 1 Summer 2019
### ID: 011862141
#### August 20th, 2019

<div dir='rtl'>
הערה: הקובץ מוקלד באנגלית כדי להמנע מעריכה דו-כיוונית (ימין לשמאל ושמאל לימין)
</div>

---

## Question 1
My suggestion is based on following the logic of Floyd-Warshal.
The "naive" solution is re-running FW for the entire graph, i.e. $n+1$ nodes. This has a runtime of $O(n^3)$.

Now we try to save time, based on finer observations about the "gaps" i.e. the effects of running FW while ignoring the last node.
When _v'_ is introduced into the graph, there are two gaps:
1. $\forall j,d_{v'j}=d_{jv'}=\infty$ i.e. the recursion's base case was not applied to the edges touching _v'_. 
1. $\forall j,d_{v'j}^{k}$ and $d_{jv'}^{k}$ were not updated in the loop for $k=1,\ldots,n$.

Let's denote the elements of _D_ (before the introduction of _v'_) as $d_{ij}$ and _D'_ (after the introduction of _v'_) as $d'_{ij}$.

**Observation**: $\forall i,j\textrm{ s.t.} 1\le i \le n, 1\le j \le n,\ d_{ij}=d'_{ij}$

**Proof**: Induction on the recusion formula of WS algorithm. Base: $w_{ij}=w'_{ij}$. Induction step: If $d_{ij}^{(k-1)}=d_{ij}'^{(k-1)}$ Then $d_{ij}^{(k)}=d_{ij}'^{(k)}$ follows directly from the formula.


#### Pseudo code
FW-ADD-NODE(W,$v'_{in}$,$v'_{out}$):
1. Copy the entries of $d_{ij}^{(k)}$ into a new $(n+1) \times (n+1)$ matrix _D'_
1. $d_{\cdot,n+1}'=\textrm{concat}(v'_{in},0)$
1. $d_{n+1,\cdot}'=\textrm{concat}(v'_{out},0)$
1. for k=1 to n+1:
1. &emsp;for i=1 to n+1:
1. &emsp;&emsp;$d_{i,n+1}' = \min\{d_{i,n+1}',d_{i,k}+d_{k,n+1} \}$
1. &emsp;&emsp;$d_{n+1,i}' = \min\{d_{n+1,i}',d_{n+1,k}+d_{k,i} \}$
1. for i=1 to n+1:
1. &emsp;for j=1 to n+1:
1. &emsp;&emsp;$d_{i,j}' = \min\{d_{i,j}',d_{i,n+1}+d_{n+1,j} \}$


#### Explanation
- In line 2 we use the weights provided in $v'_{in}$ to initialize the column in _D'_ that represents path lengths to _v'_. The concatanation of the zero is since $d_{n+1,n+1}$ is the weight from _v'_ to itself, which is zero by definition
- In line 3 we use the weights provided in $v'_{out}$ to initialize the row in _D'_ that represents path lengths from _v'_.
- in lines 4-8 we perform the loop on values of _k_ that was "missed" by the entries in the last row and last column
- In lines 8-10 We perform for all the nodes in the graph the _n+1_ -th iteration on value of k in order to find paths between every two nodes that pass through _k_.

#### Runtime analysis
- The implicit loop in line 1: $O(n^2)$
- The implicit loops in lines 2 and 3: $O(n)$
- The double loop in lines 4-7: $O(n^2)$
- The double loop in lines 8-10: $O(n^2)$

Total: $O(n^2)$, so we improved on the naive solution.

## Question 2
#### Mapping of the exchange rate matrix into a weighted graph
If we have a "path" of exchanges  $P=(p_1=i,p_2,\ldots,p_k=j)$ starting from 1 unit of currency $i$ and ending with currency $j$, the total amount of currency $j$ we will have is: $\prod\limits_{m=1}^{k-1} r_{p_m,p_{m+1}}$

We are interested in the maximum over all possible paths from _i_ to _j_. We will define a weight function: 
$$w_{ij} = -\log r_{ij}$$
The transformation is well defined since $r_{ij} \in \mathbb{R}^+$. We get $w \in \mathbb{R}$. This will lead to the following identity:
$$R_{ij}^{(p)}=\prod\limits_{m=1}^{k-1} r_{p_m,p_{m+1}} = \exp(-\sum\limits_{m=1}^{k-1} w_{p_m,p_{m+1}})$$

I used $R_{ij}^{(p)}$ to denote the "total exchange rate" from one currenct to another along a specific path $p$. 
 Since the exponent is a monotonouly increasing function, that means that a maximum of the expression inside the exponent on the r.h.s is also the maximum of the product on the l.h.s. It is also the _minimum_ of the sum inside the exponent (dropping the negative sign).



#### Part A - Finding the path for maximum of each currency
Using the transformation defined above, we can set up the corresponding graph with $w_{ij}$ as edge weights and calculate **the shortest path** from node _i_ (The efficient algorithm for this is Bellman-Ford).
The path lengths calculated by the algorithm can be transformed to the amount of each currency using the same transformation as above, namely: 
$\forall j,\ V[j]= \max\limits_{p} R_{ij}^{(p)} = \exp(-\delta_{ij})$.

The running time using Bellman-Ford is $O(n^3)$ (since in a dense graph $|E|=|V|^2$).


#### Part B - Detecting arbitrage cycles in a fully-connected graph
We note that with the mapping above, arbitrage cycles exist if and only if there are negative cycles in the graph. We also note that any node is reachable from any node since there are edges between any two nodes.

#### Negative cycle detection pseudocode
DETECT-NEGATIVE(G=(V,E),w):
1. Pick an arbitrary node s
1. Run Bellman-ford with node s as root.
1. for $(u,v) \in E$
1. &emsp; if d[v] > d[u] + w(u,v):
1. &emsp; &emsp; return TRUE
1. return FALSE


#### Explanation
For every node reachable from _s_, once d[u] is updated for some node _u_, it will always be real-valued $d[u]\in \mathbb{R}$. Having a negative cycle means the the condition 
$\forall u \in V:\ d[u]=\delta(s,v)$ cannot be met since $\exists u.\forall x \in \mathbb{R}:\ \delta(s,u) \lt x$.
So we know that at least one update will occur if we run relax again after running Bellman-Ford. So the condition for updating _d_ in RELAX (line 1 in the pseudo-code) will not be satisified for at least one edge on the path from _s_ to _u_.

## Question 3
#### Explanation/ Intuition
We assume without loss of generality the existence of a single triangle. If there is more than one triangle, additional triangles after the first one detected are ignored.

The need to run in linear time suggests a greedy/ elimination algorithm. The naive approach of splitting into two, does not work (at least not directly) since the triangle's nodes can be on both sides of the cut, and if that happens we may not find a triangle in either subgraph. Splitting to 3 also doesn't help, since the triangle may have a node in each subgroup. Spliting the graph nodes to four groups, enables us to eliminate 1/4 of the nodes as stated in the following claim:
**Claim**: When splitting a graph that has a triangle into four disjoint parts, then the triangle nodes will reside in up to 3 of the parts.
**Proof**: Since a traingle has 3 nodes, and the groups are disjoint, the 3 nodes can belong to at most 3 groups.

#### recursive formula


$split\_nodes4(G=(V,E)) =
\begin{Bmatrix}
\begin{array}{c|c}
\begin{Bmatrix}
V_1,V_2,V_3,V_4
\end{Bmatrix}
& 
\begin{matrix}
\bigcup_i V_i =V, \\
\forall i\ne j,\ V_i \cap V_j = \empty,\\
\forall i\ |V_i| \ge (|V|-3)/4
\end{matrix} 
\end{array} 
\end{Bmatrix}
$


$split\_graph4(G)=\{\begin{array}{c|c}\{G_1,G_2,G_3,G_4\} &
\begin{matrix}
G_1= \textrm{subgraph of }G \textrm{ on } V_2 \cup V_3 \cup V_4\\
G_2= \textrm{subgraph of }G \textrm{ on } V_1 \cup V_3 \cup V_4\\
G_3= \textrm{subgraph of }G \textrm{ on } V_1 \cup V_2 \cup V_4\\
G_3= \textrm{subgraph of }G \textrm{ on } V_1 \cup V_2 \cup V_3\\
\end{matrix}
\end{array}\}
$
Where
$\{V_1,V_2,V_3,V_4\} = split\_nodes4(G)$


$isolate\_triangle(G) = \begin{cases}
\empty & |V| \lt 3 \vee |V|=3 \wedge \neg A(G)\\
G & |V|=3 \wedge A(G)\\
isolate\_triangle(G_1) & A(G_1)\\
isolate\_triangle(G_2) & A(G_2)\\
isolate\_triangle(G_3) & A(G_3)\\
isolate\_triangle(G_4) & else
\end{cases}$
Where
$\{G_1,G_2,G_3,G_4\} = split\_graph4(G)$

#### Pseudocode
1. isolate_triangle(G=(V,E)):
1. if $|V| \le 3$:
1. &emsp; if A(G): return G
1. &emsp; else: return $\empty$
1. $\{G_1,G_2,G_3,G_4\}= split\_graph4(G)$
1. if $A(G_1)$:
1. &emsp;$isolate\_triangle(G_1)$
1. else if $A(G_2)$:
1. &emsp;$isolate\_triangle(G_2)$
1. else if $A(G_3)$:
1. &emsp;$isolate\_triangle(G_3)$
1. else:
1. &emsp;$isolate\_triangle(G_4)$

#### Run time
1. Split_graph4: $O(|V|)$ (adjacency list)
1. Running algorithm _A_ in lines 6, 8, 10, 12: $4\cdot f(\frac{3}{4}|V|)$

Recursive formula setting $|V|=n$:
$T(n) = O(n)+ 4\cdot f(\frac{3}{4}n) + T(\frac{3}{4}n)$
Since $f(n) \in \Omega(n)$:
$T(n) = O(f(n)) + T(\frac{3}{4}n)$
$$T(n) = O(f(n))$$


## Question 4

Recall that in BMM, $Z=X\cdot Y \Leftrightarrow (Z_{ij} = 1 \Leftrightarrow \exists k. X_{ik}=1 \wedge Y_{kj}=1)$

For transitive closure, $(u,v) \in E^* \Leftrightarrow \exist u_1,u_2,\ldots,u_k.(u,u_1)\in E \wedge (u_1,u_2)\in E \ldots \wedge (u_2,v)\in E$.

This suggests the intuition that we need to consturct a graph such that:
1. Every path contains at most 3 nodes
1. A path between a node $u_i$ and a node $v_j$ exists iff $Z_{ij}=1$

#### Graph construction
Given $X,Y \in \mathbb{R}^{n \times n}$, Our graph will consist of 3 
groups of _n_ nodes:
##### Nodes
$V= I \cup K \cup J$
$I=\{I_1,\ldots,I_n\}$
$K=\{K_1,\ldots,K_n\}$
$J=\{J_1,\ldots,J_n\}$

#####Edges
Edge $I_\ell \rightarrow J_m$ existsts if $X_{\ell m}=1$. In mathematical notation, $(I_\ell, J_m) \in E \Leftrightarrow X_{\ell m}=1$.
Similarly, An edge $J_\ell \rightarrow K_m$ exists iff $Y_{\ell m}=1$.

There are no other edges in the graph.

**Claim**: $(I_{\ell},J_m) \in E^* \Leftrightarrow Z_{\ell m}=1$
**Proof**: By construction, there are no direct edge from nodes in $I$ to nodes in $J$. A path $I_{\ell} \leadsto J_m$ can exist iff there is at least one node $K_k \in K$ such that $(I_\ell,K_k) \in E \wedge (K_k,J_m) \in E$. By construction, such edges exist iff $X_{\ell k} =1 \wedge Y_{km}=1$ which is exactly the required and sufficient condition for $Z_{\ell m}=1.\ \tiny{\square}$


#### Algorithm description
1. Construct a graph $G=(V,E)$ with $3n$ nodes as described above
1. Run _B_ on _G_ to get E*, the edges in the transitive closure of G.
1. Initialize $Z_{ij} =0$ for all $1\le i \le n,\ 1 \le j \le n$.
1. for i = 1 to n:
1. &emsp; for j = 1 to n:
1. &emsp;&emsp; if $(I_i, J_j)\in E^*$: 
1. &emsp;&emsp;&emsp; $Z_{ij}=1$

#### Correctness
Follows directly from the claim

#### Running time
1. Graph construction: $3n$ to create nodes plus $(3n)^2$ to create an adjacency matrix. 
1. Adding the edges based on the entries in $X,Y$ is $2n^2$
1. Since $G$ has 3n nodes, B runs in $g(3n)$.
1. Initialization of $Z$ takes $n^2$ operations.
1. The double loop in lines 4-7 takes $n^2$ time. ($E$ represented by an adjacency matrix)

The total run time is $g(3n)+O(n^2)$. Since the transitive closure's output may have up to $O(n^2)$ entries, it means that $g(3n) \ge O(n^2)$ and therefore the total run time is $O(g(3n))$, as required.





