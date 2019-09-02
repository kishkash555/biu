# Homework Assignment 7
### Algorithms 1 Summer 2019
### ID: 011862141
#### Sep 4th, 2019
---

## Question 1 half-connected graph

#### Suggested algorithm
1. Compute $G^{SCC}$, The DAG representing strongly connected components of $G$.
2. Sort $G^{SCC}$'s nodes by topological order
3. starting from $v_1$, the first node by topological order, if there are edges between each consecutive pair of node  $(v_i,v_{i+1})$ then return True. else return False


#### Correctness
**Claim:** If U, V are two different SCCs of G, then for all $u\in U,\ v\in V$ there is a path $u \leadsto v$ iff there is a path $U \leadsto V$ in $G^{SCC}$
**Proof  &rArr;:** Denote the path from U to V in $G^{SCC}$ as $P=(U=U_1,U_2,\ldots,U_k=V)$.
By definition of $G^{SCC}$:
$\exists u_1^{out} \in U_1, u_2^{in} \in U_2\ s.t.\ (u_1^{out},u_2^{in}) \in E$ 
$\exists u_2^{out} \in U_2, u_3^{in} \in U_3\ s.t.\ (u_2^{out},u_3^{in}) \in E$ 

and similarly form all pairs of consecutive nodes in path $P$.
By definition of GCC, there exists a path in G $u_i^{in} \leadsto u_i^{out}$ for all _i_ from 1 to k.

Combining these paths, we get the path from $u\in U_1$ to $v \in V$

**Proof  &lArr;:** Assume there is no path from U to V in $G^{SCC}$ and there is a path $P=(u_1=u,u2,\ldots u_k=v)$ in G from $u\in U$ to $v \in V$. Then denote $u_{k'} \in U'$ the first node not in U in P. Then by definition, $(U,U')\in E^{SCC}$. denote the first node not in $U \cup U'$ as $u_{k''}\in U''$. Then by definiton $(U',U'')\in E^{SCC}$. We continue this argument until we reach V, so we have shown a path from U to V in $E^{SCC}$ which contradicts the assumption.

We know that in a DAG that was topologically sorted, there are no edges pointing left. So if two nodes $(v_i,v_{i+1})$ have no edge between them, there is no path between them, since the path can neither go through $v_j$ where j > i+1 nor through $v_j$ with j < i. According to the claim, this means that G is not half-connected.


