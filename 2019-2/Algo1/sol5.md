# Homework Assignment 5
### Algorithms 1 Summer 2019
### ID: 011862141
#### July 30th, 2019
---

## Question 1
### pseudocode
```
1 huffman_linear_time(f):
2    g := initialize empty double-linked list
3    while not empty(f) or not empty(g):
4        if empty(g) or f.peek_first().weight < g.peek_first().weight:
5            a = f.next()
6        else:
7            a = g.next()
8        if empty(g) or f.peek_first().weight < g.peek_first().weight:
9            b = f.next()
10       else:
11           b = g.next()
12       z = init_node(left_child=a, right_child=b, weight= a.weight + b.weight)
13       g.insert_last(z)
14   end while
15   return z // this is the root
```

### Explanation
1. We treat the sorted array f as a list. `f.next()` returns the next elment of f (this is equivalent to keeping an index and incrementing it with every call to `next()`)
1. In contrast to the baseline algorithm, we have two queues. One is `f` from which we only extract, the other is `g` to which we insert the new nodes, and extract from it when a new node is minimal.
1. In each pass of the loop we extract two nodes and insert one, so the loop runs $n$ times.
1. `g` is automatically sorted since `f` is sorted. So when two elements are combined and inserted to the end of `g`, they are guaranteed to be larger than the any two elements previously inserted into `g`. The elements are inserted to the end, so the head of `g` has the smallest weight of the elements in `g`.
1. In the `if`-clause, I used `peek_first()`, it is a method that returns the first element of the list but does not remove it.



## Question 2
### Simple circle definition
A simple circle is a path from a node to itself, which doesn't use the same edge twice.
### m = n-1 
Graphs with this property have zero circles. Explanation: let's construct this graph gradually. We start with N nodes and zero edges. This graph contains N connectivity components (since each node is a separate connectivity component). At the end of the process we must have 1 connectivity component. So each edge we add must reduce the number of cc by one, since no edge addition can reduce the number of cc by more than one. So one "end" of each edge has to connect to a node  that was in a different cc. In order to form a circle, at least one edge has to connect two nodes that already had a path between them, so we proved there cannot be a circle in this graph.

### m = n
Graphs with this property have exactly one simple circle. 
Similarly to the proof of the previous case, but in this case we have a "budget" of 1 edge that can connect two edges that were already connected (i.e. had a path between them). Without loss of generality, we assume this edge is the one we add last. The nodes at the end of this edge have only one other path between them, so we have exactly one circle. Any edge added to a fully connected graph with n-1 edges will be between two nodes that have a single path between them, so there cannot be _less_ than one circle.

### m = n + 1
Let's denote $\overrightarrow{AB}$: A "direct" edge $\{A,B\}$ and $\overset{\smile}{AB}$ a "nondirect" path from $A$ to $B$ (that includes at least one additional node). Also, $V \in \overset{\smile}{AB}$ will denote node V is along that path from A to B.

Continuing the argument from the previous case, we have a graph with exactly one circle and we have an extra edge we can add.

Let's denote the group of nodes on the existing circle $C_1$.

We now add an additional edge $\overrightarrow{AB}$:
* if $A \in C_1,\ B \in C_1$: That means there were previously two paths $\overset{\smile}{AB}_1$ and $\overset{\smile}{AB}_2$. Now there are 3 circles: 
$\begin{matrix}
\overset{\smile}{AB}_1 \cup \overset{\smile}{AB}_2, & \overset{\smile}{AB}_1 \cup \overrightarrow{AB}, & \overset{\smile}{AB}_2 \cup \overrightarrow{AB}
\end{matrix}$

* if $A \notin C_1,\ B \notin C_1,\ \overset{\smile}{AB}\cap C_1 = \empty$: There are only two circles: $C_1$ and $\overset{\smile}{AB} \cup \overrightarrow{AB}$. They are disjoint (the circles do not have common nodes).
* if $A \notin C_1,\ B \notin C_1,\ \overset{\smile}{AB}\cap C_1 \ne \empty$: There are three simple circles:
    * $C_1$ as before
    * $\overset{\smile}{AB}\cup \overrightarrow{AB}$. 
    * We can take the union of the _edges_ participating in the two circle, and remove just the edges that are in both circles, to arrive at a third simple circle. 
* If $A \in C_1,\ B \notin C_1$: It is similar to the previous case since we know $\overset{\smile}{AB}\cap C_1 \ne \empty$ since it contains at least one element: $A$


Conclusion: in this case we have either two or three circles.


## Question 3
### A - MST with $w \in \{1,2\}$
We use Kruskal's algorithm. instead of sorting we create two lists, one for the 1-weight edges and the other for the 2-weight edges. This construction is $O(E)$ as opposed to $O(E\log E)$ with ordinary sorting.  We go over the first list first.
### B - MST with $w \in [n]$
I assume $n=|E|$ and each weight appears exactly once. 
The sorting is trivial since we immediately know the index each edge should go in the sorted array. We need to keep an additional variable, which holds the index of the last visited cell in the array.

## Question 4
### A
This statement is true. By way of contradiction, let T be an MST s.t. $e_1 \notin T$. Due to connectivity, one of $e_2 \ldots e_n$ is in T. 
Denote $e_i = \{v,v_i\}$. 
We'll break the problem into cases:
* If $\deg_{T}(v)=1$ Then $v$ must be either a leaf in T, or a root with only one child. 
    * If it is a leaf, we can build a tree T' by removing edge $e_i$ from T and replacing it with $e_1$, making $v$ a leaf hanging from $v_1$ as its parent node. We maintained connectivity and tree-ness and the weight is now smaller by $e_i-e_1$. Since weight(T') < weight(T) we have a contradiction
    * If $v$ is a root with just one child $v_i,\ i \ne 1$ then we first change T so that $v_i$ is the root and $v$ is a leaf with $v_i$ as the parent. We can then construct T' as in the previous case and reach a contradiction again.
* If $\deg_{T}(v)\gt 1$ then $v$ is the root of some subtree. We build T' by detaching $v$ from its parent (thereby reducing the total MST weight by some $e_i \gt e_1$) and re-connecting it as a child of the node $v_1$. We maintained tree-ness and reduced the weight by $e_i-e_1$, again contradicting the optimality assumption.
This proves that every MST must include $e_1$.

### B
This statement is false. let $e_2 = \{v, v_2\}$. Let's assume $v_2$ has degree 5, all edges from $v_2$ have different weights and at least one edge has weight less that $e_2$. Following the exact same arguments we stated for $v$ in sectoin A of this question, this time for $v_2$, we can show that in that case $e_2 \notin MST((G,E))$

## Question 5

### A

$A^2_{i,j} = \sum\limits_{k=1}^{n} A_{i,k}A_{k,j}=\sum\begin{cases} 1 & (i,k) \in E \wedge (k,j) \in E\\
0 &otherwise \end{cases}$.

This is the number of paths between $i$ and $j$ through exactly one other node.

### B
This is the number of paths from $i$ to $j$ through exactly $k$ nodes.

### C
$A^n = A^{n/2}A^{n/2}=(A^{n/4}A^{n/4})(A^{n/4}A^{n/4})= \cdots$

We perform $O(\log(n))$ squaring operations, each time using the product as the operand for the next step.


