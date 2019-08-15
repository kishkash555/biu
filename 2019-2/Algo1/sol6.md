# Homework Assignment 6
### Algorithms 1 Summer 2019
### ID: 011862141
#### July 31th, 2019
---

## Question 1
Since this game is a zero-sum game, a player maximizing his/her own score will also be maximizing the _difference_ between his/her score and their rival's. So if a player following the maximization strategy doesn't win the game, it means there was no other startegy that could win.

Therefore, we need to work this problem using the exact same algorithm. Then, once either player has accummulated more than half the points available, the winner is known.


## Question 2

We can solve the problem recursively.
The recursion 


    1    Knapsack_linear_time(V, W, B):
    2       n = |V|
    3       for i=1 to n:
    4           S[i] = V[i]/W[i] # specific value of each element
    5       end for
    6       k = find_median(S)
    7       S = Partition(S, k)
    8       reorder V and W according to the same index mapping as the partitioning of S in line 7
    9       weight_of_better_half = W[n/2+1] + W[n/2+2] + ... + W[n]
    10      if weight_of_better_half < B and weight_of_better_half + W[n/2] > B:
    11         solution= take 100% of all elements in W[n/2+1] ... W[n] and the portion from W[n/2] required for the weights to sum exactly to B
    12          return solution
    13      else if weight_of_better_half > B:
    14          return knapsack_linear_time(V[n/2]...V[n], W[n/2]...W[n],B)
    15      else:
    16          solution = take 100% of all items W[n/2].. W[n]
    17          solution_weight = sum(W[n/2]..W[n])
    18      return union(solution, knapsack_linear_time(V[1]..V[n/2], W[1]..W[n/2], B-solution_weight)


### Explanation
In a sorted array, we would take the top items by "specific value", up to a point where the knapsack is full - this is the "cutoff". If we knew the cutoff, then even without a sorted array, we could traverse the array in O(n) to take everything above the cutoff and leave everything below it (using a fractional quantity from the item exactly on the cutoff to fill the knapsack exactly).
Since we don't know the cutoff, we can look for it recursively, by "guessing" the median of the specific values as the cutoff. If this turns out to be an "overshoot" (i.e. the total weights above the proposed cutoff are too heavy) then we know the cutoff is somewhere in the more valuable half. We recursively pick that half where the cutoff is. If the weights of the more valuable items are less than B, then they all should go in the knaspsack, and the cutoff should be recursively searched for in the half of less-valuable items.

### Run time analysis
* The loop in lines 3-5 taken O(n)
* `find_median(S)` (line 6) can be done in O(n) using BFPRT '73
* `Partition(S,k)` (lines 7-8) can be done in O(n)
* The implied loop in line 9 takes O(n/2)
* The recursive call is on O(n/2)
* The rest of the operations are O(1)

$T(n) = O(n) + T(n/2)$
$T(n) = O(n) + O(n/2) + O(n/4) + \ldots = O(2n) = O(n)$


## Question 3

### Proof A. G is Bipartite &rArr; G can be colored by two colors
We color the left side with one color (blue), and the right side with the other color (yellow). By definition of bipartite graph, no two left nodes touch (have an edge between them) and no two yellow nodes touch.

### Proof B. G can be colored by two colors &rArr; G is Bipartite  
Let's look at all the blue nodes. There is no edge between any two of them (if there were the coloring were invalid). The same can be said on the yellow nodes. Let's place all the blue nodes on the left and all the yellow nodes on the right. This is a valid bipartite graph where any edge is between a left and a right node, and no edge is between two nodes on the same side.


### Proof C. G can be colored by two colors &rArr; There are no odd-length circle in the graph
Assume an odd length circle in the graph. We can number the nodes along the circle, from an arbitrary node on the circle, $V_1,\ V_2,\ \ldots, V_{2k+1}$. We start with $V_1$. Assume (without loss of generality) its color is blue. The next node ($V_2$) must be yellow, due to the edge $(V_1,V_2)$. $V_3$ must be blue, and so forth. So node $V_{2k+1}$ must be blue, but it is connected to $V_1$ which is also blue. This is in contradiction to the assumption that the coloring was valid.


### Proof D. G is Bipartite &rArr; There are no odd-length circle in the graph
We start arbitrarily from the left side, tagging all nodes with edges as $C_{11} \ldots C_{k_1 1}$. We then tag all nodes which connect to nodes tagged with $C_{\cdot 1}$ with $C_{\cdot 2}$. All #2 nodes are on the right, by definition of bipartite graph. We then tag all nodes connected to #2 nodes, as $C_{\cdot 3}$ We allow multiple taggings per node. but we are careful not to use the same edge twice in the same circle. All #3 nodes, are connected to #2 nodes, so they are on the left. We then continue to #4 nodes. #4 nodes cannot coincide with #1 nodes, since #1 nodes are on the left and #4 nodes are on the right. We continue to #5. #5 Nodes may coincide with #1 nodes, but that would indicate a circle of length 4 (#1=#5, #2, #3, #4). #6 nodes will not coincide with #1 nodes. #7 may coincide, for circle of length 6. We can continue this argument to show that only odd-numbered nodes in each circle may coincide with #1 nodes, which leads to the conclusion that even-length circles are possible, but odd-length circles aren't.



