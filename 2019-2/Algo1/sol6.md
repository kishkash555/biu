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
    5          end for
    6       k = find_median(S)
    7       S = Partition(S, k)
    8       reorder V and W according to the same index mapping as S
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
In a sorted array, we would take the top items by "specific value", up to a point where the knapsack is full - this is the "cutoff". If we knew the cutoff, we could traverse the array in O(n) to take everything above the cutoff and leave everything below it (using a fractional quantity from the item exactly on the cutoff to fill the knapsack exactly).
Since we don't know the cutoff, we can look for it recursively, by "guessing" the median of the specific values as the cutoff, then recursively pick the half where the cutoff is. If the cutoff is in the upper half, that means we should take all elements in the lower half.

### Run time analysis
* The loop in lines 3-5 taken O(n)
* `find_median(S)` (line 6) can be done in O(n) using BFPRT '73
* `Partition(S,k)` (lines 7-8) can be done in O(n)
* The implied loop in line 9 takes O(n/2)
* The recursive call is on O(n/2)
* The rest of the operations are O(1)

$T(n) = O(n) + T(n/2)$
$T(n) = O(n)$


## Question 3

### Proof A. G is Bipartite &rArr; G can be colored by two colors
We color the left side with one color, and the right side with the other color. 
