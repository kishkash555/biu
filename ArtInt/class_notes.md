# Artificial Intelligence 
----
___Nov. 1 2018___

#### Complexity notation
- b - branching factor
- m - depth of tree (not depth of solution)

A few points on comple
DFS has better memory complexity than BFS

#### IDS iterative deepening
Check up to a certain depth, if found solution great, otherwise go deeper.

The memory complexity of IDS is b*d rather than $b^d$ in BFS.
It is not BFS even with L =1. BFS would discover a row.
This Dijkstra. requires priority queue.

we aren't going into memory complexity.

worked out in class example from exam 2014 moed bet

- DFS: 
    - open:
    - closed:

- IDS:
    - open: S | SABC | SABCDG
    - closed: S | SABC | SADG

- UCS (Dijkstra):
    - closed: SCBBDFAG
    - open: s (0)
    A 6
    B 2
    C 1
    B 4
    F 5
    E 7
    D 4
    E 8
    D 6
    E 10
    F 9
    G 6
    G 26



### lecture 3 informed search
slide 4 a good heuristic:
- goes down as you near the goal
- admissible: the computation of the heuristic should be simpler than the true calculation
- 