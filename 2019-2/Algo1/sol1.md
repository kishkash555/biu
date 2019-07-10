## Question 1
#### Observation
$\begin{pmatrix}
g(n+2)\\
g(n+1)\\
g(n)\\
\end{pmatrix} = 
\begin{pmatrix}
1&0&2\\
1&0&0\\
0&1&0\\
\end{pmatrix} 
\begin{pmatrix}
g(n-1)\\
g(n-2)\\
g(n-3)\\
\end{pmatrix} 
$
#### Algo
let $M=\begin{pmatrix}
1&0&2\\
1&0&0\\
0&1&0\\
\end{pmatrix}, 
\quad V=\begin{pmatrix}
3\\
2\\
1\\
\end{pmatrix}$


    calc_g(n):
        if n < 4:
            return n
        g = M^(n-3) * V
        return g[1]

Where the exponent $M^{n-3}$ is calculated using the exponentiation algorithm learned in class.


## Question 2
#### Observations
* An array with 0 flips is a sorted array because $\forall i \lt j:  A[i] \lt A[j]$
* Let's say we have two sorted sub-arrays $R[-n_1], \ldots R[0],\ L[1], \ldots L[n_2]$ (such as we have during an intermediate step of _merge-sort_). Then the number of flips can be counted in $O(n_1+n_2)$: When sorting the array, each time the cursor on the left is in position $-j$ (i.e. on the j'th element from the right) and we pick an element $R[k]$ as the next smallest element, it is an indication of $k$ flips involving $R[-j]$ and each of $R[1] \ldots R[k]$.
* If we split the array to two sub-arrays, the number of flip-pairs where one member is from the left array and the other is from the right array does not depend on the internal order of the left or right arrays.


Employing these observations, the suggested algorithm is as follows:


    Count_flips(A):
        split A equally (or nearly-equally) into A1 and A2
        (sorted, f) = recurse(A1,A2)
        return f

    recurse(A1, A2):
        if len(A1) == len(A2) == 1:
            if A1 <= A2:
                return ([A1,A2], 0)
            return ([A2,A1], 1)
        split A1 equally into A11 and A12
        split A2 equally into A21 and A22

        (B1, f1) = recurse(A11, A12)
        (B2, f2) = recurse(A21, A22)

        f = 0
        C = empty array, length: len(B1) + len(B2)
        traverse B1 and B2, using cursors r1 and r2 respectively:
            r1 = r2 = 1
            if B1[r1] <= B2[r2]:
                C[r1+r2-1] = B1[r1]
                r1 = r1 + 1
            else:
                C[r1+r2-1] = B2[r2]
                r2 = r2 + 1
                f = f + len(B1) - r1
        end traverse
        return ([B1, B2], f + f1 + f2)




Note: the algo assumes $A$'s length is a power of 2. $A$ can be padded from the left with $\infty$'s until its size is exactly a power of two.


## Question 3
#### Observation:
* A 2x2 board with one missing tile can be tiles with L-pieces (using one L-piece)

Claim: 
* if we know how to tile a k &times; k board with one missing tile in an arbitrary location, we can tile a 2k &times; 2k board which has one missing tile in an arbitrary location.

Proof by induction:
* Induction base - see Observation
* Induction step
    * Let Q1, Q2, Q3 and Q4 be the four k &times; k quadrants of the board.
    * Place an L-piece in the center, covering 3 corners of 3 of the quadrants, leaving one corner uncovered: the uncovered corner belongs to the quadrant with a missing tile. 
    * Now each quadrant has either one missing or one covered tile, but is half the size
We can continue this until the board is divided into 2 &times; 2 parts.

#### Algo
recursion along the lines of the induction above


    cover_with_L_pieces(k, missing_tile_row, missing_tile_column):
        mQ = quadrant where the missing tile is located (one of 1, 2, 3, 4)
        first_piece = [(k/2,k/2), (k/2+1, k/2), (k/2, k/2+1), (k/2+1,k/2+1)] 
        remove from first_piece the coordinate that lies in quadrant mQ
        placed_pieces = empty list
        placed_pieces.append(first_piece)

        placed_pieces = concatenate(placed_pieces, 
                cover_with_L_pieces(k/2, missing_tile_row mod k/2, missing_tile_col mod k/2))

        for each quadrant q in [1, 2, 3, 4] except mQ:
            placed_pieces = concatenate(placed_pieces, (cover_with_L_pieces(k/2, k or 1, k or 1))
        
        return placed_pieces

Notes:
1. In order to get the correct covering, the coordinates returned from the call to _cover_with_L_pieces_ for each quadrant need to be converted to the coordinates relative to the bigger board, i.e K added to the rows in quadrants 3 and 4 and to the columns in quadrants 1 and 4.

2. _concatenate_ receives two lists and returns a new list which is a concatenation of the two input lists.


## Question 4
#### Observations
* Since the possible members are positive integers limited by $10n^{1.5}$, we could construct the output as a binary group-membership array.
* We could then traverse the array in a double loop, calculating $a+b$ (the sum of the current elements). Cell $a+b$ in the binary membership array is set to 1 (to represent $a+b \in A+A$). If the entry is already 1 it is left untouched.
* Finally, we traverse the output array, outputing the index of each set bit.

BUT this is $O(n^2)$, due to the double loop. We can actually do better using the FFT algo learned in class:

* We observe that when multiplying two polynomials, an element $x^w$ has a nonzero coefficient if there was at least one pair of elements $x^y,\ x^z$ in the original polynomials with nonzero coefficients such that $w=y+z$ 
* If more than one such pair exists, and we still want to be sure that $x^w$ has a positive coefficient, we can do that by assuring the original arrays have no negative coefficients (to avoid possible cancellations when summing positive and negative components). This means that if the original coefficients are not negative, a non-zero coefficient in the output will exist if (and only if) there was at least one corresponding pair in the inputs.
* we can represent group membership (of a group $A$ of positive integers) by a polynomial as follows:

$P = \sum \limits_{p \in A} 1\cdot x^p$

i.e coefficients of 1 or 0 depending on whether $p \in A$



#### Algo
    group_self_addition(A):
        B = initialize an array of size 10*n^1.5
        for a in A:
            set B[a] = 1 
        C = B*B # polynomial multiplication using FFT

        R = new empty list
        for i=1 to len(C):
            if C[i] > 0:
                R.append(i)
        return R


The most time consuming line of code is the FFT. it is $O(n^{1.5}\log n)$. The two loops are $O(n)$ and $O(n^{1.5})$.


