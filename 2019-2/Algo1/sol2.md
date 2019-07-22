# Algoirthms 1
### Assignment 2 Solution
### July 21, 2019
### ID 011862141
----

## Question 1

#### Algorithm description
For simplicity, assume n is an integer power of 2 so $n=2^m$

1. In set-up stage, pivot the array on the n/2, n/4, ..., 4th, 2nd, 1st elements. This means that the entire array is pivoted about its median, then the first half of the array is pivoted around *its* median (which is the $\frac{n}{4}$'th element of the entire array), and so forth.

1. In the query stage, take the corresponding part of the array (e.g. if k=11 then you need the part of the array which includes elements 8-15), and find the $(k-n_0)$'th element in that array, where $n_0=2^{\lfloor\log k\rfloor}$, the order statistic of the first element of the selected part of the array.

#### Time complexity
##### Setup
Finding the median is performed with BFPRT.
- Finding the median and pivoting on n elements: O(n)
- Finding the median and pivoting on n/2 elements: O(n/2)
...
- Finding the median and pivoting on 1 element: O(1)

Total: O(n)+O(n/2)+...+O(1) = O(2n) = O(n)

##### Query
- Calculating the subarray indexes: O(1)
- Size of subarray: O(k) (requires proof, which is done by showing the start and end indexes of the subarray are $2^{\lfloor\log k\rfloor}  \ge \frac{k}{2}$ and $2^{\lceil\log k\rceil}  \le 2k $, so the difference is O(k)) and applying BFPRT on it to find the $k-2^{\lfloor\log k\rfloor}$'th element is therefore O(k)



## Question 2

#### Obersvation
The principle of using FFT to solve the PM problem is that, for each offset, if the pattern and texts are encoded using 1's and 0's a match between ones (in the pattern and in the input text) would add 1 to the "score" of that offset. Since we know the length of the pattern, we know what the "full score" possible is, so all we have to do is make sure we let one value correspond to 1, the other values must be zero.

#### Algorithm description
```
s: initialize array with 3xn elements

for each character c in the alphabet {0,1,2}:
    a = array [1..n] with 1 in i'th element if T[i]==c, else 0
    b = array [1..n] with 1 in i'th element if P[i]==c, else 0
    reverse the order of the elements in b
    s[c,:] = polynimial_multipliaction(a,b) # same as algorithm we saw in class
end for

return sum of the rows of s s[0,:]+s[1,:]+s[2,:] 
```
 
The above algorithm calcualtes the total number of matches for each offset, finding the number of mismatches is by substracting the length of the pattern, m, from each entry.

Correctness: From the Observation above, since the only elments in a and b are 1's and 0's, the only possible results of multiplying them are 0 and 1, with 1 obtained when both entries are 1. This corresponds, by our construction, to identical characters in the pattern and text.

## Question 3

#### Observations
1. Since $A(x)\Big\rvert_{x_0}=r$, then $(A(x)-r)\Big\rvert_{x_0}=0$ Which means $(A(x)-r)$ is exactily divisible by $x-x_0$
1. Long division can be applied to polynmials, and costs $O(n)$ time.

#### Algorithm description
1. Evaluate $y_0 = A(x)\Big\rvert_{x_0}$ 
2. set $r=y_0$
3. Calculate $A'(x) = A(x)-r$ # by editing 1 entry
4. q: Calculate the long division $\overline{A'(x)}| x-x_0$
4. return *r* and *q*
#### Correctness
Follows from definition of division

#### Asymptotic time analysis
1. Evaluating a polynomial of bounded degree $n$: O(n)
1. Change A to A': O(1)
1. Calculate long division of a polynomial of bounded degree n with polynomial of degree 1: O(n)

Total: O(n)


## Question 4
_Comment: Unfortunately it is almost midnight. I will do my best to describe what I found_.

From Question 3 we have: $A(x) = q_0(x)(x-x_0)+y_0$
We know that $A(x_i)=y_i$, 
so we can write:
$q_0(x_i)(x_i-x_0)+y_0 = y_i$
From which we can isolate:
$q_0(x_i) = \frac{y_i-y_0}{x_i-x_0}\equiv y'_k,\ i=1\ldots n$

This formula is a "prescription" for calculating the coefficients of A(k) by successive shifts and summation:

$q_0$ is a new polynomial, with bounded degree $n-1$, for which we have $n-1$ points and their corresponding values. The subscript denots the fact that a new $q$ polynomial would be created in each iteration.


```
Interpolate(x,y):
    # P provided as n (xi, yi) pairs
    d = |x| - 1 # the degree of the polynomial
    if d = 0:
        return array of length 1 with value y0
    x',y' = empty arrays of length d
    # Calculate P' as follows
    For i=1 to d:
        x'[i-1] = x[i]
        y'[i-1] = (y[i]-y[0])/(x[i]-x[0])
    A = interpolate(x',y') 
    initliaze A' array of 0s, size d+1
    
    for i=1 to d:
        A'[i] = A[i-1] # promote all entries i.e. multiply by x
        A'[i] = A'[i] - x[0]*A[i]
    A[0] = A[0] + y[0]
    return A
```

While it requires a more methodical proof, the structure of the algorithm suggests time complexity $O(n^2)$.