### Q1
Define segments on $[0,1]$
$$
S_1 = [0,\frac{1}{2}]\\
\\
S_2 = [0,\frac{1}{4}] \cup [\frac{1}{2},\frac{3}{4}]\\
\\
S_3 = [0,\frac{1}{8}] \cup [\frac{1}{4},\frac{3}{8}] \cup [\frac{1}{2},\frac{5}{8}] \cup [\frac{3}{4},\frac{7}{8}]\\
...\\
S_n= \bigcup_{l=0}^{l=2^{n-1}-1}[\frac{2l}{2^n},\frac{2l+1}{2^n}]
$$
and $A_n$ will be: $A_n: x \in S_n$
$$
\mathbb{P}(A_n) = \mathbb{P}(x \in S_n) = |S_n| = \frac{1}{2}\\
\forall k\lt m, \mathbb{P}(A_k|A_m) = \mathbb{P}(A_k \cap A_m)/\mathbb{P}(A_m)\\
=|S_k \cap S_m|/|S_m|\\
|S_k \cap S_m | = \big(\bigcup_{l=0}^{l=2^{m-1}-1}[\frac{2l}{2^m},\frac{2l+1}{2^m}]\big) 
\cap
\big(\bigcup_{l=0}^{l=2^{k-1}-1}[\frac{2^{m-k}2l}{2^m},\frac{2^{m-k}(2l+1)}{2^m}]\big)
$$
