
$$
W = \big[ \begin{array}{cccc}
   w_{11} & w_{12} & ... & w_{1j}\\
   \vdots & & & \vdots \\
   w_{i1} & w_{i2} & ... & w_{ij}
  \end{array}
  \big]\\
$$
$$
\mathbf{Z} \equiv x\mathbf{W}+\mathbf{b}
$$

$$
\frac{\partial Z_k}{\partial W_{ij}} = 
\begin{cases}
0, & k \neq j\\
x_i, & k=j
\end{cases}
$$

$$
\frac{\partial Z_k}{\partial b_{j}} = 
\begin{cases}
0, & k \neq j\\
1, & k=j
\end{cases}
$$


$$
-\frac{\partial loss}{\partial W_{ij}}= -\hat y_j x_i (\sum y_k) + x_iy_j \\
\frac{\partial loss}{\partial W_{ij}}= x_i (\hat y_j  - y_j)
$$