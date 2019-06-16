$softmax = \begin{pmatrix} \frac{e^x}{e^x+e^y+e^z}, & \frac{e^y}{e^x+e^y+e^z}, & \frac{e^z}{e^x+e^y+e^z} \end{pmatrix}$

if the training sample belongs to the 2nd class, the CE  loss will be:
$CE\ Loss = -\log \frac{e^y}{e^x+e^y+e^z}$

and the _gradient_ of this function is a vector of the partial derivative as follows:
$\begin{pmatrix} \frac{\partial}{\partial x} -\log \frac{e^y}{e^x+e^y+e^z} &
\frac{\partial}{\partial y} -\log \frac{e^y}{e^x+e^y+e^z} &
\frac{\partial}{\partial z} -\log \frac{e^y}{e^x+e^y+e^z} \end{pmatrix}$

Calculating the derivative in the first entry:
$\frac{\partial}{\partial x} -\log \frac{e^y}{e^x+e^y+e^z}=
\frac{e^x+e^y+e^z}{e^y}\cdot\frac{e^y}{(e^x+e^y+e^z)^2}\cdot e^x = 
\frac{e^x}{e^x+e^y+e^z}$

The last entry has the same structure and therefore:
$\frac{\partial}{\partial z} -\log \frac{e^y}{e^x+e^y+e^z}=\frac{e^z}{e^x+e^y+e^z}$

Now the one in the middle:
$\frac{\partial}{\partial y} -\log \frac{e^y}{e^x+e^y+e^z}=
\frac{e^x+e^y+e^z}{e^y} \cdot \frac{(e^x+e^y+e^z)e^y-(e^y)^2}{(e^x+e^y+e^z)^2} = 1- \frac{e^y}{e^x+e^y+e^z}$

So, if we call the 3 components of the softmax vector _x_, _y_, _z_:

$softmax = \begin{pmatrix}softmax_x, & softmax_y, & softmax_z \end{pmatrix}$
We see that the derivative of the loss function based on the $\log$ of the 2nd term is:

$-\nabla \log softmax_y = \begin{pmatrix}softmax_x, & 1- softmax_y, & softmax_z \end{pmatrix}$

Which "happens to be" the distance of the probabilities from the "true" probability vector (0,1,0)