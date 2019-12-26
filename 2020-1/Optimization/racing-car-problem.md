## Introduction
A race track is a curved road on which cars race. In a *time trial* session, each car races by itself to achieve the best possible timing. 
The driver uses steering and accelaration/ decelaration inputs to control the car's position and attitude relative to the track's centerline and its speed. To successfully complete the track while keeping the car under control, the driver must consider the following limitations:

* The car must remain on the paved road.
* When negotiating a curve, the speed is limited by the turning radius, to avoid skid due to loss of static friction. 

Additionally, the car has some unavoidable physical limitations:
* The acceleration (rate of change of speed) is limited by the engine's and brakes' power.
* The car has a top speed (e.g. due to bearing failure)
* The car has a minimum turning radius (i.e. tightest possible turn).
* The rate of change of steering angle is also limited (either by the driver's or car's response times).

An additional important physical effect is that traction improves during acceleration, and degrades when decelarating. Therefore, reducing the speed *before* entering a curve allows reaching higher speeds inside the curve.

Therefore, optimizing the driver's inputs is a nontrivial optimization problem with a few inherent tradeoffs. 

## Proposed model
The guiding principle when developing the model was mathematical simplicity. Whereever possible, 1st- or 2nd-order approximations were used. While the real-world problem is continuous in space and time, both the geometry and the decision variables are described using a finite number of variables, by dividing the track into $N$ sections and assuming constant characteristics per section (or at the endpoints of sections). As always with discretization, when more variables are used to define the same problem, the approximation error can be reduced.

### Track's geometry
A track's segment curvature is $k_i,\ i \in 1\ldots N$. The centerline's equation is $y=k_i\cdot x^2,\ x \in [0,1]$ in a coordinate system $C_i$ attached to segment $i$. We require the track's curve to be $\mathbf{C^1}$,  which means that relative to $C_i$, the next section's coordinate system is located at $(1,q_i)$ at a rotation angle $\phi$ given by 

$$tan \phi = \frac{dy_i}{dx_i}|_{x=1}=2q_i$$. 

The track's width is assumed constant, $n$.

### Car's path geometry
The car's path is similarly modeled as a sequence of parabolas, "seamed" together to ensure 1-st order continuity. The car path's curvature at each segment will be represented by $q_i$. The car's initial position is aligned and on the track's centerline.

For each section, the car traces a parabolic path $a_ix^2+b_ix+c_i$ which is expressed in terms of *the coordinates of the same segment*. 

Let's find recursive expressions for the parameters $a_{i+1},\ b_{i+1},\ c_{i+1}$
(i.e. based on $a_i,b_i,c_i,k_i$):
* $a_{i+1}=q_{i+1}$ (for small rotation angle there is no change in the quadratic coefficient).
* The derivative at $x=0$ in segment $i+1$ should equal the derivative at $x=1$ in segment $i$, after rotation by the track's angle at the segment's contact point, $2k_i$: $b_{i+1}= 2a_i+b_i-2k_i$.
* By equating the distance from the road centerline curve at the end of segment $i$ to the distance at the beginning of segment $i+1$, we get $c_{i+1} = a_i+b_i+c_i$.

These expressions are all first-order approximations based on $k_i \ll 1,\ q_i \ll 1$

Summarizing, we have:

$$a_i=q_i\\
b_1=c_1=0\\
b_i=2\sum \limits_{j=1}^{i-1}(q_j-k_j)\\
c_i=\sum \limits_{j=1}^{i-1}(b_j+q_j)=\sum\limits_{j=1}^{i-1}2(i-j-1)(q_j-k_j)+\sum\limits_{j=1}^{i-1}q_j\\
$$


These expressions will allow us to describe the constraints compactly, in terms of a,b, and c, while in the implementation these constraints will be applied to the decivision variables $\{q_i\}$ using the formulas above.

### Car's speed
The decision variables $u_j,\ j \in 0 \ldots N$ will denote the *reciprocal* of the car's instantenous speed at the endpoints of the track's sections. We will assume the car starts from rest, i.e. $u_0=0$.

## Formulation of optimization problem
### Objective function
The objective is top speed, or min travel time. The travel time in segment $i$ is approximated as $(u_{i-1}+u_i)/2$. Therefore the objective is:

$$\min \sum\limits_{i=1}^N (u_i+u_{i-1})/2= \min \frac{1}{2}u_N+\sum \limits_{i=1}^{N-1}u_i$$


The track's centerline is a curve in 2D space. We require the curve be $\mathbf{C^1}$ i.e. with a continuous first derivative. We want the track's geometry to be specified section by section.

### Constraints
* $u_i \gt u_{\min}$ The minimum travel time to complete a segment
* 