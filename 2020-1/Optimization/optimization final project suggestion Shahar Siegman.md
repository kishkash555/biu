## Car maneuver solver

The car maneuver problem is the problem of taking a car from an initial position-and-orientation A into a final position-and-orientation B using steering and forward/backward inputs. If obstacles need to be avoided in order to prevent damage to the car or to surrounding installations, then the problem becomes constrained. If further to that we are seeking a path which is optimal in some sense (such as shortest path, or using least steering input, or a combination of these), then the problem becomes a constrained optimization problem.

The next sections will give a brief and tentative description of the proposed geometric model and the resulting mathematical optimization problem.

### Car geometry and movement
The geometry of a car can be approximated by a rectangle. The state of the car can be described by a triplet $(x,y,\theta)$, where $x$ and $y$ signify the location and $\theta$ the current orientation, which can be defined as the angle of the central longitudinal axis relative to the x axis. The reference point on the car is fixed to the middle of the rear wheels' contact with the ground. 

When the car is steering, systematic skidding behavior can only be avoided if the center of the rotation circle is on the line connecting the rear wheels' contact points. We assume [Ackermann steering geometry](https://en.wikipedia.org/wiki/Ackermann_steering_geometry) which allows skid-free movement for all 4 wheels.

Except for the constraints imposed by surrounding obstacles, the car movement is contrained by its minimum turning radius $R_{min}$.

### Modeling the car in state-space
In order to arrive from State A = $(x_A,y_A,\theta_A)$ to State B = $(x_B,y_B,\theta_B)$, The car must move continuously through its state space. Description of the problem using a finite number of optimization parameters requires piecewise approximation, which can be most easily achieved by a spline model. Thus the problem becomes setting the parameters of  $N$ intermediary points $(x_i,y_i, \theta_i),\ 1 \le i \le N$ in a way that adheres to the constraints and is optimal. It will be assumed that the the streeing input is piecewise constant, i.e. the wheels are at a constant angle (relative to the car's body) between points. 


### Kinematic Constraints
The following constraints will be used. Not that each constraint involves just one or two adjacent points:
* The change of the car's orientation $|\theta_{i+1}-\theta_i|$ is limited by the distance traveled between points $\sqrt{(x_{i+1}-x_i)^2+(y_{i+1}-y_i)^2}$ (if both sides are squared this becomes a simple quadratic constraint)
* The travel distance between subsequent points will be limited in order to avoid excessive misrepresentation of the true path's arc between the points.

### Obstacle modeling
The geometry of the obstacles will be modeled by a certain number of rectangles parallel to the axes. The problem of prohibited  
