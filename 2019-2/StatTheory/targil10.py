import numpy as np
from scipy.optimize import root_scalar
from scipy.special import digamma
import matplotlib.pyplot as plt

xs = np.array([0.79, 0.67, 1.05, 0.31, 1.37, 0.4, 0.09, 1.25, 1.78])

a1 = xs.mean()
a2 = np.log(xs).mean()
f = lambda theta: digamma(theta/a1) + np.log(theta) - a2

print("a1={:.3}, a2={:.3}".format(a1,a2))


# plot
x = np.arange(0.5,2, 0.02)
plt.plot(x,f(x))
#plt.show()

root_obj = root_scalar(f,bracket=[0.5,2])
print("theta: {:.6} alpha: {:.6}".format(root_obj.root, a1/root_obj.root))

#a1=0.857, a2=-0.441
#theta: 0.95794 alpha: 0.89428