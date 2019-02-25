Generally, NN produce D functions. In classification and other tasks, however we consolidate these functions until we have a mapping of each point in input space to a single integer (this may be a very large integer).

The original data used some training set. even if we don't have access to that training set, we know that it was finite, and that the learning process was able to (at most) encode the correct integer in each point of space occupied by a training example. 

Errors in the dev/ test set indicate that one or more of these happened:
1. there were errors in some sample labels of the training set
2. The NN over-divided the space and created bogus borders which cannot be supported with the training data.
3. The NN under-divided the space. it also failed to model the training data.
4. The NN made incorrect interpolations into areas where no training data existed, letting the wrong class dominate.


Any of these errors is an indication 
There should be a known scale- the jnd of the input space. That scale can vary in space. We have two upper bounds for the global mesh:
1. the number of training points
2. The actual results of the NN.
 
We can use multigrid methods to realize a new training set based on the results of the classifier. We will then need to find a new method to train the classifier. That method may have not been good enough on the original data, since we don't know how to handle generalization. But it may be good enough given a classifier which does not have a generalization requirement.