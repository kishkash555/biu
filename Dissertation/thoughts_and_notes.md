## Thoughts and notes
---

Jan 23

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

---

Feb 27

This paper also led me to an important observations: 
Suppose we start with a large, multi-purpose, pre-trained ANN such as BERT. We intend to use BERT for a particular task of relatively limited scope, such as question answering for a specific document corpus. We are looking for an ad-hoc compression of the large network, that will remove redundancies without negatively affecting its performance on the particular task.

We choose an input set, which represents the task, and run it through BERT. We select a specific "hidden" layer and look at the values at the outputs of this layer, for all the inputs. If two (or more) of these intermediate values are highly correlated, that means that they convey similar information and can be compressed into a single value. Conceptually, we're performing a Principle Component Analysis on an hidden layer and trying to replace the redundant representation with a more compact one.


---

LSTMs have a few one-layer MLPs. So figuring out leaner MLPs can help with LSTMs as well. The final layer is not a prediction, it is gates. So interpretation of 

The techniques so far - from literature and "original"
* repeating coefficients using hash function - not clear on why this works, and the paper does not provide much insight
* using human engineered/ SGD-computed "vocabularly" of features. need to understand why you can't just let the network figure out the attributes itself. 
* finding a rotation of the existing matrix so as to render some of the output irrelevant, then dropping the corresponding columns - this suggestion was not made in literature, it might not work for output layer
* Need to see if the batch normalization can tell us more than good intuitions - can we directly utilize these ideas?
 
---

March 25 2019
#### From reading
* "soft target" are easier to train - why? my guess: they "parallelize" the search (they don't fight over borders, they settle for ambivalence in areas that cannot eventually be decided)

#### Other ideas
* The linear layer concept: linear layers are "redundant" because if they can in succession they can be merged. However an architecture where a linear layer is "available" as input to all layers may help

* The convolution idea: forget about matrix multiplication. think of the operation as a convolution between a spaced-out feature vector and a step function with the values of the different features interlaced. This turns the whole matrix into a single function with the $z$'s appearing as the values of the function. This makes a natural coarsening/refining axis (the details of the function along the axis). This may be an interesting mathematical framework to explore.

* LSTM is just an MLP with a constraint of repeating (all) coefficients between layers :)


#### From today's experiments
* Convincing demonstration of how cross entropy pushes all the layers towards nonlinearity and hardens the margins
* very interesting how SGD can hurt performance on the train data.
* would be interesting to look at the rate of change. Think of the "active set" theory and how it can be used to test any compaction on a much smaller set than the whole train.
* Do the large magnitudes lead to less stability (in training of cases near the margin)?
* redundancy of the representation is directly evident





conclusions from today's experiments + stuff to present
