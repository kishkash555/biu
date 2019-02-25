
This paper also led me to an important observations: 
Suppose we start with a large, multi-purpose, pre-trained ANN such as BERT. We intend to use BERT for a particular task of relatively limited scope, such as question answering for a specific document corpus. We are looking for an ad-hoc compression of the large network, that will remove redundancies without negatively affecting its performance on the particular task.

We choose an input set, which represents the task, and run it through BERT. We select a specific "hidden" layer and look at the values at the outputs of this layer, for all the inputs. If two (or more) of these intermediate values are highly correlated, that means that they convey similar information and can be compressed into a single value. Conceptually, we're performing a Principle Component Analysis on an hidden layer and trying to replace the redundant representation with a more compact one.
