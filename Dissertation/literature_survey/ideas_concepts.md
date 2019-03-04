we want correlated features to have weights between them. the weights help decorrelate the noise, but it's possible that I will prefer some correlation in the noise the correlation matrix of the signals might be different than the correlation matrix of the noise.

if i have no idea about the structure, then the best guess for initial features would be combinations of one, two, three etc. of the input features in both positive and negative signs. this amounts to $O(n^3)$ features in the first layer (n in the number of input features). this is prohibitively large. also this does not allow for any nonlinearity. this assumes that the scale of all inputs is similar, which can be synthetically achieved via a (pre-initialized) diagonal matrix as the first layer. this matrix does not handle decorrelation, it just roughly centralizes and scales the data so that combinations of &pm;1 are successful initail guesses for the useful features. Even if we were to start with such a large feature matrix, many of the features may not be useful. if we are more cheap and start with, say just a few random pairs and completely avoid triplets and higher orders, then deeper layers would be able to weigh in and combine the pairs in a meaningful way, into 3- and 4-way combinations and so forth. Even if each layer is restricted to combining pairs, they still "wield more power" since each of their inputs represents more than one (raw) input.

The nonlinearity allows us to squash outliers and treat them similarly to non-outliers at the edge of the feature range. If we want to single out an intermediate segment of a particular coordinate, or have a multi-lobe distribution which we want to reorder, it would require multiple oblique outscalings to reach separation; in that case, it would be a better design (especially if we are aware of the lobe pattern and suspect that the natural ordering is not the most useful for the classification task) to replace (or augment) the variable with a nominalized version and use that version to train embeddings.

If the relation between the inputs and the classes in not pathological, we can expect to be able to represent the classification task with a number of sparse matrices according to the following rule:

$$B^d\gt N$$

The first hidden layer should be at most $N^B$ wide; this width would allow it to represent all conceivable B-way interactions between variables. Can we expect the number of high order-interactions to decrease exponentially with the interaction order? since we postulated that they are required as a means to decrease noise, and we believe the number of latent useful indicators for a specific problem is smaller than N, the answer is definitely yes.

## Approaches to describing (and analyzing concepts) in MLPs

### rotation, scaling, clipping
Singular Value Decomposition shows that you can view any matrix as performing 3 operation: rotation (multiplication by a unitary matrix), scaling (with some of the singular values possibly 0) and another rotation, in the new coordinates. More simply, and less accurately, this can be viewed as a single rotation followed by shrinking/ stretching of some axes.

### projections
This is a more robust approach. We think of each column (if the vector is on the right) as a _feature_, or a vector in the space of the input. For each input, we calculate the projection by performing a dot product between the input vector and the column. We can have more output dimensions than input dimensions, but in that case we know that, barring the nonlinearity, there will be linear dependence between outputs.

### functions
here we think of the entire transformation leading to a single output neuron as a way to construct an elaborate function of the inputs. we have m functions of n variables, all combinations of nested, scaled and shifted tanh's of the original n variables.

### image
This might be forced, for inputs that are not image-like, but it's important to note that it's not far-fetched that some non-image inputs will be visualizable as images

### structure
the transformations can be viewed as stiffness matrices and the outputs as displacements. requires more development.




