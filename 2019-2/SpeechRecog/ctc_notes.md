- Together, these outputs define the probabilities of all possible ways of aligning all possible label sequences with the input sequence.
The total probability of any one label sequence can then be found by summing the probabilities of its different alignments.

symbols:
- $\mathbf{x}$ input, sequence of m-dimensional real-valued vectors
- $\mathbf{z}$ training labels, sequence of labels (letters from an alphabet)