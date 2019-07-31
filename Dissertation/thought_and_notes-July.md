maybe some of these observations are interesting enough by themselves.


try to define the search as an algorithmic process. The physics analogies may help.

The features are planes. given the locations of the inputs and their charges, the activation represnts the energy of bringing the plane from inifinity to its locations. This requires a correction term in case the charges do not sum to zero. The energy is for one feature, so energy minimization by itself cannot solve the problem since it doesn't take into account the synergysm between features. 

Regarding the synergysm, one of the more interesting "mysteries" of deep learning is the ability to find synergetic features without explicit interaction between features in the same layer in the search (learning) phase.

It's possible that a big part of the difficulty of learning a deep network is due to the undetermined sign of the delta (in the physics analogy, it corresponds to the sign of the charge). even if the inputs to a layer remain relatively fixed (or drift slowly enough so that the feature planes can track their drift), drifts in the signs would seem harder to track since they may require more agressive shifts in plane locations to track. The signs may fluctuate as deeper layers gradually move from random setup to learned setup. This may also be one of the reasons dropout helps learning deep network. If some of the contributions to delta are masked, the delta sample is more biased, hence faster learning. Although the dominant direction is not guaranteed to be correct for the current input, the more consistent updates may help drive the system to converge, or at least explore other minima faster.

The final layer is set up to reward one-vs-all linear separation. The question of number of layers and features may be correlated to a metric of mixture between points of different types. This metric can be calculated by nearest-neighbor histograms per class. These can be calculated on a sample, to reduce time and memory complexity. The best sample may be of centroids obtained from clustering, or using a geometric indexing structure (R-Tree). 

Additionally, in order to avoid the sign noise from deeper layers, it may be good to pre-allocate signs to the classes at each layer. There are $2^{n-1}$ possible class-sign configurations, so the nearest-neighbor histogram may help decide which classes should be kept together in the more shallow layers ("harder separations"). It is not yet clear how these more highly coupled classes are separated later, but still it is intuitive to presume, especially due to the naive search performed in the current algorithm, that the tasks that are easier should be handled first.

weight normalization is an example of a relatively simple enhancement which should probably enjoy wider adotpion that it enjoy. This "reparametrization" is both intuitive and shown to expedite learning. Its only drawback is its inferior performance to batch-normalization.

A recursive, one-plane-at-a-time solution results in a tree-like separation of the data, as opposed to the matrix shape in the traditional algorithm. This means that the tradeoff of solving subproblems separately vs. using the same geometric entities as separators tends towards the latter, with generalization ability probably suffering if the tree-like approach is adopted.

For most availabe examples - images and sound samples - similar local structures are expeced. So convolution networks must be analyzed using the same tools in order to reach practical conclusions and improve state-of-the-art feed-forward networks.