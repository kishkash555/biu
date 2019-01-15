Shahar Siegman 011862141

# Methods in DL Assignment 4 Report

Summary:
* The chosen article is "Decomposable attention model" (Parikh et al. '16)
* The Maximum score reached on the training and dev set was above the 80% mark, but short of the authors' reported 89.5%
* The best score was achieved with slight modification of the network and inputs. The modification contributed significantly both to the max score on the dev set and to the convergence speed. 
* The model training time was about 14 hours.
* Several changes and tweaks were attempted, they are discusses in more details in section X of this report.

## I. Article overview, network architecture and innovation
The authors chose a three-submodule architecture which is based on _attention_ (by now, a well-established approach), along with other standard techniques (e.g. vector concatenation and shallow MLPs). The main innovation is in implementing an LSTM-free _attention_ model. This allows intra-sentence parallelization at test time i.e. words in the sentence can be evaluated in parallel. No parallelization scheme is suggested for training. The asymptotic complexity of network feedforward (i.e. prediction) is analyzed by the authors and found to be $O({ld^2+dl^2})$. The first term originates from the (constant) number of matrix-vector multiplications per word while the second originates from the implementation of the attention. When disregarding parallelization, the asymptotic complexity is on par with LSTM-based attention models. Contrary to LSTM-based models, the current model is agnostic to word order in each sentence. An extension, described in a followup article by the same authors, handles word order through within-sentence attention, and achieves slighltly better scores on the SNLI leaderboard.
The following sub-sections describe very briefly the four modules of the network proposed and tested in this article.

### 0. Dimension reduction matrix
A trainable purely linear transformation, reduces the dimensions of the input embedding vectors (300d) to a desired smaller dimension, such as 200.

### 1. Attend
The attention is achieved by "converting" the first- and second- sentence word embedding vectors, $a_i$ and $b_j$ respectively into $\alpha _j$ and $\beta _i$ through:
![attention_formulas1.png](attention_formulas1.png)

Where $e$ is a matrix defined by:
![attention_formulas2.png](attention_formulas2.png)

This can be thought of as multplying the original embedding vectors (or more precisely, transformed versions of the original embedding vectors), then using softmax to normalize the results in a way that assures each word in sentence A "sees" a well-defined "probability" over the words in B, and vice versa.

### 2. Compare
After "softly aligning" the words between the two sentences, the network is ready to perform the main task - compare. The intuition that the authors provide, is that if there are sufficient analogies between words in the first and second sentences, that means they describe a similar situation - and therefore chances are greater that one entails the other. In the authors' words:
![intuition.png](intuition.png)

Comparison is effected using a shallow MLP.

### 3. Aggregate
The final step is more technical. Since the previous step resulted in $l_1 + l_2$ vectors, a third step is needed to "unwind" this information into a 3 way decision: neutral, entailment, or contradiction. This is achieved using a third shallow MLP.


## Implementation process
### Network structure
The implementation was based solely on the description in the article. The article is mostly clear regarding the network topology and design decisions. Two points were ambiguous:
* The article states they used networks of "2-layers". I was not sure whether this means only one transformation (matrix multiplication + nonlinearity) is due, or more (two or perhaps even three, if "2-layer" is actually "2 hidden layers"). I decided to experiment with the topologies and choose what worked best. The final configuration I chose was of a single hidden layer, i.e two transformations.

* The article stated that the layers were "each with 200 neurons". However the structure of the _compare_ and _aggregate_ networks calls for them to have an input twice as large, since it is fed with the concatentation of two vectors output by the previous module. I had the choice of:
	* shrinking the last layer of the _attend_ and _compare_ phases to 100
	* expanding the input layer of the _compare_ phase to 400, then using 200 as the dimension of the subsequent layers
	* setting _compare_ input and hidden layer size to 400, then reducing the size to 200 in the subsequent layer.
	* and similarly for the _aggregate_ network.

### Other parameters and design decisions
#### Learning rate and matrix initialization
Regarding learning rate, the optimal initial learning rate according to the authors is 0.05, but there is no specification of the stepsize scheme. I used the basic "SimpleSGDTrainer".

There is also no mention of the initialization scheme for the transformation matrices, so I chose the Dynet's default (Glorot).

#### Manual and automated autobatching
The structure of the attend phase, where each word from both sentences is fed to the same MLP, seems natural for batching, by concatenating the individual training vectors to a matrix and feeding the matrix to the MLP. I implemented this using dynet's "manual batching" scheme. To my disappointment, no speedup was observed following this effort.

Applying dynet's auto-batching scheme yielded an immediate speedup of approx. 30%.

#### GLOVE loading speedup
In order to speedup GLOVE vector loading, and avoid unnecessary memory overhead when training, I filtered the GLOVE embedding input file so it contains only vectors for words that are found in the train, dev and test files.

#### running on GPU
In order to increase the capacity for experiments, and for long training, I set up an Amazon EC2 instance with an NVIDIA Tesla V100 GPU. I compiled dynet with CUDA as described in the documentation. The results were disappointing:
* The run time on the GPU was about 2.5 times _slower_ than on my laptop's CPU, and 3 times slower than running on the CPU of the Amazon instance
* Not only that, the same models which performed well when run locally on a PC, performed much worse in terms of convergence, in fact they did not seem to improve accuracy at all over the training. 

All results presented here were obtained on a CPU. I can provide further information on the attempts to run on GPU if needed.

## Experiments
### Initial results and tuning the network
The network was run for 5 epochs (about 2.75 million examples, about 6 hours wall-clock time) and reached a score of 69% on the dev set. In several subsequent runs, I experimented with the following:
* Changes to the network sizes and depths as described above in "network structure".
* Changes to the learning rate scheme: Adagard, Cyclical, Adam and Simple were tested
* dropout: no dropout, 20%-27% dropout.
* batch size: 1, 4, 10
* Dimension reduction matrix:
	* Complete removal the dimension-reduction matrix (phase 0)
	* Applying a nonlinear activation (ReLU) to the output

### Improving the attend module by replacing the activation function
The authors used ReLU as the activation layer - this seems to hold for all modules' networks. However, the GLOVE embedding vectors values are randomly distributed around 0 - this holds for all 300 coordinates. This means that the first layer of the attend network, with its ReLU activation, will set to zero, on average, half the coordinates of each input embedding vector. This can be avoided, if the network learns a positive, large enough bias term, but such progress seems unlikely as the gradient will not contain the necessary information to make the leap from a small, near-zero bias to a large positive bias.

So in order to improve network's results, I tried the following:
* Scale all embedding vectors by a constant, so all their coorinates are less than 1 (The constant chosen is 0.05)
* Replace the ReLU with a $\tanh$ activation.

This change was the single most successful improvement to the network. It significantly accelerated convergence and allowed to reach higher score. The highest score (82%) was obtained using this scheme.
### batch size
I experimented with different batch sizes, but saw no differences - neither run time nor convergence ratio were affected.

### Final configuration
My final configuration is different than what the article described in several aspects. This will be further discussed in the following section.

Best configuration - 82% accuracy on dev, 81% on test:
* A 300 to 200 linear dimension reduction matrix
* 200-node Attend network with one hidden layer, $\tanh$ activation.
* SimpleSGDTrainer with initial learning rate of 0.05
* Embedding vectors scaled by 0.05 
* Compare network input dimension 400, hidden layer of 200 and output of 200.
* Aggregate network input dimension 400, hidden layer of 400, and output of 3.
* Dropout rate of 0.25
* batch size 10

## Conclusion
### General
A proposed network was implemented based solely on its specification in a journal article. The results that the authors were able to obtain were not successfully recreated. Although the article is well-proofread and uses plain writing style, it is to succint regarding some important implementation details, most notably the dimension of the layers of the different modules. The MLP's depths are specified in terms that might seem clear to readers from within the NLP/DL community, but I felt they were somewhat vague.

### Possible sources of discrepancy with the original authors' work
Possible reasons for not being able to recreate the author's original results:
* Differences in implementation details between dynet, and the framework used by the authors (assuming it was not dynet). For example, learning rate and dropout are not necessarily implemented the same way.
* Differences in initialization scheme
* The authors may have run the network for considerably longer, it is possible (although not probable) that very long training results in better performance
* Randomness - When I ran the same model twice or more, I saw very similar results between the runs. However it is possible that with a very large number of runs, one specific random configuration, relating to the initialization and the order of feeding the training samples, results in better performance.
* Lack of understanding/ missing of important details on my part

### Personal summary
It was interesting to experiment with a network that is actually "world-class" (as opposed to the more limited architectures that we experimented with in the previous assignments). Despite all the difficulties of fully reporducing the results, it is nice to know that a working version of a world-class network can be obtained with reasonable effort. I got the impression that NLP/DL literature is less self-critical than is common in scientific reporting, and also does not put much emphasis on reproduceability. I am still not quite sure regarding how GLOVE vectors interact with ReLU activations and I believe that the choice of activation functions should be made more carefully, and more transparently, especially when deciding between bound and unbound activations. 
 

