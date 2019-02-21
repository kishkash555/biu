## Distilling a Neural Network Into a Soft DecisionTree
https://arxiv.org/pdf/1711.09784.pdf

Distilling a Neural Network Into a Soft DecisionTree. Nicholas Frosst, Geoffrey Hinton, Google Brain Team
Channeling examples to nodes that act as NN layers. Each leaf decides among a smaller number of classes that the entire set of available classes. the flow of an example in the tree provides an explanation/ transparency to its classification

## MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/pdf/1704.04861.pdf
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
17 April 2017
   > This  paper  proposes  a class  of  network  architectures  that  allows  a  model  devel-oper  to  specifically  choose  a  small  network  that  matches the resource restrictions (latency, size) for their application. MobileNets primarily focus on optimizing for latency butalso yield small networks.  Many papers on small networks focus only on size but do not consider speed.


Good "Prior work" section describing the following approaches:
### training small networks
* Flattened networks [16] - fully factorized convolutions
* Factorized networks [34] - factorized convolution + topological connections
* Xception network [3] - scale up depthwise separable filters, outperform inception v3
* Squeezenet [12] - bottleneck approach to design a very small network
* structured transform network [28]
* deep fried convents[37]

### shrinking, factorizing or compressing pretrained networks
* Compression based on product quantization [36],
* Hashing [2]
* Pruning, quantization, Huffman [5]
* Factorization speed up [14,20]
* Distillation [9] Using a large network to teach a small network
* Low bit networks [4, 22, 11]

hard to read without understanding the following terms:
* batchnorm [13]
* depthwise separable convolution

## 4 Compressing Neural Networks with the Hashing Trick
https://arxiv.org/pdf/1504.04788.pdf
Wenlin Chen
James T. Wilson
Stephen Tyree
Kilian Q. Weinberger
Yixin Chen
19 Apr 2015
>   As  deep  nets  are  increasingly  used  in  applications  suited  for  mobile  devices,  a  fundamental  dilemma  becomes  apparent: the  trend  in deep learning is to grow models to absorb ever-increasing data set sizes; however mobile devices are designed with very little memory and cannot store such large models. We present a novel net-work architecture, HashedNets, that exploits inherent redundancy in neural networks to achieve drastic  reductions  in  model  sizes.   Hashed Nets uses a low-cost hash function to randomly group connection  weights  into  hash  buckets,  and  allconnections  within  the  same  hash  bucket  share a single parameter value.  These parameters are tuned to adjust to the Hashed Nets weight sharing architecture with standard backprop during training. Our  hashing  procedure  introduces  no  additional memory overhead, and we demonstrate on several benchmark data sets that Hashed Nets shrink  the  storage  requirements  of  neural  net-works substantially while mostly preserving generalization performance.

This is ref. 2 in Mobilenets article

> Ba & Caruana (2014) [6] show that deep neural networks can be successfully compressed into “shallow” single-layer neural networks by training the small network on the (log-) outputs of the fully trained deepnetwork  (Bucilu  et  al.,  2006)

In this article, the authors discuss a memory-reduction technique. The concepts are developed with ordinary, fully-connected MLPs in mind, but it is mentioned they can be applied to many other ANN architectures (including recurrent networks), and also combined with memory reduction techniques proposed by other authors (some of which are cited and briefly explained). 

The $m \times n$ parameters of the weight matrix $V_{ij}$ between two consecutive hidden layers of sizes $m$ and $n$, is replaced with a vector of $k$ parameters, where $k$ can be decided by the user based on the desired memory-potential performance degredation tradeoff. The usual feed-forward calculation  $z = Va$ is then replaced by 
$z=\sum \limits_{j=1}^m w_{h(i,j)}a_j$
Where $w$ is a vector of length $k$ and 
$h(i,j)$ is a mapping $h:\ m \times n \to k$


The "hashing trick" in the title refers to the fact that $h$, rather than being tabulated in memory, is an instance from a suitable family of hashing functions and therfore takes up $O(1)$ memory. A different hashing function is assigned to each layer.

The modifications required to the SGD training procedure are detailed. 

Other ways to explain or think of this approach:
*  The $m \times n$ weights of the original transfer matrix are only used in fixed, predetermined additive combinations, never  individually. 

* $V$ is populated by only $k$ different values, where $(i,j)$'s  that are mapped to the same hash bin, share the same parameter value.

* This is equivalent to an ordinary MLP where an extra linear layer of dimension $k$ was added between the two layers of the original MLP. The weights in the two extra transfer matrices (before and after the new layer) are all 0's and 1's, and are predetermined (i.e. not learnable). This representation is useful when comparing performance with ordinary MLP architectures, and possibly opens the door for analogies with LSTM architectures with its learnable gates.


## 5 Predicting Parameters in Deep Learning
https://arxiv.org/pdf/1306.0543.pdf
Misha Denil Babak Shakibi Laurent Dinh Marc’Aurelio Ranzato Nando de Freitas,
27 Oct 2014
 > We  demonstrate  that  there  is  significant  redundancy  in  the  parameterization  ofseveral deep learning models.  Given only a few weight values for each feature itis possible to accurately predict the remaining values. Moreover, we show that notonly can the parameter values be predicted, but many of them need not be learnedat all. We train several different architectures by learning only a small number ofweights and predicting the rest. In the best case we are able to predict more than 95% of the weights of a network without any drop in accuracy.

This is referenced in [4] above
Code available on github: https://github.com/mdenil/parameter_prediction

## [6] Do Deep Nets Really Need to be Deep?
Lei Jimmy Ba
Rich Caruana
NIPS 2014
https://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf

Seems like an insightful and highly relevant paper. The authors perform a "shallowing" of existing neural networks and report that learning a smooth function (i.e. the soft outputs of a trained ANN) is faster than learning the hard classifications with the same shallow architecture.

They also point that a matrix between a layer of size $D$ and a layer of size $H$ does not need to have $O(HD)$ parameters. The number of parameters can be reduced, and controlled using a parameter $k$ by writing it as 

$W = UV$ where $W \in \mathbb{R}^{H \times D},\ U \in \mathbb{R}^{H \times k}, \ V \in \mathbb{R}^{k \times D},\ k \ll \min(H,D)$

[factorization insight excerpt](do-they-really-need-quote1.png)

This paper also led me to an important observations: 
Suppose we start with a large, multi-purpose, pre-trained ANN such as BERT. We intend to use BERT for a particular task of relatively limited scope, such as question answering for a specific document corpus. We are looking for an ad-hoc compression of the large network, that will remove redundancies without negatively affecting its performance on the particular task.

We choose an input set, which represents the task, and run it through BERT. We select a specific "hidden" layer and look at the values at the outputs of this layer, for all the inputs. If two (or more) of these intermediate values are highly correlated, that means that they convey similar information and can be compressed into a single value. Conceptually, we're performing a Principle Component Analysis on an hidden layer and trying to replace the redundant representation with a more compact one.

