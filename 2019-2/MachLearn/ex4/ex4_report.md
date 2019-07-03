Introduction to ML Exercise 4

## Problem description description
The problem that was presented is to train an NN-based classifier 1-second-long speech samples, each containing a single word (from a vocabularly of 31 English words), by different speakers in "real world" situations (background noise, different intonations, accents etc.).

### Dataset and preprocessing
The dataset contains 30,000 samples, all are one-second long and with a sample rate of 16 KHz. The samples were all converted using short-time fourier-transform (STFT), resulting in each sample being represnted by an array of 161 amplitudes  by 101 time frames. Hamming window was used for smoothing. Phase information was discarded and magnitude distribution was  made more even using $\log(1+x)$ transformation. Finally, each sample was shifted and scaled to a mean of 0 and standard deviation 1.

The normalized arrays were used as inputs to the learning algorithm.

## Approach
Two classifiers were developed, The first based on a 1-D convolutional neural network, the other based on 2-D layers. Both reached good results, with the 2-D configuration slightly surpassing the 1-D version on the validation set.


### Development process

#### One-dimensional convolution
First, I opted for a 1-D convolution in time. This has the following advantages:
* taking each of the 161 frequencies as a separate channel, means that the most extreme frequencies (especially high frequencies above ~4KHz) can be naturally filtered out by the learning system if they provide less value, while the rest of the frequencies can be used as needed.

* Each convolutional feature has just 100 (or less) outputs, a size that is both large enough to convey the necessary information forward and small enough to be easily manageable by downstream layers

After a few attempts, I reached the following configuration: 

* 1-D convolution, 161 input channels, 40 output channels, kernel size 12, no padding &rarr; 40x90 output
* $\tanh$ activation
* Average pooling, window size 10 and stride 10 &rarr; 40x9 output
* reshape to vector
* fully-connected 360x80 &rarr; 80
* batch-normalization and $\tanh$ activation
* fully connected 80x31 &rarr; $\textcf{argmax}$ output

This reached reasonable results (83% accuracy on validation), with just 4 epochs required to surpass the 80% mark.

The development process to reach this configuration included:
* Alternating $\tanh$ and $\textcf{ReLU}$ activations
* Trying with and without batch-norm.
* trying out smaller and larger kernels
* testing both average pooling and max pooling (the fact that average pooling worked better was mildly surprizing for me)


#### Two-dimensional convolution
Next I decided to build a network based on 2-D convolutions. The initial number of parameters is 16160, with a typical convolutional layer reducing this number only slightly. Pooling layers are useful to quickly reach smaller sizes, but care should be taken to make sure these layers don't oversimplify. The first several attempts I made were based on large kernels yielded very poor learners, which only reached 10% accuracy. so the architecture needs to reduce that number from layer to layer. Pooling layers 


