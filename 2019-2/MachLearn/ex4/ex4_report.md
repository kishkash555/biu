# Introduction to ML Exercise 4
Submitted by: ID 011862141

## Problem description description
The problem that was presented is to train an NN-based classifier on 1-second-long speech samples, each containing a single word (from a vocabularly of 31 English words), by different speakers in "real world" situations (background noise, different intonations, accents etc.).

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
* fully connected 80x31 &rarr; $\textrm{argmax}$ output

This reached reasonable results (83% accuracy on validation), with just 4 epochs required to surpass the 80% mark.

The development process to reach this configuration included:
* Alternating $\tanh$ and $\textrm{ReLU}$ activations
* Trying with and without batch-norm.
* trying out smaller and larger kernels
* testing both average pooling and max pooling (the fact that average pooling worked better was mildly surprizing for me)


#### Two-dimensional convolution
Next I decided to build a network based on 2-D convolutions. The expected advantage is that voices of different pitches can be learned simultaneously when the filters traverese the frequency dimension.

The initial number of paramtexeters is 16160, with a typical convolutional layer reducing this number only slightly. Pooling layers are useful to quickly reach smaller sizes, but care should be taken to make sure these layers don't oversimplify and lose important data. The first several attempts I made were based on large kernels and just one pooling layer. For example: a 2D convolution with kernel size 20x20, 40 output channels, ReLU activation, directly feeding another layer with same-size kernel and 10 output channels. This is sent to a max-pooling layer and then on to two feed-forward layers. This produced a very poor learner, which only reached 12% accuracy. The subsequent architectures used pooling layers between the convolutional layers. This allowed me to reduce the kernel size while keeping the fully-connected layers reasonably sized. The performace immediately improved. I tried 2 or 3 convolutional layers as well as 2 or 3 fully connected layers; I experimented with different kernel sizes, padding and strides; I also introduced dropout regularization and tested several dropout factors.
The final configuration is:
* 2D conv layer, kernel size 6x6, stride 2, padding 1, output channels: 32. 
* ReLU activation
* 2x2 MaxPool
* batch norm, dropout (20%)
* 2D conv layer, kernel size 8x8, stride 2, padding 1, 16 output channels
* ReLU activation
* 2x2 MaxPool
* batch norm, dropout (20%)
* flatten output
* fully-connected 640x100 with ReLU activation
* fully-connected 100x31

### Summary
Spoken words classifiers based on 1D and 2D convolutions were developed. I attempted many configuration, and reached good results. Additional configurations that could be considered: different activations, non-square filters, L2 regularizers, and other neural architectures.






