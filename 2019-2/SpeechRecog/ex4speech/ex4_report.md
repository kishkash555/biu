% Speech Recognition Ex. 4
% ID 011862141
% 15 July 2019



## Overview
The goal of this assignment was to train a deep-learning based end-to-end Automated Speech Recognition system on the Google Commands Dataset. The major parts of the system, from input to output:

1. Input - *wav* file, recordings of length 1 second containing a single word
1. feature extraction - transformed STFT outputs (resulting in 161 frequencies by 101 time steps)
1. Deep Neural Network - "transforms" the sequence of feature vectors into a matrix of output probabilities over characters (plus the silence symbol)
1. A decoding stage, deducing a single word (characters without spaces) for the input sound recording.

### Training
The "raw output" of the neural network is a "matrix" of probabilities over the dicitionary of characters, with a length (time-dimension) that is longer than the target sequence. Therefore, direct application of the popular softmax-NLL loss is not possible. Since the decoding is a many-to-one process, in order to be "fair" we need to sum the disjoint probabilities of all the decoding paths that would lead to the correct output. This is the role of the CTC loss function, a dynamic programming algorithm which can handle the arbitrary-size dictionary. It requires a convention on the ordinal of the $\epsilon$ character (the default is 0). CTC loss is implemented in Pytorch as a differentiable component and is therefore a valid component in the training pipeline.

### Decoding
A "perfect" decoder would need to output the top character-sequence in terms of probability over all possible decoding paths. In this assignment, per the instructor's guidelines, a much simpler, greeedy decoding scheme was used: the single highest-probability symbol in each time frame was extracted, then duplicate characters omitted (unless separated by an $\epsilon$) the $\epsilon$s themselves are dropped.


## Network configuration
This problem calls for several layers. The high-level architecture that was suggested in class includes CNN layer(s), LSTM layer(s), and fully-connected layer(s). This provides an important guideline, but leaves several architectural questions open:

1. Convolutional layers - in order to reduce the problem dimensions, a single conv layer is not enough (unless it has large filters and a large stride). To achieve a desirable size, 2 convolutional layers are the minimum, with a max-pool between them.
1. LSTM - bi-LSTM, and stacking LSTM layers, may be more useful than a "vanilla" LSTM layer. The choice of hidden vector size is also important
1. FC - The main question is the size of the hidden layer(s) between the two FC transfromations.

### Configuration process
My first rather-successful configuration had reached a CER of 47.5% on the training set after 20 epochs. It had 2 conv layers (12x8 with stride 2, then 4x4 with stride 2. Between them, a 2x2 max-pool), and 2 FC layers, without LSTM. It also seemed to work without any batch-normalization and dropout layers.
Once the LSTM was introduced, the performance dropped, and since the training is slow, testing of different configurations was a very tedious process. Additionally, often the network would seem like it's making progress on the training set (in terms of loss and CER), but on the validation set, it would output a sequence of $\epsilon$'s - for all the words! Some of the improvements attempts were more methodical, and some were based on guesses and intuitions.

### Final configuration
#### General
1. All activations between layers are `ReLU` (except the `tanh` in the LSTM)
1. Adam optimizer with learning rate of 0.01

#### Network architecture
1. Conv 8x8, stride 2, 12 output channels**
1. Max Pool 2x2 (stride 2)
1. Conv 4x4, stride 1, 4 output channels**
1. Bi-LSTM, 2 layers, hidden size 80*
1. Fully-connected layer, output size 120*
1. Fully connected layer, output size 24 (the number of output symbols)*
1. For training: log-softmax followed by CTC loss.

\* Trained with dropout

** Batch-normalization applied to layer (before non-linearity)

This configuration scored 14% CER on the validation set after 58 epochs.


## Implementation
Using PyTorch, implementation of a neural network is straightforwad. The remaining parts that were required are the training loop (including the evaluation in validation) and decoding. The decoding is performed in `generate_guess_strings(guess, class_to_idx)`. I changed the implementation of the class-to-index dictionary to an OrderedDict to support the reverse transformation. I also create a special `GCommandLoader` for the test set. it sends the filename, rather than the label, together with the input features. The evaluation on the test is performed every time a new "high-score" is reached on the validation set. This is easier to implement than saving-and-loading functionality.


