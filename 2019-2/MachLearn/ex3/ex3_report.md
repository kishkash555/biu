## Report on Fashion-MNIST network

I used a three-layer network (input,hidden,output), with a ReLU activation on the hidden layer. The hidden layer size is 100 neurons.

The network was trained with minibatch-SGD with a batch size of 64. The gradient descent uses a constant learning rate of 0.001 with a moderate moementum coefficient (&gamma;=0.8). This value is based on trial and error, and despite being low it helped increase the dev-set accuracy. Raising gamma above this level seemed to increase the overfitting as well, which I wanted to avoid. During training, I kept 1000 of the input examples as validation set (i.e. they were not used for training). My accuracies on the dev set were around 88%.


