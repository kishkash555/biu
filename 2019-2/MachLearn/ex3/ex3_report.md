## Report on Fashion-MNIST network

My final configuration uses a three-layer network (input,hidden,output), with a ReLU activation on the hidden layer (size 100). My code is completely modular and I can add as many layers I needed.
I also tried 4 layers (728-100-50-10) However, I did not manage to get an imporvement in the result, my dev accuracy was again almost exactly what I got with 2 layers - 88%. The number of epochs to get to this accuracy was longer.

I set the number of epochs for each run to 120. The 2-layer model reaches accuracy of aroun 85%-87% after about 20 iteration, while the 3-layer model takes about 60 epochs.

I also tried hidden layer of size 200. The numbers 200 and 100 were based on intuition with the ordinary MNIST dataset, where my experience is that 100 nodes are sufficient to get good results.

I also made various attempts with dropout of 30% and 40%. The performance did not improve. I implemented dropout of neurons, i.e. dropping of whole columns in train time, and scaling the other neurons accordingly. during evaluation of test (or dev), no dropout is applied of course.

The network was trained with minibatch-SGD with a batch size of 64 this batch size is common (and works well for MNIST). The gradient descent uses a constant learning rate of 0.001 with a moderate moementum coefficient (&gamma;=0.8). This value is based on trial and error, and despite being low it helped increase the dev-set accuracy. Setting a very high &gamma; (0.99) led to almost "catastrophic" results (i.e. performace stayed at initial random-guess level). The good gamma range was 0.8-0.85, but the higher values in this range  seemed to increase the overfitting as well, which I wanted to avoid. 

During training, I kept 1000 of the input examples as validation set (i.e. they were not used for training). My accuracies on the dev set were around 88%.

Another interesting attempt which succeeded nicely: after training for 120 epochs, I reloaded the trained model, normalized all the weights to unit vectors (and scaled the bias terms accordingly), and then proceeded to another 120 epochs of training. This model yielded the best accuracies: it was about 1.5% better (89.5%) on the dev set, and about 8% better (98.5%) on the training set. The gap of 9% between dev and train is an indication of overfitting, but still it was remarkable that the technique was able to reach accuracy levels that I couldn't reach using other methods.

Some highlights of my code architecture:
* completely modular, adding layer types requires defining their forward and backward. Adding layers to the network is by simply specifying the type with its parameters in a list.
* Training for 120 epochs with 5500 trainable parameters and batch size 64 completes in less than 4 minutes on an Intel-Core i7.
* Models can be saved as `pickle` files, then reloaded, in order to classify a test set or use as starting point for additional training.
* I implemented several learning rate rules, finally I settled on constant + momentum.
* I implemented dropout and momentum.



