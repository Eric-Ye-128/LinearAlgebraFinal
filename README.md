# Linear Algebra Final Project: Neural Network

## General Overview
This is a neural network that can be trained to recognize or categorize images. First, the PNG files in `Training set` are converted to grayscale and given a matrix representation. After flattening the matrices, they are passed through the neural network. An error is then computed and, using gradient descent with backpropagation, changes in the error as a result of changes to each individual weight and bias is also computed. These are then used with the learning rate to update the weights and biases to minimize the error during each epoch. 

## Inside the Neural Network
The neural network consists of an input layer, hidden layers, and an output layer. Each layer contains neurons that has a bias associated with that neuron and weights connecting that neuron to other neurons in neighboring layers. To compute the value of a neuron, the weighted sum of neurons connected to it from the previous layer is added to its bias. Then, that value is put through an activation function which finally gives you the value of that neuron. For this neural network, the sigmoid function is used as the activation function. The desired outputs and the actual outputs are used with a loss function to determine the error. The square error function is used as the loss function for this neural network. Using the chain rule, the effects of each individual weight or bias on the error can be determined. This provides information regarding how each weight or bias should be changed to minimize the error. To simplify the process, the weights, biases, and partial derivatives can be represented as matrices and vectors. 

## Notes for Further Consideration
- A CNN (convolutional neural network), which consists of additional convolutional layers and pooling layers, can be used to learn features of images as opposed to using a traditonal feedforward neural network.
- Gradient descent with backpropagation can be used to find a local minimum for the error function, but not the global minimum.
- Stochastic gradient descent or mini-batch gradient descent can be used to speed up the backpropagation.
- Extensions to the gradient descent algorithm such as momentum can be used to converge quicker.
- The learning rate, eta, must change depending on the training data.
- A learning rate schedule can be implemented to optimize the learning rate between epochs. (This is commented out on line 118 of `Main.java`.)
- The number of hidden layers and the number of neurons in each hidden layer should change based on the complexity of the task.
- Different activation functions (such as ReLU) and loss functions (such as the cross entropy loss function) can be used.

## References
- [Neural Networks From Linear Algebraic Perspective | Khalid Saifullah](https://khalidsaifullaah.github.io/neural-networks-from-linear-algebraic-perspective)
- [Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation#Loss_function)
- [Neural Networks - 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
