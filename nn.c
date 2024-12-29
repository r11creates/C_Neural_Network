#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// creating a basic neural network that can learn the XOR function.

// function to initialize the weights for our neural network.
// we use the random function to initialize the weights but cast the produced integers to doubles to allow for floating point division.
// dividing our random value by rand_max allows for our weight to fall in the standard range of [0,1).
double initialize_weights() {
    return ((double)rand()) / ((double)RAND_MAX);
}

// sigmoid function.
// applies the sigmoid activation to squash the input values into the range [0,1].
double sigmoid(double x) { 
    return 1 / (1 + exp(-x));
}

// calculates the derivative of the sigmoid function for use in backpropagation.
double sigmoidDerivative(double x) {
    return x * (1 - x);
}

// function to shuffle the dataset. using the fisher-yates shuffle algorithm.
// @param array is the training data set.
// @param n is the size of the data set.
void shuffle(int *array, size_t n) {

    // there is nothing to shuffle if there is only 1 element in the dataset.
    if (n > 1) { 
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1); // choosing an index j in the range [i, n - i].
            int t = array[j]; // dummy variable to store the element at the j'th index.
            array[j] = array[i]; // swapping elements at the j'th and i'th indices.
            array[i]  = t; // completing the swap.
        }
    }

}

#define inputCount 2 // number of inputs we take in. 
#define hiddenNodeCount 2 // number of hidden layer nodes.
#define outputCount 1 // number of output nodes.
#define trainingSetCount 4 // number of training sets we use.

int main(void) {

    const double learningRate = 0.1f; // initializing learning rate as  0.1.

    double hiddenLayer[hiddenNodeCount]; // creating an array for the hidden layer with the number of nodes defined above.
    double outputLayer[outputCount]; // creating an array for the output layer with the number of nodes defined above.
    
    double hiddenBias[hiddenNodeCount]; // creating an array for the hidden layer biases with the number of nodes defined above.
    double outputBias[outputCount]; // creating an array for the output layer biases with the number of nodes defined above.

    double hiddenWeight[inputCount][hiddenNodeCount]; // creating a 2-dimensional array for the hidden layer weights with the number of inputs and nodes defined above.
    double outputWeight[hiddenNodeCount][outputCount]; // creating a 2-dimensional array for the output layer weights with the number of inputs and nodes defined above.

    // the following is for the XOR function which means the output will be 1 only if the inputs are different.
    double trainingInput[trainingSetCount][inputCount] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}}; // initializing a 2-dimensional array for training inputs.
    double trainingOutput[trainingSetCount][outputCount] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}}; // initializing a 2-dimensional array for training outputs.

    // initializing weights for input to hidden layer.
    for (int i = 0; i < inputCount; i++) {
        for (int j = 0; j < hiddenNodeCount; j++){
            hiddenWeight[i][j] = initialize_weights(); // randomly initializing the weights for input to hidden layer.
        }
    }

    // initializing weights for hidden to output layer.
    for (int i = 0; i < hiddenNodeCount; i++) {
        for (int j = 0; j < outputCount; j++){
            outputWeight[i][j] = initialize_weights(); // randomly initializing the weights for hidden to output layer.
        }
    }

    // initializing output biases.
    for (int i = 0; i < outputCount; i++) {
        outputBias[i] = initialize_weights(); // randomly initializing biases for the output layer.
    }
    
    // array to track the order of training samples for shuffling.
    int trainingSetOrder[] = {0, 1, 2, 3}; 

    // number of epochs we train the neural network for.
    int epochCount = 10000; 

    // training.
    for (int epoch = 0; epoch < epochCount; epoch++) {
        shuffle(trainingSetOrder, trainingSetCount); // shuffling the dataset to randomize training order in each epoch.

        for (int x = 0; x < trainingSetCount; x++) {

            int i = trainingSetOrder[x]; // selecting the training sample based on the shuffled order.

            // implementing forward pass.
        
            // hidden layer activation.
            for (int j = 0; j < hiddenNodeCount; j++) {

                double activation = hiddenBias[j]; // starting with the bias value for each hidden node.

                for (int k = 0; k < inputCount; k++) {
                    activation += trainingInput[i][k] * hiddenWeight[k][j]; // summing weighted inputs for the hidden node.
                }

                hiddenLayer[j] = sigmoid(activation); // applying sigmoid activation function to compute the hidden node output.

            }

            // output layer activation.
            for (int j = 0; j < outputCount; j++) {

                double activation = outputBias[j]; // starting with the bias value for each output node.

                for (int k = 0; k < hiddenNodeCount; k++) {
                    activation += hiddenLayer[k] * outputWeight[k][j];  // summing weighted inputs from hidden layer to output node.
                }

                outputLayer[j] = sigmoid(activation); // applying sigmoid activation function to compute the output node value.

            }            

            printf("Input: %g %g Output: %g  Expected Output: %g \n",
                trainingInput[i][0], trainingInput[i][1],
                outputLayer[0], trainingOutput[i][0]); // displaying the input, output, and expected output.

            // implementing backpropagation.

            // updating weights.

            double deltaOutput[outputCount]; // change in output weights.

            for (int j = 0; j < outputCount; j++) {

                double error = trainingOutput[i][j] - outputLayer[j]; // calculating the error for the output node.
                deltaOutput[j] = error * sigmoidDerivative(outputLayer[j]); // computing the gradient for the output layer.

            }

            double deltaHidden[hiddenNodeCount]; // change in hidden weights.

            for (int j = 0; j < hiddenNodeCount; j++) {
                
                double error = 0.0f;
                for (int k = 0; k < outputCount; k++) {
                    error += deltaOutput[k] * outputWeight[j][k]; // summing the gradients propagated back from the output layer.
                }
                deltaHidden[j] = error * sigmoidDerivative(hiddenLayer[j]); // computing the gradient for the hidden layer.
            }

            // applying the change in output weights. 
            for (int j = 0; j < outputCount; j++) {
                outputBias[j] += deltaOutput[j] * learningRate; // updating the bias for the output layer.
                for (int k = 0; k < hiddenNodeCount; k++) {
                    outputWeight[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate; // updating the weights from hidden to output layer.
                }
            }

            // applying the change in hidden weights. 
            for (int j = 0; j < hiddenNodeCount; j++) {
                hiddenBias[j] += deltaHidden[j] * learningRate; // updating the bias for the hidden layer.
                for (int k = 0; k < inputCount; k++) {
                    hiddenWeight[k][j] += trainingInput[i][k] * deltaHidden[j] * learningRate; // updating the weights from input to hidden layer.
                }
            }
        } 
    } 

    return 0;
}
