# XOR Neural Network in C

This project implements a small feedforward neural network in C that learns the XOR function using the backpropagation algorithm. The program demonstrates how basic neural networks can be built and trained from scratch without relying on external libraries.

## Features

- **Custom Implementation**: All aspects of the neural network, including forward propagation, backpropagation, and weight updates, are implemented in pure C.
- **Adjustable Parameters**:
  - **Learning Rate**: Configurable via the `learningRate` variable.
  - **Epoch Count**: Adjustable through the `epochCount` variable for controlling the number of training iterations.
  - **Network Architecture**:
    - 2 input nodes
    - 2 hidden layer nodes
    - 1 output node
- **Shuffling Training Data**: Uses the Fisher-Yates algorithm to randomize the order of training samples in each epoch.
- **Sigmoid Activation Function**: Used for both hidden and output layers, with its derivative employed in backpropagation.

## How It Works

- **Initialization**:
  - Weights and biases are randomly initialized to small values in the range [0, 1).
- **Forward Pass**:
  - Inputs are propagated through the hidden and output layers using the sigmoid activation function.
- **Backward Pass**:
  - Errors are computed at the output and propagated back to adjust weights and biases using the gradient of the sigmoid function.
- **Training**:
  - The network is trained on the XOR dataset:
    ```
    Input: 0, 0 -> Output: 0
    Input: 0, 1 -> Output: 1
    Input: 1, 0 -> Output: 1
    Input: 1, 1 -> Output: 0
    ```
  - After sufficient epochs, the network learns to predict the XOR function accurately.

## Adjustable Parameters

- **Learning Rate**: Modify `learningRate` to control the step size of weight updates.
- **Epoch Count**: Adjust `epochCount` to train the network for a specific number of iterations.

## Compilation and Execution

1. Clone the repository.
2. Compile the program using `gcc`:
   ```bash
   gcc -o xor_nn nn.c -lm
The `-lm` flag links the math library for functions like `exp`.

3. Run the program:
   ```bash
   ./xor_nn
   
## Output

The program prints the network's predictions for each input alongside the expected outputs during training. Over multiple epochs, the predictions converge to match the expected XOR outputs.

Example output:
```plaintext
Input: 0 0 Output: 0.003 Expected Output: 0
Input: 0 1 Output: 0.978 Expected Output: 1
Input: 1 0 Output: 0.982 Expected Output: 1
Input: 1 1 Output: 0.005 Expected Output: 0
```

## Future Improvements

- Add support for more complex activation functions.
- Implement momentum-based gradient descent for improved training efficiency.
- Generalize the network to support arbitrary architectures.



