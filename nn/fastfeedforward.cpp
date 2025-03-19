#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

// Sigmoid activation function and its derivative
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

// A small leaf neural network
class LeafNetwork {
public:
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> last_input;
    
    LeafNetwork(int input_size, int output_size) {
        // Initialize weights and biases
        weights = vector<vector<double>>(output_size, vector<double>(input_size, 0.01));
        biases = vector<double>(output_size, 0.01);
    }

    vector<double> forward(const vector<double>& input) {
        last_input = input;  // Store input for backprop
        vector<double> output(weights.size(), 0.0);
        
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < input.size(); j++) {
                output[i] += weights[i][j] * input[j];
            }
            output[i] += biases[i];
        }
        return output;
    }

    vector<double> backward(const vector<double>& grad_output, double learning_rate) {
        vector<double> grad_input(last_input.size(), 0.0);
        
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < last_input.size(); j++) {
                grad_input[j] += grad_output[i] * weights[i][j];  // Backprop to input
                weights[i][j] -= learning_rate * grad_output[i] * last_input[j];  // Update weights
            }
            biases[i] -= learning_rate * grad_output[i];  // Update biases
        }
        return grad_input;
    }
};

// Decision node that determines which leaf to use
class DecisionNode {
public:
    vector<double> weights;
    double bias;
    double last_input;
    double last_choice;

    DecisionNode(int input_size) {
        weights = vector<double>(input_size, 0.01);
        bias = 0.01;
    }

    double forward(const vector<double>& input) {
        last_input = 0.0;
        for (size_t i = 0; i < weights.size(); i++) {
            last_input += weights[i] * input[i];
        }
        last_input += bias;
        last_choice = sigmoid(last_input);  // Soft choice
        return last_choice;
    }

    double backward(double grad_choice, double learning_rate) {
        double grad_sigmoid = sigmoid_derivative(last_input);
        double grad_input = grad_choice * grad_sigmoid;

        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] -= learning_rate * grad_input;  // Update decision weights
        }
        bias -= learning_rate * grad_input;  // Update bias
        return grad_input;
    }
};

// Fast Feedforward Network (1 decision layer + 2 leaf layers)
class FastFeedforwardNetwork {
public:
    DecisionNode decision;
    LeafNetwork left_leaf;
    LeafNetwork right_leaf;

    FastFeedforwardNetwork(int input_size, int output_size)
        : decision(input_size), left_leaf(input_size, output_size), right_leaf(input_size, output_size) {}

    vector<double> forward(const vector<double>& input) {
        double choice = decision.forward(input);  // Compute soft decision

        vector<double> left_output = left_leaf.forward(input);
        vector<double> right_output = right_leaf.forward(input);

        vector<double> output(left_output.size(), 0.0);
        for (size_t i = 0; i < output.size(); i++) {
            output[i] = choice * right_output[i] + (1 - choice) * left_output[i];  // Mixture of leaves
        }
        return output;
    }

    void backward(const vector<double>& grad_output, double learning_rate) {
        vector<double> left_grad = left_leaf.backward(grad_output, learning_rate);
        vector<double> right_grad = right_leaf.backward(grad_output, learning_rate);

        double choice_grad = 0.0;
        for (size_t i = 0; i < grad_output.size(); i++) {
            choice_grad += grad_output[i] * (right_leaf.last_input[i] - left_leaf.last_input[i]);  // Gradient w.r.t choice
        }

        decision.backward(choice_grad, learning_rate);
    }

    size_t getParameterCount() const {
        size_t total = 0;
        
        // Count decision node parameters
        total += decision.weights.size();  // weights
        total += 1;  // bias
        
        // Count left leaf parameters
        total += left_leaf.weights.size() * left_leaf.weights[0].size();  // weights
        total += left_leaf.biases.size();  // biases
        
        // Count right leaf parameters
        total += right_leaf.weights.size() * right_leaf.weights[0].size();  // weights
        total += right_leaf.biases.size();  // biases
        
        return total;
    }
};

// Test the FFF network
int main() {
    FastFeedforwardNetwork fff(10000000, 10);  // 2 inputs, 1 output

    // Print parameter count
    cout << "Total number of parameters: " << fff.getParameterCount() << endl;
    
    //vector<double> input = {1.0, 2.0};
    //vector<double> target = {0.5};

    // Create input vector with 100 values
    vector<double> input(10000000, 0.0);
    // Fill input with some values (example: increasing numbers)
    for(int i = 0; i < 10000000; i++) {
        input[i] = i * 0.000001;  // or any other initialization you prefer
    }

    // Create target vector with 10 values
    vector<double> target(10, 0.0);
    // Fill target with desired output values
    for(int i = 0; i < 10; i++) {
        target[i] = 0.5;  // or any other target values you want
    }

    double learning_rate = 0.0000001;
    int epochs = 100;

    for (int epoch = 0; epoch < epochs; epoch++) {
        vector<double> output = fff.forward(input);

        // Calculate total loss across all outputs
        double loss = 0.0;
        vector<double> loss_grad(10);
        for(int i = 0; i < 10; i++) {
            loss += 0.5 * pow(output[i] - target[i], 2);
            loss_grad[i] = output[i] - target[i];
        }

        fff.backward({loss_grad}, learning_rate);  // Backpropagation

        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << " - Loss: " << loss << endl;
        }
    }

    return 0;
}
