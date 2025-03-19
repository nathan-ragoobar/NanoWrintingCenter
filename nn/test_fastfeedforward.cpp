#include <gtest/gtest.h>
#include "../nn/fastfeedforward.hpp"
#include <random>
#include <cmath>

namespace gpt {
namespace testing {

class DecisionNodeTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Default setup for tests
    input_size_ = 4;
    batch_size_ = 2;
    
    // Create random input
    std::default_random_engine generator(42); // Fixed seed for reproducibility
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    
    input_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                batch_size_ * input_size_);
    auto input = input_data_->matrix<float>(batch_size_, input_size_);
    
    for (int b = 0; b < batch_size_; ++b) {
      for (int i = 0; i < input_size_; ++i) {
        input(b, i) = distribution(generator);
      }
    }
    
    // Create output buffer
    output_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                 batch_size_);
    
    // Create gradient buffers
    output_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                 batch_size_);
    input_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                batch_size_ * input_size_);
    
    // Random gradients
    auto grad = output_grad_->matrix<float>(batch_size_, 1);
    for (int b = 0; b < batch_size_; ++b) {
      grad(b, 0) = distribution(generator);
    }
  }
  
  DecisionNode CreateDecisionNode(int node_index = 0) {
    return DecisionNode(input_size_, node_index);
  }
  
  int input_size_;
  int batch_size_;
  std::unique_ptr<nn::Parameter> input_data_;
  std::unique_ptr<nn::Parameter> output_data_;
  std::unique_ptr<nn::Parameter> output_grad_;
  std::unique_ptr<nn::Parameter> input_grad_;
};

TEST_F(DecisionNodeTest, ForwardOutputsInRange) {
  // Test that the sigmoid output is always between 0 and 1
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  
  node.Forward(input, output);
  
  for (int b = 0; b < batch_size_; ++b) {
    EXPECT_GE(output(b, 0), 0.0f) << "Output should be >= 0";
    EXPECT_LE(output(b, 0), 1.0f) << "Output should be <= 1";
  }
}

TEST_F(DecisionNodeTest, NodeIndexPreserved) {
  // Test that the node_index_ is correctly set
  const int test_index = 42;
  auto node = CreateDecisionNode(test_index);
  EXPECT_EQ(node.node_index_, test_index);
}

TEST_F(DecisionNodeTest, BackwardProducesGradient) {
  // Test that backward pass produces non-zero gradients
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  auto output_grad = output_grad_->const_matrix<float>(batch_size_, 1);
  auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
  
  // Initialize input gradients to zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      input_grad(b, i) = 0.0f;
    }
  }
  
  // Forward pass
  node.Forward(input, output);
  
  // Backward pass
  node.Backward(input, output_grad, input_grad);
  
  // Check that at least some gradients are non-zero
  bool has_nonzero_grad = false;
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      if (std::abs(input_grad(b, i)) > 1e-6) {
        has_nonzero_grad = true;
        break;
      }
    }
    if (has_nonzero_grad) break;
  }
  
  EXPECT_TRUE(has_nonzero_grad) << "Backward pass should produce non-zero gradients";
}

TEST_F(DecisionNodeTest, ForwardOutputConsistency) {
  // Test that the output is consistent across multiple Forward calls
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output1 = output_data_->matrix<float>(batch_size_, 1);
  
  // First forward pass
  node.Forward(input, output1);
  
  // Create a second output buffer
  nn::Parameter output_data2(nn::DataTypeToEnum<float>::value, batch_size_);
  auto output2 = output_data2.matrix<float>(batch_size_, 1);
  
  // Second forward pass
  node.Forward(input, output2);
  
  // Compare outputs
  for (int b = 0; b < batch_size_; ++b) {
    EXPECT_NEAR(output1(b, 0), output2(b, 0), 1e-6) 
        << "Forward pass should be deterministic";
  }
}

TEST_F(DecisionNodeTest, ZeroGradInput) {
  // Test that zero gradient input produces zero gradient output
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
  
  // Initialize input gradients to zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      input_grad(b, i) = 0.0f;
    }
  }
  
  // Forward pass
  node.Forward(input, output);
  
  // Create zero gradient input
  nn::Parameter zero_grad(nn::DataTypeToEnum<float>::value, batch_size_);
  auto zero_grad_matrix = zero_grad.matrix<float>(batch_size_, 1);
  for (int b = 0; b < batch_size_; ++b) {
    zero_grad_matrix(b, 0) = 0.0f;
  }
  
  // Backward pass with zero gradients
  node.Backward(input, zero_grad.const_matrix<float>(batch_size_, 1), input_grad);
  
  // Check that all output gradients are zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      EXPECT_NEAR(input_grad(b, i), 0.0f, 1e-6) 
          << "Zero gradient input should produce zero gradient output";
    }
  }
}

TEST_F(DecisionNodeTest, NumParameters) {
  // Test that NumParameters returns the correct count
  auto node = CreateDecisionNode();
  
  // A linear layer with input_size inputs and 1 output has input_size + 1 parameters
  // (weights + bias)
  size_t expected_num_params = input_size_ + 1;
  
  EXPECT_EQ(node.NumParameters(), expected_num_params);
}

TEST_F(DecisionNodeTest, NumActivations) {
  // Test that NumActivations returns the correct count
  auto node = CreateDecisionNode();
  
  // Forward pass to allocate activations
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  node.Forward(input, output);
  
  // Expected activations:
  // 1. Linear layer activations
  // 2. decision_output_ (batch_size * 1)
  // 3. sigmoid_output_ (batch_size * 1)
  size_t expected_activations = node.decision_->NumActivations() + 
                               batch_size_ + 
                               batch_size_;
  
  EXPECT_EQ(node.NumActivations(), expected_activations);
}

TEST_F(DecisionNodeTest, ParametersCollection) {
  // Test that Parameters collects all parameters
  auto node = CreateDecisionNode();
  
  std::vector<nn::Parameter*> params;
  node.Parameters(&params);
  
  // Should collect all parameters from the linear layer
  EXPECT_EQ(params.size(), 2); // Weights and bias
}

TEST_F(DecisionNodeTest, GradientCheckSigmoid) {
    // Simple numerical gradient check
    auto node = CreateDecisionNode();
    
    // Use a small test input
    nn::Parameter small_input(nn::DataTypeToEnum<float>::value, 4);
    auto x = small_input.matrix<float>(1, 4);
    x(0, 0) = 0.1f; x(0, 1) = -0.2f; x(0, 2) = 0.3f; x(0, 3) = -0.4f;
    
    // Initialize parameters for gradient check
    // Get parameters through the Parameters method
    std::vector<nn::Parameter*> params;
    node.Parameters(&params);
    
    // Weight is the first parameter (weight is out_features x in_features = 1x4)
    auto* weight = params[0];
    auto w = weight->matrix<float>(1, 4);
    w(0, 0) = 0.5f; w(0, 1) = -0.5f; w(0, 2) = 0.5f; w(0, 3) = -0.5f;
    
    // Bias is the second parameter
    auto* bias = params[1];
    auto b = bias->matrix<float>(1, 1);
    b(0, 0) = 0.1f;
    
    // Forward pass
    nn::Parameter output(nn::DataTypeToEnum<float>::value, 1);
    auto y = output.matrix<float>(1, 1);
    node.Forward(small_input.const_matrix<float>(1, 4), y);
    
    // Create output gradient (1.0)
    nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, 1);
    auto dy = y_grad.matrix<float>(1, 1);
    dy(0, 0) = 1.0f;
    
    // Backward pass
    nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, 4);
    auto dx = x_grad.matrix<float>(1, 4);
    for (int i = 0; i < 4; ++i) dx(0, i) = 0.0f;
    
    node.Backward(small_input.const_matrix<float>(1, 4), y_grad.const_matrix<float>(1, 1), dx);
    
    // For each input dimension, calculate numerical gradient
    const float epsilon = 1e-4f;
    for (int i = 0; i < 4; ++i) {
      // Slightly increase input
      float orig_val = x(0, i);
      x(0, i) = orig_val + epsilon;
      
      nn::Parameter output_plus(nn::DataTypeToEnum<float>::value, 1);
      auto y_plus = output_plus.matrix<float>(1, 1);
      node.Forward(small_input.const_matrix<float>(1, 4), y_plus);
      
      // Slightly decrease input
      x(0, i) = orig_val - epsilon;
      
      nn::Parameter output_minus(nn::DataTypeToEnum<float>::value, 1);
      auto y_minus = output_minus.matrix<float>(1, 1);
      node.Forward(small_input.const_matrix<float>(1, 4), y_minus);
      
      // Restore original value
      x(0, i) = orig_val;
      
      // Calculate numerical gradient
      float numerical_grad = (y_plus(0, 0) - y_minus(0, 0)) / (2 * epsilon);
      
      // Compare with analytical gradient
      EXPECT_NEAR(dx(0, i), numerical_grad, 1e-2) 
          << "Gradient check failed for input dimension " << i;
    }
  }



  class LeafNetworkTest : public ::testing::Test {
    protected:
      void SetUp() override {
        // Default setup for tests
        input_size_ = 4;
        hidden_size_ = 8;
        output_size_ = 2;
        batch_size_ = 3;
        
        // Create random input
        std::default_random_engine generator(42); // Fixed seed for reproducibility
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        
        input_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                   batch_size_ * input_size_);
        auto input = input_data_->matrix<float>(batch_size_, input_size_);
        
        for (int b = 0; b < batch_size_; ++b) {
          for (int i = 0; i < input_size_; ++i) {
            input(b, i) = distribution(generator);
          }
        }
        
        // Create output buffer
        output_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                    batch_size_ * output_size_);
        
        // Create gradient buffers
        output_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                    batch_size_ * output_size_);
        auto grad = output_grad_->matrix<float>(batch_size_, output_size_);
        
        // Random gradients
        for (int b = 0; b < batch_size_; ++b) {
          for (int j = 0; j < output_size_; ++j) {
            grad(b, j) = distribution(generator);
          }
        }
        
        input_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                   batch_size_ * input_size_);
      }
      
      LeafNetwork CreateLeafNetwork(int leaf_index = 0) {
        return LeafNetwork(input_size_, hidden_size_, output_size_, leaf_index);
      }
      
      int input_size_;
      int hidden_size_;
      int output_size_;
      int batch_size_;
      std::unique_ptr<nn::Parameter> input_data_;
      std::unique_ptr<nn::Parameter> output_data_;
      std::unique_ptr<nn::Parameter> output_grad_;
      std::unique_ptr<nn::Parameter> input_grad_;
    };
    
    TEST_F(LeafNetworkTest, LeafIndexPreserved) {
      // Test that the leaf_index_ is correctly set
      const int test_index = 42;
      auto leaf = CreateLeafNetwork(test_index);
      EXPECT_EQ(leaf.leaf_index_, test_index);
    }
    
    TEST_F(LeafNetworkTest, ForwardProducesOutput) {
      // Test that the forward pass produces non-zero output
      auto leaf = CreateLeafNetwork();
      
      auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
      auto output = output_data_->matrix<float>(batch_size_, output_size_);
      
      // Initialize output to zeros
      for (int b = 0; b < batch_size_; ++b) {
        for (int j = 0; j < output_size_; ++j) {
          output(b, j) = 0.0f;
        }
      }
      
      // Forward pass
      leaf.Forward(input, output);
      
      // Check that at least some outputs are non-zero
      bool has_nonzero_output = false;
      for (int b = 0; b < batch_size_; ++b) {
        for (int j = 0; j < output_size_; ++j) {
          if (std::abs(output(b, j)) > 1e-6) {
            has_nonzero_output = true;
            break;
          }
        }
        if (has_nonzero_output) break;
      }
      
      EXPECT_TRUE(has_nonzero_output) << "Forward pass should produce non-zero outputs";
    }
    
    TEST_F(LeafNetworkTest, ForwardOutputConsistency) {
      // Test that the output is consistent across multiple Forward calls
      auto leaf = CreateLeafNetwork();
      
      auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
      auto output1 = output_data_->matrix<float>(batch_size_, output_size_);
      
      // First forward pass
      leaf.Forward(input, output1);
      
      // Create a second output buffer
      nn::Parameter output_data2(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
      auto output2 = output_data2.matrix<float>(batch_size_, output_size_);
      
      // Second forward pass
      leaf.Forward(input, output2);
      
      // Compare outputs
      for (int b = 0; b < batch_size_; ++b) {
        for (int j = 0; j < output_size_; ++j) {
          EXPECT_NEAR(output1(b, j), output2(b, j), 1e-6) 
              << "Forward pass should be deterministic";
        }
      }
    }
    
    TEST_F(LeafNetworkTest, BackwardProducesGradient) {
      // Test that backward pass produces non-zero gradients
      auto leaf = CreateLeafNetwork();
      
      auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
      auto output = output_data_->matrix<float>(batch_size_, output_size_);
      auto output_grad = output_grad_->const_matrix<float>(batch_size_, output_size_);
      auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
      
      // Initialize input gradients to zero
      for (int b = 0; b < batch_size_; ++b) {
        for (int i = 0; i < input_size_; ++i) {
          input_grad(b, i) = 0.0f;
        }
      }
      
      // Forward pass
      leaf.Forward(input, output);
      
      // Backward pass
      leaf.Backward(input, output_grad, input_grad);
      
      // Check that at least some gradients are non-zero
      bool has_nonzero_grad = false;
      for (int b = 0; b < batch_size_; ++b) {
        for (int i = 0; i < input_size_; ++i) {
          if (std::abs(input_grad(b, i)) > 1e-6) {
            has_nonzero_grad = true;
            break;
          }
        }
        if (has_nonzero_grad) break;
      }
      
      EXPECT_TRUE(has_nonzero_grad) << "Backward pass should produce non-zero gradients";
    }
    
    TEST_F(LeafNetworkTest, ZeroGradInput) {
      // Test that zero gradient input produces zero gradient output
      auto leaf = CreateLeafNetwork();
      
      auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
      auto output = output_data_->matrix<float>(batch_size_, output_size_);
      auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
      
      // Initialize input gradients to zero
      for (int b = 0; b < batch_size_; ++b) {
        for (int i = 0; i < input_size_; ++i) {
          input_grad(b, i) = 0.0f;
        }
      }
      
      // Forward pass
      leaf.Forward(input, output);
      
      // Create zero gradient input
      nn::Parameter zero_grad(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
      auto zero_grad_matrix = zero_grad.matrix<float>(batch_size_, output_size_);
      for (int b = 0; b < batch_size_; ++b) {
        for (int j = 0; j < output_size_; ++j) {
          zero_grad_matrix(b, j) = 0.0f;
        }
      }
      
      // Backward pass with zero gradients
      leaf.Backward(input, zero_grad.const_matrix<float>(batch_size_, output_size_), input_grad);
      
      // Check that all output gradients are zero
      for (int b = 0; b < batch_size_; ++b) {
        for (int i = 0; i < input_size_; ++i) {
          EXPECT_NEAR(input_grad(b, i), 0.0f, 1e-6) 
              << "Zero gradient input should produce zero gradient output";
        }
      }
    }
    
    TEST_F(LeafNetworkTest, NumParameters) {
      // Test that NumParameters returns the correct count
      auto leaf = CreateLeafNetwork();
      
      // First layer: input_size * hidden_size weights + hidden_size bias
      // Second layer: hidden_size * output_size weights + output_size bias
      size_t expected_params = (input_size_ * hidden_size_ + hidden_size_) + 
                              (hidden_size_ * output_size_ + output_size_);
      
      EXPECT_EQ(leaf.NumParameters(), expected_params);
    }
    
    TEST_F(LeafNetworkTest, NumActivations) {
      // Test that NumActivations returns the correct count
      auto leaf = CreateLeafNetwork();
      
      // Forward pass to allocate activations
      auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
      auto output = output_data_->matrix<float>(batch_size_, output_size_);
      leaf.Forward(input, output);
      
      // Expected activations:
      // 1. Linear layer activations (should be 0 for Linear layers)
      // 2. hidden_ (batch_size * hidden_size)
      // 3. activated_ (batch_size * hidden_size)
      // 4. output_ (batch_size * output_size)
      size_t expected_activations = leaf.fc1_->NumActivations() + 
                                   leaf.fc2_->NumActivations() +
                                   batch_size_ * hidden_size_ + 
                                   batch_size_ * hidden_size_ + 
                                   batch_size_ * output_size_;
      
      EXPECT_EQ(leaf.NumActivations(), expected_activations);
    }
    
    TEST_F(LeafNetworkTest, ParametersCollection) {
      // Test that Parameters collects all parameters
      auto leaf = CreateLeafNetwork();
      
      std::vector<nn::Parameter*> params;
      leaf.Parameters(&params);
      
      // Should collect parameters from both fc1 and fc2 (weights and biases)
      EXPECT_EQ(params.size(), 4); // Two weights and two biases
    }
    
    TEST_F(LeafNetworkTest, GradientCheckReLU) {
      // Simple numerical gradient check for ReLU MLP
      auto leaf = CreateLeafNetwork();
      
      // Use a small test input
      nn::Parameter small_input(nn::DataTypeToEnum<float>::value, input_size_);
      auto x = small_input.matrix<float>(1, input_size_);
      x(0, 0) = 0.1f; x(0, 1) = -0.2f; x(0, 2) = 0.3f; x(0, 3) = -0.4f;
      
      // Initialize parameters for gradient check
      std::vector<nn::Parameter*> params;
      leaf.Parameters(&params);
      
      // First layer weights and bias
      auto* w1 = params[0];
      auto w1_mat = w1->matrix<float>(hidden_size_, input_size_);
      auto* b1 = params[1];
      auto b1_mat = b1->matrix<float>(1, hidden_size_);
      
      // Second layer weights and bias
      auto* w2 = params[2];
      auto w2_mat = w2->matrix<float>(output_size_, hidden_size_);
      auto* b2 = params[3];
      auto b2_mat = b2->matrix<float>(1, output_size_);
      
      // Set some specific values for reproducible testing
      for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
          w1_mat(i, j) = 0.1f * ((i + j) % 5 - 2); // Small values with some negative
        }
        b1_mat(0, i) = 0.01f * (i % 3);
        
        for (int j = 0; j < output_size_; ++j) {
          if (i < output_size_) {
            w2_mat(j, i) = 0.1f * ((i + j) % 3 - 1);
          }
        }
      }
      
      for (int j = 0; j < output_size_; ++j) {
        b2_mat(0, j) = 0.01f * j;
      }
      
      // Forward pass
      nn::Parameter output(nn::DataTypeToEnum<float>::value, output_size_);
      auto y = output.matrix<float>(1, output_size_);
      leaf.Forward(small_input.const_matrix<float>(1, input_size_), y);
      
      // Create output gradient (1.0 for all outputs)
      nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, output_size_);
      auto dy = y_grad.matrix<float>(1, output_size_);
      for (int j = 0; j < output_size_; ++j) {
        dy(0, j) = 1.0f;
      }
      
      // Backward pass
      nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, input_size_);
      auto dx = x_grad.matrix<float>(1, input_size_);
      for (int i = 0; i < input_size_; ++i) dx(0, i) = 0.0f;
      
      leaf.Backward(small_input.const_matrix<float>(1, input_size_), 
                    y_grad.const_matrix<float>(1, output_size_), dx);
      
      // For each input dimension, calculate numerical gradient
      const float epsilon = 1e-4f;
      for (int i = 0; i < input_size_; ++i) {
        // Slightly increase input
        float orig_val = x(0, i);
        x(0, i) = orig_val + epsilon;
        
        nn::Parameter output_plus(nn::DataTypeToEnum<float>::value, output_size_);
        auto y_plus = output_plus.matrix<float>(1, output_size_);
        leaf.Forward(small_input.const_matrix<float>(1, input_size_), y_plus);
        
        // Slightly decrease input
        x(0, i) = orig_val - epsilon;
        
        nn::Parameter output_minus(nn::DataTypeToEnum<float>::value, output_size_);
        auto y_minus = output_minus.matrix<float>(1, output_size_);
        leaf.Forward(small_input.const_matrix<float>(1, input_size_), y_minus);
        
        // Restore original value
        x(0, i) = orig_val;
        
        // Calculate numerical gradient
        float numerical_grad = 0.0f;
        for (int j = 0; j < output_size_; ++j) {
          numerical_grad += (y_plus(0, j) - y_minus(0, j)) / (2 * epsilon);
        }
        
        // Compare with analytical gradient
        EXPECT_NEAR(dx(0, i), numerical_grad, 1e-2) 
            << "Gradient check failed for input dimension " << i;
      }
    }
    
    TEST_F(LeafNetworkTest, BatchProcessingCorrectness) {
      // Test that batch processing gives the same result as individual processing
      auto leaf = CreateLeafNetwork();
      
      auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
      auto batch_output = output_data_->matrix<float>(batch_size_, output_size_);
      
      // Process whole batch
      leaf.Forward(input, batch_output);
      
      // Process each example individually
      for (int b = 0; b < batch_size_; ++b) {
        nn::Parameter single_input(nn::DataTypeToEnum<float>::value, input_size_);
        auto x_single = single_input.matrix<float>(1, input_size_);
        
        // Copy single example
        for (int i = 0; i < input_size_; ++i) {
          x_single(0, i) = input(b, i);
        }
        
        nn::Parameter single_output(nn::DataTypeToEnum<float>::value, output_size_);
        auto y_single = single_output.matrix<float>(1, output_size_);
        
        // New leaf network with same parameters
        auto leaf_single = CreateLeafNetwork();
        
        // Copy parameters from original network
        std::vector<nn::Parameter*> orig_params;
        leaf.Parameters(&orig_params);
        
        std::vector<nn::Parameter*> single_params;
        leaf_single.Parameters(&single_params);
        
        for (size_t i = 0; i < orig_params.size(); ++i) {
          // For simplicity, just copy the first layer weights and biases
          if (i == 0) { // First layer weights
            auto w_orig = orig_params[i]->matrix<float>(hidden_size_, input_size_);
            auto w_single = single_params[i]->matrix<float>(hidden_size_, input_size_);
            
            for (int h = 0; h < hidden_size_; ++h) {
              for (int j = 0; j < input_size_; ++j) {
                w_single(h, j) = w_orig(h, j);
              }
            }
          } else if (i == 1) { // First layer bias
            auto b_orig = orig_params[i]->matrix<float>(1, hidden_size_);
            auto b_single = single_params[i]->matrix<float>(1, hidden_size_);
            
            for (int h = 0; h < hidden_size_; ++h) {
              b_single(0, h) = b_orig(0, h);
            }
          } else if (i == 2) { // Second layer weights
            auto w_orig = orig_params[i]->matrix<float>(output_size_, hidden_size_);
            auto w_single = single_params[i]->matrix<float>(output_size_, hidden_size_);
            
            for (int o = 0; o < output_size_; ++o) {
              for (int h = 0; h < hidden_size_; ++h) {
                w_single(o, h) = w_orig(o, h);
              }
            }
          } else if (i == 3) { // Second layer bias
            auto b_orig = orig_params[i]->matrix<float>(1, output_size_);
            auto b_single = single_params[i]->matrix<float>(1, output_size_);
            
            for (int o = 0; o < output_size_; ++o) {
              b_single(0, o) = b_orig(0, o);
            }
          }
        }
        
        // Forward pass for single example
        leaf_single.Forward(single_input.const_matrix<float>(1, input_size_), y_single);
        
        // Compare with batch result for this example
        for (int j = 0; j < output_size_; ++j) {
          EXPECT_NEAR(batch_output(b, j), y_single(0, j), 1e-5) 
              << "Batch processing inconsistent with individual processing";
        }
      }
    }

    class FastFeedforwardNetworkTest : public ::testing::Test {
        protected:
          void SetUp() override {
            // Default setup for tests
            input_size_ = 4;
            hidden_size_ = 8;
            output_size_ = 2;
            batch_size_ = 3;
            
            // Create random input
            std::default_random_engine generator(42); // Fixed seed for reproducibility
            std::normal_distribution<float> distribution(0.0f, 1.0f);
            
            input_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                       batch_size_ * input_size_);
            auto input = input_data_->matrix<float>(batch_size_, input_size_);
            
            for (int b = 0; b < batch_size_; ++b) {
              for (int i = 0; i < input_size_; ++i) {
                input(b, i) = distribution(generator);
              }
            }
            
            // Create output buffer
            output_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                        batch_size_ * output_size_);
            
            // Create gradient buffers
            output_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                        batch_size_ * output_size_);
            auto grad = output_grad_->matrix<float>(batch_size_, output_size_);
            
            // Random gradients
            for (int b = 0; b < batch_size_; ++b) {
              for (int j = 0; j < output_size_; ++j) {
                grad(b, j) = distribution(generator);
              }
            }
            
            input_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                       batch_size_ * input_size_);
          }
          
          FastFeedforwardNetwork CreateNetwork(int depth = 1, bool train_hardened = false, float region_leak = 0.0f) {
            return FastFeedforwardNetwork(input_size_, hidden_size_, output_size_, depth, train_hardened, region_leak);
          }
          
          int input_size_;
          int hidden_size_;
          int output_size_;
          int batch_size_;
          std::unique_ptr<nn::Parameter> input_data_;
          std::unique_ptr<nn::Parameter> output_data_;
          std::unique_ptr<nn::Parameter> output_grad_;
          std::unique_ptr<nn::Parameter> input_grad_;
        };
        
        TEST_F(FastFeedforwardNetworkTest, ConstructorParametersCheck) {
          // Test that the constructor correctly sets parameters
          const int test_depth = 2;
          const bool test_hardened = true;
          const float test_leak = 0.1f;
          
          auto ffn = CreateNetwork(test_depth, test_hardened, test_leak);
          
          EXPECT_EQ(ffn.depth_, test_depth);
          EXPECT_EQ(ffn.train_hardened_, test_hardened);
          EXPECT_FLOAT_EQ(ffn.region_leak_, test_leak);
          EXPECT_EQ(ffn.n_leaves_, 4); // 2^depth
          EXPECT_EQ(ffn.n_nodes_, 3);  // 2^depth - 1
        }
        
        TEST_F(FastFeedforwardNetworkTest, NumNodesAndLeaves) {
          // Test that the correct number of nodes and leaves are created based on depth
          for (int depth = 1; depth <= 3; ++depth) {
            auto ffn = CreateNetwork(depth);
            
            int expected_leaves = 1 << depth;      // 2^depth
            int expected_nodes = (1 << depth) - 1; // 2^depth - 1
            
            EXPECT_EQ(ffn.n_leaves_, expected_leaves);
            EXPECT_EQ(ffn.n_nodes_, expected_nodes);
            EXPECT_EQ(ffn.decision_nodes_.size(), expected_nodes);
            EXPECT_EQ(ffn.leaf_networks_.size(), expected_leaves);
          }
        }
        
        TEST_F(FastFeedforwardNetworkTest, ForwardDepth1Output) {
          // Test that ForwardDepth1 produces output for depth=1 networks
          auto ffn = CreateNetwork(1);
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output = output_data_->matrix<float>(batch_size_, output_size_);
          
          // Initialize output to zeros
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              output(b, j) = 0.0f;
            }
          }
          
          // Forward pass
          ffn.Forward(input, output);
          
          // Check that output has non-zero values
          bool has_nonzero_output = false;
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              if (std::abs(output(b, j)) > 1e-6) {
                has_nonzero_output = true;
                break;
              }
            }
            if (has_nonzero_output) break;
          }
          
          EXPECT_TRUE(has_nonzero_output) << "Forward pass should produce non-zero outputs";
        }
        
        TEST_F(FastFeedforwardNetworkTest, ForwardOutputConsistency) {
          // Test that the output is consistent across multiple Forward calls
          auto ffn = CreateNetwork(1);
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output1 = output_data_->matrix<float>(batch_size_, output_size_);
          
          // First forward pass
          ffn.Forward(input, output1);
          
          // Create a second output buffer
          nn::Parameter output_data2(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
          auto output2 = output_data2.matrix<float>(batch_size_, output_size_);
          
          // Second forward pass
          ffn.Forward(input, output2);
          
          // Compare outputs
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              EXPECT_NEAR(output1(b, j), output2(b, j), 1e-6) 
                  << "Forward pass should be deterministic";
            }
          }
        }
        
        TEST_F(FastFeedforwardNetworkTest, TrainingHardenedEffect) {
          // Test that train_hardened parameter makes decisions binary
          auto ffn_soft = CreateNetwork(1, false);
          auto ffn_hard = CreateNetwork(1, true);
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
        
          nn::Parameter output_soft(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
          auto out_soft = output_soft.matrix<float>(batch_size_, output_size_);
          
          nn::Parameter output_hard(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
          auto out_hard = output_hard.matrix<float>(batch_size_, output_size_);
          
          // Forward pass on both networks
          ffn_soft.Forward(input, out_soft);
          ffn_hard.Forward(input, out_hard);
          
          // Outputs with hard decisions should tend to be more extreme (closer to pure leaf output)
          // This is hard to test definitively, but we can check that the outputs are different
          bool outputs_differ = false;
          for (int b = 0; b < batch_size_ && !outputs_differ; ++b) {
            for (int j = 0; j < output_size_ && !outputs_differ; ++j) {
              if (std::abs(out_soft(b, j) - out_hard(b, j)) > 1e-6) {
                outputs_differ = true;
              }
            }
          }
          
          EXPECT_TRUE(outputs_differ) << "Hard and soft routing should produce different outputs";
        }
        
        TEST_F(FastFeedforwardNetworkTest, TrainingVsEvalModes) {
          // Test that training and eval modes produce potentially different outputs
          auto ffn = CreateNetwork(2); // Use depth=2 to engage more complex routing
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          
          nn::Parameter output_train(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
          auto out_train = output_train.matrix<float>(batch_size_, output_size_);
          
          nn::Parameter output_eval(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
          auto out_eval = output_eval.matrix<float>(batch_size_, output_size_);
          
          // Forward pass in both modes
          ffn.Forward(input, out_train, true);   // Training mode
          ffn.Forward(input, out_eval, false);   // Eval mode
          
          // For depth > 1, training and eval should use different routing algorithms
          // This may result in different outputs, but not guaranteed due to network weights
          // So we'll just check that the code doesn't crash
          // In a real application, you'd want to verify the behavior more thoroughly
          
          // Just check that outputs have reasonable values
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              EXPECT_FALSE(std::isnan(out_train(b, j))) << "Training output contains NaN";
              EXPECT_FALSE(std::isnan(out_eval(b, j))) << "Eval output contains NaN";
              EXPECT_FALSE(std::isinf(out_train(b, j))) << "Training output contains Inf";
              EXPECT_FALSE(std::isinf(out_eval(b, j))) << "Eval output contains Inf";
            }
          }
        }
        
        TEST_F(FastFeedforwardNetworkTest, BackwardProducesGradient) {
          // Test that backward pass produces non-zero gradients
          auto ffn = CreateNetwork(1);
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output = output_data_->matrix<float>(batch_size_, output_size_);
          auto output_grad = output_grad_->const_matrix<float>(batch_size_, output_size_);
          auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
          
          // Initialize input gradients to zero
          for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < input_size_; ++i) {
              input_grad(b, i) = 0.0f;
            }
          }
          
          // Forward pass
          ffn.Forward(input, output);
          
          // Backward pass
          ffn.Backward(input, output_grad, input_grad);
          
          // Check that at least some gradients are non-zero
          bool has_nonzero_grad = false;
          for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < input_size_; ++i) {
              if (std::abs(input_grad(b, i)) > 1e-6) {
                has_nonzero_grad = true;
                break;
              }
            }
            if (has_nonzero_grad) break;
          }
          
          EXPECT_TRUE(has_nonzero_grad) << "Backward pass should produce non-zero gradients";
        }
        
        TEST_F(FastFeedforwardNetworkTest, ZeroGradInput) {
          // Test that zero gradient input produces zero gradient output
          auto ffn = CreateNetwork(1);
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output = output_data_->matrix<float>(batch_size_, output_size_);
          auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
          
          // Initialize input gradients to zero
          for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < input_size_; ++i) {
              input_grad(b, i) = 0.0f;
            }
          }
          
          // Forward pass
          ffn.Forward(input, output);
          
          // Create zero gradient input
          nn::Parameter zero_grad(nn::DataTypeToEnum<float>::value, batch_size_ * output_size_);
          auto zero_grad_matrix = zero_grad.matrix<float>(batch_size_, output_size_);
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              zero_grad_matrix(b, j) = 0.0f;
            }
          }
          
          // Backward pass with zero gradients
          ffn.Backward(input, zero_grad.const_matrix<float>(batch_size_, output_size_), input_grad);
          
          // Check that all output gradients are zero
          for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < input_size_; ++i) {
              EXPECT_NEAR(input_grad(b, i), 0.0f, 1e-6) 
                  << "Zero gradient input should produce zero gradient output";
            }
          }
        }
        
        TEST_F(FastFeedforwardNetworkTest, DeepNetworkForward) {
          // Test that networks with depth > 1 can perform forward pass
          auto ffn = CreateNetwork(3); // Depth = 3 => 8 leaves, 7 decision nodes
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output = output_data_->matrix<float>(batch_size_, output_size_);
          
          // Forward pass (training mode)
          ffn.Forward(input, output, true);
          
          // Check that output has reasonable values
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              EXPECT_FALSE(std::isnan(output(b, j))) << "Output contains NaN";
              EXPECT_FALSE(std::isinf(output(b, j))) << "Output contains Inf";
            }
          }
          
          // Forward pass (eval mode)
          ffn.Forward(input, output, false);
          
          // Check that output has reasonable values
          for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < output_size_; ++j) {
              EXPECT_FALSE(std::isnan(output(b, j))) << "Output contains NaN";
              EXPECT_FALSE(std::isinf(output(b, j))) << "Output contains Inf";
            }
          }
        }
        
        TEST_F(FastFeedforwardNetworkTest, DeepNetworkBackward) {
          // Test that networks with depth > 1 can perform backward pass
          auto ffn = CreateNetwork(2); // Depth = 2 => 4 leaves, 3 decision nodes
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output = output_data_->matrix<float>(batch_size_, output_size_);
          auto output_grad = output_grad_->const_matrix<float>(batch_size_, output_size_);
          auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
          
          // Initialize input gradients to zero
          for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < input_size_; ++i) {
              input_grad(b, i) = 0.0f;
            }
          }
          
          // Forward pass
          ffn.Forward(input, output);
          
          // Backward pass
          ffn.Backward(input, output_grad, input_grad);
          
          // Check that at least some gradients are non-zero
          bool has_nonzero_grad = false;
          for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < input_size_; ++i) {
              if (std::abs(input_grad(b, i)) > 1e-6) {
                has_nonzero_grad = true;
                break;
              }
            }
            if (has_nonzero_grad) break;
          }
          
          EXPECT_TRUE(has_nonzero_grad) << "Backward pass should produce non-zero gradients";
        }
        
        TEST_F(FastFeedforwardNetworkTest, NumParameters) {
          // Test that NumParameters returns expected values for different depths
          for (int depth = 1; depth <= 3; ++depth) {
            auto ffn = CreateNetwork(depth);
            
            // Calculate expected parameters
            size_t expected_params = 0;
            
            // Decision nodes: each has input_size + 1 parameters
            int num_decision_nodes = (1 << depth) - 1; // 2^depth - 1
            expected_params += num_decision_nodes * (input_size_ + 1);
            
            // Leaf networks: each has input_size*hidden_size + hidden_size + hidden_size*output_size + output_size parameters
            int num_leaf_networks = 1 << depth; // 2^depth
            expected_params += num_leaf_networks * ((input_size_ * hidden_size_ + hidden_size_) + 
                                                   (hidden_size_ * output_size_ + output_size_));
            
            EXPECT_EQ(ffn.NumParameters(), expected_params)
                << "NumParameters incorrect for depth = " << depth;
          }
        }
        
        TEST_F(FastFeedforwardNetworkTest, NumActivations) {
          // Test that NumActivations returns expected values for depth=1
          // (Full calculation for arbitrary depth is complex, so we'll do depth=1 as sanity check)
          auto ffn = CreateNetwork(1);
          
          auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
          auto output = output_data_->matrix<float>(batch_size_, output_size_);
          
          // Forward pass to allocate activations
          ffn.Forward(input, output);
          
          // Calculate expected activations for depth=1
          size_t expected_activations = 0;
          
          // One decision node: activations from decision node + choice tensor
          expected_activations += ffn.decision_nodes_[0]->NumActivations();
          expected_activations += batch_size_; // choice_ tensor (batch_size x 1)
          
          // Two leaf networks
          expected_activations += ffn.leaf_networks_[0]->NumActivations();
          expected_activations += ffn.leaf_networks_[1]->NumActivations();
          
          // leaf_outputs_ tensor for depth=1
          expected_activations += batch_size_ * output_size_ * 2;
          
          // For depth=1, no mixture_weights_
          
          EXPECT_EQ(ffn.NumActivations(), expected_activations)
              << "NumActivations incorrect for depth = 1";
        }
        
        TEST_F(FastFeedforwardNetworkTest, Parameters) {
          // Test that Parameters collects all parameters
          auto ffn = CreateNetwork(2); // Depth = 2 => 4 leaves, 3 decision nodes
          
          std::vector<nn::Parameter*> params;
          ffn.Parameters(&params);
          
          // Expected parameters
          size_t expected_param_count = 0;
          
          // Decision nodes: each has weight and bias
          expected_param_count += 3 * 2;
          
          // Leaf networks: each has two layers, each with weight and bias
          expected_param_count += 4 * 2 * 2;
          
          EXPECT_EQ(params.size(), expected_param_count)
              << "Parameters function should collect all parameters";
        }
        
        TEST_F(FastFeedforwardNetworkTest, GradientCheckDepth1) {
          // Simple numerical gradient check for depth=1
          // Only testing a single input dimension for brevity
          auto ffn = CreateNetwork(1);
          
          // Use a small test input
          nn::Parameter small_input(nn::DataTypeToEnum<float>::value, input_size_);
          auto x = small_input.matrix<float>(1, input_size_);
          x(0, 0) = 0.1f; x(0, 1) = -0.2f; x(0, 2) = 0.3f; x(0, 3) = -0.4f;
          
          // Forward pass
          nn::Parameter output(nn::DataTypeToEnum<float>::value, output_size_);
          auto y = output.matrix<float>(1, output_size_);
          ffn.Forward(small_input.const_matrix<float>(1, input_size_), y);
          
          // Create output gradient (1.0 for all outputs)
          nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, output_size_);
          auto dy = y_grad.matrix<float>(1, output_size_);
          for (int j = 0; j < output_size_; ++j) {
            dy(0, j) = 1.0f;
          }
          
          // Backward pass
          nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, input_size_);
          auto dx = x_grad.matrix<float>(1, input_size_);
          for (int i = 0; i < input_size_; ++i) dx(0, i) = 0.0f;
          
          ffn.Backward(small_input.const_matrix<float>(1, input_size_), 
                      y_grad.const_matrix<float>(1, output_size_), dx);
          
          // For simplicity, just check the first input dimension
          const float epsilon = 1e-4f;
          int i = 0; // First dimension
          
          // Slightly increase input
          float orig_val = x(0, i);
          x(0, i) = orig_val + epsilon;
          
          nn::Parameter output_plus(nn::DataTypeToEnum<float>::value, output_size_);
          auto y_plus = output_plus.matrix<float>(1, output_size_);
          ffn.Forward(small_input.const_matrix<float>(1, input_size_), y_plus);
          
          // Slightly decrease input
          x(0, i) = orig_val - epsilon;
          
          nn::Parameter output_minus(nn::DataTypeToEnum<float>::value, output_size_);
          auto y_minus = output_minus.matrix<float>(1, output_size_);
          ffn.Forward(small_input.const_matrix<float>(1, input_size_), y_minus);
          
          // Restore original value
          x(0, i) = orig_val;
          
          // Calculate numerical gradient
          float numerical_grad = 0.0f;
          for (int j = 0; j < output_size_; ++j) {
            numerical_grad += (y_plus(0, j) - y_minus(0, j)) / (2 * epsilon);
          }
          
          // Compare with analytical gradient (allowing larger tolerance due to complexity)
          EXPECT_NEAR(dx(0, i), numerical_grad, 0.1) 
              << "Gradient check failed for input dimension " << i;
        }
        

}  // namespace testing
}  // namespace gpt

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}