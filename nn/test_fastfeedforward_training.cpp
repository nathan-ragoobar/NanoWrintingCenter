#include <iostream>
#include <random>
#include <vector>
#include "../nn/fastfeedforward.hpp"
#include "../optimizer/optim.hpp"
#include <chrono>

// A simple synthetic training task: XOR problem with more dimensions
// This is a classic non-linear problem that requires hidden layers to solve
void create_xor_dataset(int num_samples, int input_dim, int output_dim,
                        std::vector<std::vector<float>>& inputs,
                        std::vector<std::vector<float>>& targets) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  
  inputs.resize(num_samples);
  targets.resize(num_samples);
  
  for (int i = 0; i < num_samples; ++i) {
    // Create input vector
    inputs[i].resize(input_dim);
    for (int j = 0; j < input_dim; ++j) {
      inputs[i][j] = dist(gen);
    }
    
    // Compute XOR on first 2 dimensions, then extend pattern
    bool xor_result = (inputs[i][0] > 0.0f) != (inputs[i][1] > 0.0f);
    
    // Create target vector
    targets[i].resize(output_dim);
    for (int j = 0; j < output_dim; ++j) {
      // First output is XOR of inputs, others are variations
      if (j == 0) {
        targets[i][j] = xor_result ? 1.0f : -1.0f;
      } else {
        // Add some pattern variation for multi-dimensional output
        targets[i][j] = xor_result ? (j % 2 == 0 ? 1.0f : -1.0f) : (j % 2 == 0 ? -1.0f : 1.0f);
      }
    }
  }
}

// Calculate MSE loss
float calculate_loss(const std::vector<std::vector<float>>& outputs,
                    const std::vector<std::vector<float>>& targets) {
  float total_loss = 0.0f;
  int num_samples = outputs.size();
  int output_dim = outputs[0].size();
  
  for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < output_dim; ++j) {
      float diff = outputs[i][j] - targets[i][j];
      total_loss += diff * diff;
    }
  }
  
  return total_loss / (num_samples * output_dim);
}

int main() {
  // Parameters
  const int input_dim = 4;
  const int hidden_dim = 16;
  const int output_dim = 2;
  const int depth = 1;  // Use depth=2 (4 leaves) for more capacity
  const int num_train_samples = 1000;
  const int num_test_samples = 200;
  const int batch_size = 32;
  const int epochs = 1000;
  const float learning_rate = 0.01f;
  
  std::cout << "Creating FastFeedforward network with:"
            << "\n - Input dimension: " << input_dim
            << "\n - Hidden dimension: " << hidden_dim
            << "\n - Output dimension: " << output_dim
            << "\n - Tree depth: " << depth
            << std::endl;
  
  // Create FFN model
  gpt::FastFeedforwardNetwork model(input_dim, hidden_dim, output_dim, depth);
  
  // Create Adam optimizer
  std::vector<nn::Parameter*> params;
  model.Parameters(&params);
  optim::AdamW optimizer(params, learning_rate); 
  
  // Create synthetic dataset
  std::vector<std::vector<float>> train_inputs, train_targets;
  std::vector<std::vector<float>> test_inputs, test_targets;
  
  std::cout << "Creating training dataset with " << num_train_samples << " samples..." << std::endl;
  create_xor_dataset(num_train_samples, input_dim, output_dim, train_inputs, train_targets);
  
  std::cout << "Creating test dataset with " << num_test_samples << " samples..." << std::endl;
  create_xor_dataset(num_test_samples, input_dim, output_dim, test_inputs, test_targets);
  

  // Training loop
  std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0f;
    int num_batches = (num_train_samples + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; ++batch) {
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, num_train_samples);
      int current_batch_size = end_idx - start_idx;

      // Create properly sized parameter buffers for this batch
    nn::Parameter x_data(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
    nn::Parameter y_data(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
    nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
    nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
    nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
       
      // Prepare input and target tensors
      auto x_batch = x_data.matrix<float>(batch_size, input_dim);
      auto y_batch = y_data.matrix<float>(batch_size, output_dim);
      
      for (int i = 0; i < current_batch_size; ++i) {
        int sample_idx = start_idx + i;
        for (int j = 0; j < input_dim; ++j) {
          x_batch(i, j) = train_inputs[sample_idx][j];
        }
        for (int j = 0; j < output_dim; ++j) {
          y_batch(i, j) = train_targets[sample_idx][j];
        }
      }
      
      // Zero-pad the rest
        for (int i = current_batch_size; i < batch_size; ++i) {
            for (int j = 0; j < input_dim; ++j) {
            x_batch(i, j) = 0.0f;
            }
            for (int j = 0; j < output_dim; ++j) {
            y_batch(i, j) = 0.0f;
            }
        }
        
        // Forward pass with FULL batch size
        auto x_const = x_data.const_matrix<float>(batch_size, input_dim);
        auto y_output = y_pred.matrix<float>(batch_size, output_dim);
        model.Forward(x_const, y_output, true);  // Training mode
      
      // Only compute loss on actual data
    float batch_loss = 0.0f;
    auto y_target = y_data.const_matrix<float>(batch_size, output_dim);
    auto grad = y_grad.matrix<float>(batch_size, output_dim);
    
    // Zero out gradient
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_dim; ++j) {
        grad(i, j) = 0.0f;
        }
    }
    
    // Compute loss and grad only on actual samples
    for (int i = 0; i < current_batch_size; ++i) {
        for (int j = 0; j < output_dim; ++j) {
        float diff = y_output(i, j) - y_target(i, j);
        batch_loss += diff * diff;
        grad(i, j) = 2.0f * diff / (current_batch_size * output_dim);
        }
    }
    batch_loss /= (current_batch_size * output_dim);
    
    // Backward with FULL batch size
    auto x_grad_mat = x_grad.matrix<float>(batch_size, input_dim);
    model.Backward(x_const, y_grad.const_matrix<float>(batch_size, output_dim), x_grad_mat);
    
    // Missing step: After backward pass, update the parameters
    optimizer.Step(epoch * num_batches + batch + 1, learning_rate);
    optimizer.ZeroGrad();

    // Accumulate batch loss into epoch loss
    epoch_loss += batch_loss;

}
    
    epoch_loss /= num_batches;
    
    // Evaluate on test set every 10 epochs
    if (epoch % 10 == 0 || epoch == epochs - 1) {
      std::vector<std::vector<float>> test_outputs(num_test_samples, std::vector<float>(output_dim));
      
      // Process test data in batches
      for (int batch = 0; batch < (num_test_samples + batch_size - 1) / batch_size; ++batch) {
        int start_idx = batch * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_test_samples);
        int current_batch_size = end_idx - start_idx;

        // Create properly sized buffers for this test batch
        // Always use full batch size for allocation
        nn::Parameter x_data(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
        nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
  
        // Use the actual batch size for the views
        auto x_test = x_data.matrix<float>(batch_size, input_dim);
        auto y_out = y_pred.matrix<float>(batch_size, output_dim);
        
        for (int i = 0; i < current_batch_size; ++i) {
          int sample_idx = start_idx + i;
          for (int j = 0; j < input_dim; ++j) {
            x_test(i, j) = test_inputs[sample_idx][j];
          }
        }

        // Zero out the rest
        for (int i = current_batch_size; i < batch_size; ++i) {
            for (int j = 0; j < input_dim; ++j) {
            x_test(i, j) = 0.0f;
            }
        }
        
        // Forward pass in evaluation mode
        model.Forward(x_data.const_matrix<float>(batch_size, input_dim), 
                     y_out, false);  // Eval mode
        
        // Copy outputs
        for (int i = 0; i < current_batch_size; ++i) {
          int sample_idx = start_idx + i;
          for (int j = 0; j < output_dim; ++j) {
            test_outputs[sample_idx][j] = y_out(i, j);
          }
        }
      }
      
      float test_loss = calculate_loss(test_outputs, test_targets);
      std::cout << "Epoch " << epoch << ", Train Loss: " << epoch_loss
                << ", Test Loss: " << test_loss << std::endl;
    } else {
      std::cout << "Epoch " << epoch << ", Train Loss: " << epoch_loss << std::endl;
    }
  }
  
  std::cout << "Training completed successfully!" << std::endl;
  
  //---------------------------------------------------------------------
  // INFERENCE
  //---------------------------------------------------------------------

  // Create some test examples for inference timing
  const int num_inference_samples = 1000;
  const int num_inference_iterations = 10;
  
  std::vector<std::vector<float>> inference_inputs;
  std::vector<std::vector<float>> inference_outputs(num_inference_samples, std::vector<float>(output_dim));
  
  std::cout << "Creating " << num_inference_samples << " samples for inference timing..." << std::endl;
  create_xor_dataset(num_inference_samples, input_dim, output_dim, inference_inputs, inference_outputs);
  
  // Warm-up run
  std::cout << "Performing warm-up inference..." << std::endl;
  for (int batch = 0; batch < (num_inference_samples + batch_size - 1) / batch_size; ++batch) {
    int start_idx = batch * batch_size;
    int end_idx = std::min(start_idx + batch_size, num_inference_samples);
    int current_batch_size = end_idx - start_idx;
    
    nn::Parameter x_data(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
    nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
    
    auto x_test = x_data.matrix<float>(batch_size, input_dim);
    auto y_out = y_pred.matrix<float>(batch_size, output_dim);
    
    // Copy input data
    for (int i = 0; i < current_batch_size; ++i) {
      int sample_idx = start_idx + i;
      for (int j = 0; j < input_dim; ++j) {
        x_test(i, j) = inference_inputs[sample_idx][j];
      }
    }
    
    // Zero-pad
    for (int i = current_batch_size; i < batch_size; ++i) {
      for (int j = 0; j < input_dim; ++j) {
        x_test(i, j) = 0.0f;
      }
    }
    
    // Inference pass
    model.Forward(x_data.const_matrix<float>(batch_size, input_dim), y_out, false);
  }
  
  // Timing test
  std::cout << "Running inference timing test over " << num_inference_iterations << " iterations..." << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < num_inference_iterations; ++iter) {
    for (int batch = 0; batch < (num_inference_samples + batch_size - 1) / batch_size; ++batch) {
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, num_inference_samples);
      int current_batch_size = end_idx - start_idx;
      
      nn::Parameter x_data(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
      nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
      
      auto x_test = x_data.matrix<float>(batch_size, input_dim);
      auto y_out = y_pred.matrix<float>(batch_size, output_dim);
      
      // Copy input data
      for (int i = 0; i < current_batch_size; ++i) {
        int sample_idx = start_idx + i;
        for (int j = 0; j < input_dim; ++j) {
          x_test(i, j) = inference_inputs[sample_idx][j];
        }
      }
      
      // Zero-pad
      for (int i = current_batch_size; i < batch_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
          x_test(i, j) = 0.0f;
        }
      }
      
      // Inference pass (evaluation mode)
      model.Forward(x_data.const_matrix<float>(batch_size, input_dim), y_out, false);
    }
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end_time - start_time;
  
  double total_seconds = duration.count() / 1000.0;
  double samples_per_second = (num_inference_samples * num_inference_iterations) / total_seconds;
  double ms_per_sample = duration.count() / (num_inference_samples * num_inference_iterations);
  
  std::cout << "Inference timing results:" << std::endl;
  std::cout << "- Total time: " << total_seconds << " seconds" << std::endl;
  std::cout << "- Samples processed: " << (num_inference_samples * num_inference_iterations) << std::endl;
  std::cout << "- Throughput: " << samples_per_second << " samples/second" << std::endl;
  std::cout << "- Latency: " << ms_per_sample << " ms/sample" << std::endl;
  
  // Also check accuracy on a few examples
  std::cout << "\nChecking model accuracy on a few examples:" << std::endl;
  for (int i = 0; i < 5; ++i) {
    nn::Parameter x_single(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
    nn::Parameter y_single(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
    
    auto x_view = x_single.matrix<float>(batch_size, input_dim);
    auto y_view = y_single.matrix<float>(batch_size, output_dim);
    
    // Fill with the example data
    for (int j = 0; j < input_dim; ++j) {
      x_view(0, j) = inference_inputs[i][j];
    }
    
    // Zero-pad the rest
    for (int b = 1; b < batch_size; ++b) {
      for (int j = 0; j < input_dim; ++j) {
        x_view(b, j) = 0.0f;
      }
    }
    
    // Run inference
    model.Forward(x_single.const_matrix<float>(batch_size, input_dim), y_view, false);
    
    // Print the results
    std::cout << "Example " << i << ":" << std::endl;
    std::cout << "  Input:  [";
    for (int j = 0; j < input_dim; ++j) {
      std::cout << inference_inputs[i][j];
      if (j < input_dim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Target: [";
    for (int j = 0; j < output_dim; ++j) {
      std::cout << inference_outputs[i][j];
      if (j < output_dim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Output: [";
    for (int j = 0; j < output_dim; ++j) {
      std::cout << y_view(0, j);
      if (j < output_dim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
  
  
  return 0;
}