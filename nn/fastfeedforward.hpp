#ifndef LLM_CPP__FAST_FEEDFORWARD_NETWORK_HPP_
#define LLM_CPP__FAST_FEEDFORWARD_NETWORK_HPP_

#include "nn.hpp"
#include "activation.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {

// Decision node that determines routing between leaves
struct DecisionNode {
  using T = floatX;
  
  explicit DecisionNode(int input_size, int node_index = 0) 
      : node_index_(node_index) {
    // Decision layer is a single output linear layer
    decision_ = std::make_unique<nn::Linear>(input_size, 1);
    
    // For storing intermediate values
    auto dtype = nn::DataTypeToEnum<T>::value;
    decision_output_ = std::make_unique<nn::Activation>(dtype);
    sigmoid_output_ = std::make_unique<nn::Activation>(dtype);
  }
  
  int node_index_;

  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix choice) {
    PROFILE_TRACE_FN("DecisionNode");
    
    int BT = x.dimension(0);
    // Clear prior allocations to avoid size mismatch
    decision_output_ = std::make_unique<nn::Activation>(nn::DataTypeToEnum<T>::value);
    sigmoid_output_ = std::make_unique<nn::Activation>(nn::DataTypeToEnum<T>::value);
    
    decision_output_->LazyAllocate(BT);
    sigmoid_output_->LazyAllocate(BT);
    
    // Linear projection to scalar
    auto decision_out = decision_output_->matrix<T>(BT, 1);
    decision_->Forward(x, decision_out);
    
    // Apply sigmoid for soft routing decision
    auto sigmoid_out = sigmoid_output_->matrix<T>(BT, 1);
    nn::Sigmoid::Forward<T>(MakeConstFlat(decision_out.data(), decision_out.size()),
                         MakeFlat(sigmoid_out.data(), sigmoid_out.size()));
    
    // Copy to output
    for (int b = 0; b < BT; ++b) {
      choice(b, 0) = sigmoid_out(b, 0);
    }
  }
  
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix choice_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("DecisionNode");
    
    int BT = x.dimension(0);
    decision_output_->LazyAllocateGradient();
    sigmoid_output_->LazyAllocateGradient();
    decision_output_->ZeroGrad();
    sigmoid_output_->ZeroGrad();
    
    // Backprop through sigmoid
    auto sigmoid_out = sigmoid_output_->const_matrix<T>(BT, 1);
    auto sigmoid_grad = sigmoid_output_->matrix_grad<T>(BT, 1);
    
    for (int b = 0; b < BT; ++b) {
      sigmoid_grad(b, 0) = choice_grad(b, 0);
    }
    
    auto decision_out = decision_output_->const_flat<T>();
    auto sigmoid_grad_flat = sigmoid_output_->const_flat_grad<T>();
    auto decision_grad = decision_output_->flat_grad<T>();
    
    nn::Sigmoid::Backward<T>(decision_out, sigmoid_grad_flat, decision_grad);
    
    // Backprop through linear
    auto decision_grad_2d = decision_output_->const_matrix_grad<T>(BT, 1);
    decision_->Backward(x, decision_grad_2d, x_grad);
  }
  
  size_t NumParameters() const {
    return decision_->NumParameters();
  }
  
  size_t NumActivations() const {
    return decision_->NumActivations() + 
           decision_output_->size() + 
           sigmoid_output_->size();
  }
  
  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    decision_->Parameters(parameters);
  }
  
  // Linear projection for decision
  std::unique_ptr<nn::Linear> decision_;
  
  // Activation tensors
  std::unique_ptr<nn::Activation> decision_output_;
  std::unique_ptr<nn::Activation> sigmoid_output_;
};

// Leaf network (standard MLP)
struct LeafNetwork {
  using T = floatX;
  
  explicit LeafNetwork(int input_size, int hidden_size, int output_size, int leaf_index = 0) 
      : leaf_index_(leaf_index) {
    // two-layer MLP with activation
    fc1_ = std::make_unique<nn::Linear>(input_size, hidden_size);
    fc2_ = std::make_unique<nn::Linear>(hidden_size, output_size);
    
    // For storing activations
    auto dtype = nn::DataTypeToEnum<T>::value;
    hidden_ = std::make_unique<nn::Activation>(dtype);
    activated_ = std::make_unique<nn::Activation>(dtype);
    output_ = std::make_unique<nn::Activation>(dtype);
  }

  // Leaf index field
  int leaf_index_;
  
  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("LeafNetwork");
    
    int BT = x.dimension(0);
    int hidden_size = fc1_->out_features_;
    int output_size = y.dimension(1);
    
    // Clear prior allocations to avoid size mismatch
    hidden_ = std::make_unique<nn::Activation>(nn::DataTypeToEnum<T>::value);
    activated_ = std::make_unique<nn::Activation>(nn::DataTypeToEnum<T>::value);
    output_ = std::make_unique<nn::Activation>(nn::DataTypeToEnum<T>::value);
    
    // Allocate activation tensors
    hidden_->LazyAllocate(BT * hidden_size);
    activated_->LazyAllocate(BT * hidden_size);
    output_->LazyAllocate(BT * output_size);
    
    // First linear projection (input -> hidden)
    auto hidden_out = hidden_->matrix<T>(BT, hidden_size);
    fc1_->Forward(x, hidden_out);
    
    // Apply activation function (ReLU)
    auto activated_out = activated_->matrix<T>(BT, hidden_size);
    nn::ReLU::Forward<T>(
        MakeConstFlat(hidden_out.data(), hidden_out.size()),
        MakeFlat(activated_out.data(), activated_out.size())
    );
    
    // Second linear projection (hidden -> output)
    // Convert to ConstMatrix before passing to Forward
    auto activated_const = activated_->const_matrix<T>(BT, hidden_size);
    auto output_out = output_->matrix<T>(BT, output_size);
    fc2_->Forward(activated_const, output_out);
    
    // Copy to output matrix
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < output_size; ++j) {
        y(b, j) = output_out(b, j);
      }
    }
  }
  
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("LeafNetwork");
    
    int BT = x.dimension(0);
    int hidden_size = fc1_->out_features_;
    int output_size = y_grad.dimension(1);
    
    // Allocate gradient tensors
    output_->LazyAllocateGradient();
    activated_->LazyAllocateGradient();
    hidden_->LazyAllocateGradient();
    output_->ZeroGrad();
    activated_->ZeroGrad();
    hidden_->ZeroGrad();
    
    // Copy output gradients
    auto output_grad = output_->matrix_grad<T>(BT, output_size);
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < output_size; ++j) {
        output_grad(b, j) = y_grad(b, j);
      }
    }
    
    // Backprop through second linear layer
    auto output_grad_const = output_->const_matrix_grad<T>(BT, output_size);
    auto activated_out = activated_->const_matrix<T>(BT, hidden_size);
    auto activated_grad = activated_->matrix_grad<T>(BT, hidden_size);
    
    fc2_->Backward(activated_out, output_grad_const, activated_grad);
    
    // Backprop through activation function
    auto hidden_out = hidden_->const_flat<T>();
    auto activated_grad_flat = activated_->const_flat_grad<T>();
    auto hidden_grad = hidden_->flat_grad<T>();
    
    nn::ReLU::Backward<T>(hidden_out, activated_grad_flat, hidden_grad);
    
    // Backprop through first linear layer
    auto hidden_grad_const = hidden_->const_matrix_grad<T>(BT, hidden_size);
    fc1_->Backward(x, hidden_grad_const, x_grad);
  }
  
  
  size_t NumParameters() const {
    return fc1_->NumParameters() + fc2_->NumParameters();
  }
  
  size_t NumActivations() const {
    return fc1_->NumActivations() + fc2_->NumActivations() + 
           hidden_->size() + activated_->size() + output_->size();
  }
  
  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    fc1_->Parameters(parameters);
    fc2_->Parameters(parameters);
  }
  
  // Two-layer MLP
  std::unique_ptr<nn::Linear> fc1_;
  std::unique_ptr<nn::Linear> fc2_;
  
  // Activation tensors
  std::unique_ptr<nn::Activation> hidden_;     // Output of first linear layer
  std::unique_ptr<nn::Activation> activated_;  // Output after activation function
  std::unique_ptr<nn::Activation> output_;     // Final output
};

// Fast Feedforward Network (1 decision node + 2 leaf networks)
struct FastFeedforwardNetwork {
  using T = floatX;
  
  explicit FastFeedforwardNetwork(int input_width, int hidden_width, int output_width, int depth = 1,
                                  bool train_hardened = false, float region_leak = 0.0f) 
      : input_width_(input_width), 
        hidden_width_(hidden_width),
        output_width_(output_width), 
        depth_(depth),
        train_hardened_(train_hardened),
        region_leak_(region_leak),
        n_leaves_(1 << depth),  // 2^depth
        n_nodes_((1 << depth) - 1)  // 2^depth - 1
  {
    CHECK_GE(depth, 0) << "Depth must be non-negative";
    CHECK_GE(region_leak, 0.0f) << "Region leak must be non-negative";
    CHECK_LE(region_leak, 1.0f) << "Region leak must be <= 1.0";
    
    auto dtype = nn::DataTypeToEnum<T>::value;
    
    // Create decision nodes
    for (int i = 0; i < n_nodes_; ++i) {
      decision_nodes_.push_back(std::make_unique<DecisionNode>(input_width, i));
    }
    
    // Create leaf networks
    for (int i = 0; i < n_leaves_; ++i) {
      leaf_networks_.push_back(std::make_unique<LeafNetwork>(input_width, hidden_width, output_width, i));
    }
    
    // Allocate tensors for intermediate results
    choice_ = std::make_unique<nn::Activation>(dtype);
    leaf_outputs_ = std::make_unique<nn::Activation>(dtype);
    
    if (depth_ > 1) {  // Only needed for depth > 1
      mixture_weights_ = std::make_unique<nn::Activation>(dtype);
    }
  }
  
  // Forward method for both training and inference based on mode
  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y,
               bool is_training = true,
               bool use_hard_decisions = false) {
    
    if (is_training) {
      TrainingForward(x, y, use_hard_decisions || train_hardened_);
    } else {
      EvalForward(x, y);
    }
  }
  
  // Use your existing Forward method with minimal changes for depth=1
  void ForwardDepth1(typename TTypes<T>::ConstMatrix x,
                 typename TTypes<T>::Matrix y) {
  PROFILE_TRACE_FN("FastFeedforwardNetwork::ForwardDepth1");
  
  CHECK_EQ(x.dimension(1), input_width_);
  CHECK_EQ(y.dimension(1), output_width_);
  CHECK_EQ(x.dimension(0), y.dimension(0));
  
  int BT = x.dimension(0);
  
  // Allocate tensors
  choice_->LazyAllocate(BT);
  
  // Allocate for all leaf outputs (only need 2 for depth=1)
  leaf_outputs_->LazyAllocate(BT * output_width_ * 2);
  
  // Compute the routing decision
  auto choice_matrix = choice_->matrix<T>(BT, 1);
  decision_nodes_[0]->Forward(x, choice_matrix);
  
  // Create left view the same way you create right view
  auto left_view_start = leaf_outputs_->data<T>();
  auto left_out = TTypes<T>::Matrix(left_view_start, BT, output_width_);
  
  // Right output uses the second half of the buffer
  auto right_view_start = leaf_outputs_->data<T>() + (BT * output_width_);
  auto right_out = TTypes<T>::Matrix(right_view_start, BT, output_width_);
  
  // Compute outputs from both leaves
  leaf_networks_[0]->Forward(x, left_out);
  leaf_networks_[1]->Forward(x, right_out);
  
  // Mix outputs according to routing decision
  for (int b = 0; b < BT; ++b) {
    float choice_val = choice_matrix(b, 0);
    for (int j = 0; j < output_width_; ++j) {
      y(b, j) = choice_val * right_out(b, j) + (1.0f - choice_val) * left_out(b, j);
    }
  }
}
  
  // Training forward pass for arbitrary depth
  void TrainingForward(typename TTypes<T>::ConstMatrix x,
                       typename TTypes<T>::Matrix y,
                       bool use_hard_decisions = false) {
    PROFILE_TRACE_FN("FastFeedforwardNetwork::TrainingForward");
    
    // Special case for depth=1 to maintain backward compatibility
    if (depth_ == 1) {
      ForwardDepth1(x, y);
      return;
    }
    
    CHECK_EQ(x.dimension(1), input_width_);
    CHECK_EQ(y.dimension(1), output_width_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    
    int BT = x.dimension(0);
    
    // Allocate mixture weights tensor
    mixture_weights_->LazyAllocate(BT * n_leaves_);
    auto mixture = mixture_weights_->matrix<T>(BT, n_leaves_);
    
    // Initialize mixture to 1.0 for all leaves
    for (int b = 0; b < BT; ++b) {
      for (int l = 0; l < n_leaves_; ++l) {
        mixture(b, l) = 1.0f;
      }
    }
    
    // Allocate for choice values
    choice_->LazyAllocate(BT);
    
    // Traverse tree from root to leaves
    for (int current_depth = 0; current_depth < depth_; ++current_depth) {
      const int platform = (1 << current_depth) - 1;  // 2^d - 1
      const int nodes_at_level = 1 << current_depth;  // 2^d
      
      for (int node_idx = 0; node_idx < nodes_at_level; ++node_idx) {
        const int node_id = platform + node_idx;
        
        // Compute the decision for this node
        auto choice_matrix = choice_->matrix<T>(BT, 1);
        decision_nodes_[node_id]->Forward(x, choice_matrix);
        
        // Apply region leak if needed
        if (region_leak_ > 0.0f) {
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          
          for (int b = 0; b < BT; ++b) {
            if (dist(gen) < region_leak_) {
              choice_matrix(b, 0) = 1.0f - choice_matrix(b, 0);
            }
          }
        }
        
        // Apply hard decisions if requested
        if (use_hard_decisions) {
          for (int b = 0; b < BT; ++b) {
            choice_matrix(b, 0) = choice_matrix(b, 0) >= 0.5f ? 1.0f : 0.0f;
          }
        }
        
        // Update mixture weights
        const int leaves_per_branch = n_leaves_ / (1 << (current_depth + 1));
        const int left_start = 2 * node_idx * leaves_per_branch;
        const int right_start = (2 * node_idx + 1) * leaves_per_branch;
        
        for (int b = 0; b < BT; ++b) {
          float right_prob = choice_matrix(b, 0);
          float left_prob = 1.0f - right_prob;
          
          // Update left subtree leaves
          for (int l = 0; l < leaves_per_branch; ++l) {
            mixture(b, left_start + l) *= left_prob;
          }
          
          // Update right subtree leaves
          for (int l = 0; l < leaves_per_branch; ++l) {
            mixture(b, right_start + l) *= right_prob;
          }
        }
      }
    }
    
    // Allocate space for all leaf outputs
    leaf_outputs_->LazyAllocate(BT * output_width_ * n_leaves_);
    
    // Initialize output tensor to zeros
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < output_width_; ++j) {
        y(b, j) = 0.0f;
      }
    }
    
    // Compute outputs from all leaves and mix
    for (int leaf_idx = 0; leaf_idx < n_leaves_; ++leaf_idx) {
      // Skip leaves with zero mixture weight for all examples in batch
      bool any_nonzero = false;
      for (int b = 0; b < BT && !any_nonzero; ++b) {
        if (mixture(b, leaf_idx) > 1e-6f) {
          any_nonzero = true;
        }
      }
      
      if (!any_nonzero) continue;
      
      // Compute leaf output
      auto dtype = nn::DataTypeToEnum<T>::value;
      nn::Activation leaf_buffer(dtype);
      leaf_buffer.LazyAllocate(BT * output_width_);
      auto leaf_out = leaf_buffer.matrix<T>(BT, output_width_);
            
      // Forward through leaf network
      leaf_networks_[leaf_idx]->Forward(x, leaf_out);

      // Copy results to the appropriate section of leaf_outputs_
      auto all_outputs = leaf_outputs_->matrix<T>(BT, output_width_ * n_leaves_);
      for (int b = 0; b < BT; ++b) {
        for (int j = 0; j < output_width_; ++j) {
          all_outputs(b, leaf_idx * output_width_ + j) = leaf_out(b, j);
        }
      }
      
      // Mix with weights
      for (int b = 0; b < BT; ++b) {
        float mix_weight = mixture(b, leaf_idx);
        for (int j = 0; j < output_width_; ++j) {
          y(b, j) += mix_weight * leaf_out(b, j);
        }
      }
    }
  }
  
  // Optimized inference implementation
  void EvalForward(typename TTypes<T>::ConstMatrix x,
                   typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("FastFeedforwardNetwork::EvalForward");
    
    // Special case for depth=1
    if (depth_ == 1) {
      ForwardDepth1(x, y);
      return;
    }
    
    CHECK_EQ(x.dimension(1), input_width_);
    CHECK_EQ(y.dimension(1), output_width_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    
    int BT = x.dimension(0);
    choice_->LazyAllocate(BT);
    
    // Use vector to track current node for each batch element
    std::vector<int> current_nodes(BT, 0);
    std::vector<int> leaf_indices(BT, 0);
    
    // Traverse the decision tree for each example
    for (int depth = 0; depth < depth_; ++depth) {
      const int platform = (1 << depth) - 1;
      const int next_platform = (1 << (depth + 1)) - 1;
      
      auto choice_matrix = choice_->matrix<T>(BT, 1);
      
      // Group examples by current node to batch decision computation
      std::vector<std::vector<int>> node_to_examples(n_nodes_);
      for (int b = 0; b < BT; ++b) {
        node_to_examples[current_nodes[b]].push_back(b);
      }
      
      // Process each active node
      for (int node_id = platform; node_id < next_platform; ++node_id) {
        auto& examples = node_to_examples[node_id];
        if (examples.empty()) continue;
        
        // Create input matrix for these examples
        nn::Parameter node_x(nn::DataTypeToEnum<T>::value, examples.size() * input_width_);
        auto node_x_matrix = node_x.matrix<T>(examples.size(), input_width_);
        
        // Copy inputs
        for (size_t i = 0; i < examples.size(); ++i) {
          int b = examples[i];
          for (int j = 0; j < input_width_; ++j) {
            node_x_matrix(i, j) = x(b, j);
          }
        }
        
        // Create temp choice matrix
        nn::Parameter node_choice(nn::DataTypeToEnum<T>::value, examples.size());
        auto node_choice_matrix = node_choice.matrix<T>(examples.size(), 1);
        auto node_x_matrix_const = node_x.const_matrix<T>(examples.size(), input_width_);
        
        // Forward through decision node
        decision_nodes_[node_id]->Forward(node_x_matrix_const, node_choice_matrix);
        
        // Update current nodes and save choices
        for (size_t i = 0; i < examples.size(); ++i) {
          int b = examples[i];
          bool go_right = node_choice_matrix(i, 0) >= 0.5f;
          
          // Compute next node or leaf index
          if (depth == depth_ - 1) {
            // Leaf level: calculate leaf index
            leaf_indices[b] = 2 * (node_id - platform) + (go_right ? 1 : 0);
          } else {
            // Internal level: calculate next node
            current_nodes[b] = 2 * (node_id - platform) + (go_right ? 1 : 0) + next_platform;
          }
        }
      }
    }
    
    // Group examples by leaf for batched leaf computation
    std::vector<std::vector<int>> leaf_to_examples(n_leaves_);
    for (int b = 0; b < BT; ++b) {
      leaf_to_examples[leaf_indices[b]].push_back(b);
    }
    
    // Process each active leaf
    for (int leaf_idx = 0; leaf_idx < n_leaves_; ++leaf_idx) {
      auto& examples = leaf_to_examples[leaf_idx];
      if (examples.empty()) continue;
      
      // Create input matrix for these examples
      nn::Parameter leaf_x(nn::DataTypeToEnum<T>::value, examples.size() * input_width_);
      auto leaf_x_matrix = leaf_x.matrix<T>(examples.size(), input_width_);
      
      // Create output matrix for these examples
      nn::Parameter leaf_y(nn::DataTypeToEnum<T>::value, examples.size() * output_width_);
      auto leaf_y_matrix = leaf_y.matrix<T>(examples.size(), output_width_);
      
      // Copy inputs
      for (size_t i = 0; i < examples.size(); ++i) {
        int b = examples[i];
        for (int j = 0; j < input_width_; ++j) {
          leaf_x_matrix(i, j) = x(b, j);
        }
      }
      // Create const view of the matrix before passing to Forward
      auto leaf_x_matrix_const = leaf_x.const_matrix<T>(examples.size(), input_width_);
      
      // Forward through leaf network
      leaf_networks_[leaf_idx]->Forward(leaf_x_matrix_const, leaf_y_matrix);
      
      // Copy outputs to main output tensor
      for (size_t i = 0; i < examples.size(); ++i) {
        int b = examples[i];
        for (int j = 0; j < output_width_; ++j) {
          y(b, j) = leaf_y_matrix(i, j);
        }
      }
    }
  }
  
  void Backward(typename TTypes<T>::ConstMatrix x,
              typename TTypes<T>::ConstMatrix y_grad,
              typename TTypes<T>::Matrix x_grad) {
  PROFILE_TRACE_FN("FastFeedforwardNetwork::Backward");
  
  CHECK_EQ(x.dimension(1), input_width_);
  CHECK_EQ(y_grad.dimension(1), output_width_);
  CHECK_EQ(x.dimension(0), y_grad.dimension(0));
  CHECK_EQ(x.dimension(0), x_grad.dimension(0));
  CHECK_EQ(x.dimension(1), x_grad.dimension(1));
  
  int BT = x.dimension(0);
  
  // Special case for depth=1
  if (depth_ == 1) {
    BackwardDepth1(x, y_grad, x_grad);
    return;
  }
  
  // For deeper networks, need to initialize x_grad to zero
  for (int b = 0; b < BT; ++b) {
    for (int j = 0; j < input_width_; ++j) {
      x_grad(b, j) = 0.0f;
    }
  }
  
  // Allocate gradients for mixture weights and leaf outputs
  mixture_weights_->LazyAllocateGradient();
  leaf_outputs_->LazyAllocateGradient();
  mixture_weights_->ZeroGrad();
  leaf_outputs_->ZeroGrad();
  
  auto mixture = mixture_weights_->const_matrix<T>(BT, n_leaves_);
  auto all_outputs = leaf_outputs_->const_matrix<T>(BT, output_width_ * n_leaves_);
  
  // Compute gradients for leaf outputs based on mixture weights
  for (int leaf_idx = 0; leaf_idx < n_leaves_; ++leaf_idx) {
    // Skip leaves with zero mixture weight for all examples in batch
    bool any_nonzero = false;
    for (int b = 0; b < BT && !any_nonzero; ++b) {
      if (mixture(b, leaf_idx) > 1e-6f) {
        any_nonzero = true;
      }
    }
    
    if (!any_nonzero) continue;
    
    // Create leaf_grad tensor containing gradient for this specific leaf
    auto dtype = nn::DataTypeToEnum<T>::value;
    nn::Parameter leaf_grad(dtype, BT * output_width_);
    auto leaf_grad_matrix = leaf_grad.matrix<T>(BT, output_width_);
    
    // Compute gradient for this leaf based on mixture weights
    for (int b = 0; b < BT; ++b) {
      float mix_weight = mixture(b, leaf_idx);
      for (int j = 0; j < output_width_; ++j) {
        leaf_grad_matrix(b, j) = y_grad(b, j) * mix_weight;
      }
    }

    // Create const view of gradient matrix
    auto leaf_grad_matrix_const = leaf_grad.const_matrix<T>(BT, output_width_);
    
    // Create temporary gradient buffer
    nn::Parameter temp_grad(dtype, BT * input_width_);
    auto temp_grad_matrix = temp_grad.matrix<T>(BT, input_width_);
    temp_grad.ZeroData();
    
    // Extract the outputs for this leaf
    nn::Parameter leaf_output(dtype, BT * output_width_);
    auto leaf_output_matrix = leaf_output.matrix<T>(BT, output_width_);
    
    // Copy from the stored leaf outputs
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < output_width_; ++j) {
        leaf_output_matrix(b, j) = all_outputs(b, leaf_idx * output_width_ + j);
      }
    }
    
    // Backprop through leaf network
    leaf_networks_[leaf_idx]->Backward(x, leaf_grad_matrix_const, temp_grad_matrix);
    
    // Accumulate gradients
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < input_width_; ++j) {
        x_grad(b, j) += temp_grad_matrix(b, j);
      }
    }
    
    // Compute gradients w.r.t. mixture weights
    auto mixture_grad = mixture_weights_->matrix_grad<T>(BT, n_leaves_);
    for (int b = 0; b < BT; ++b) {
      float grad_sum = 0.0f;
      for (int j = 0; j < output_width_; ++j) {
        grad_sum += y_grad(b, j) * leaf_output_matrix(b, j);
      }
      mixture_grad(b, leaf_idx) = grad_sum;
    }
  }
  
  // Allocate gradient for choice values
  choice_->LazyAllocateGradient();
  choice_->ZeroGrad();
  
  // Backpropagate through the decision tree, level by level, bottom-up
  for (int d = depth_ - 1; d >= 0; --d) {
    const int platform = (1 << d) - 1;
    const int nodes_at_level = 1 << d;
    const int leaves_per_branch = n_leaves_ / (1 << (d + 1));
    
    for (int node_idx = 0; node_idx < nodes_at_level; ++node_idx) {
      const int node_id = platform + node_idx;
      const int left_start = 2 * node_idx * leaves_per_branch;
      const int right_start = (2 * node_idx + 1) * leaves_per_branch;
      
      // Compute gradient for this node's decision
      auto choice_grad = choice_->matrix_grad<T>(BT, 1);
      auto mixture_grad = mixture_weights_->const_matrix_grad<T>(BT, n_leaves_);
      
      for (int b = 0; b < BT; ++b) {
        // Get current accumulated values in the mixture weights gradients
        float left_sum = 0.0f;
        float right_sum = 0.0f;
        
        for (int l = 0; l < leaves_per_branch; ++l) {
          left_sum += mixture_grad(b, left_start + l);
          right_sum += mixture_grad(b, right_start + l);
        }
        
        // The gradient of choice is (right_sum - left_sum)
        choice_grad(b, 0) = right_sum - left_sum;
      }
      
      // Create temporary gradient buffer
      auto dtype = nn::DataTypeToEnum<T>::value;
      nn::Parameter temp_grad(dtype, BT * input_width_);
      auto temp_grad_matrix = temp_grad.matrix<T>(BT, input_width_);
      temp_grad.ZeroData();
      
      // Backprop through decision node
      auto choice_grad_const = choice_->const_matrix_grad<T>(BT, 1);
      decision_nodes_[node_id]->Backward(x, choice_grad_const, temp_grad_matrix);
      
      // Accumulate gradients
      for (int b = 0; b < BT; ++b) {
        for (int j = 0; j < input_width_; ++j) {
          x_grad(b, j) += temp_grad_matrix(b, j);
        }
      }
    }
  }
}

void BackwardDepth1(typename TTypes<T>::ConstMatrix x,
                   typename TTypes<T>::ConstMatrix y_grad,
                   typename TTypes<T>::Matrix x_grad) {
  PROFILE_TRACE_FN("FastFeedforwardNetwork::BackwardDepth1");
  
  int BT = x.dimension(0);
  
  // Allocate gradients
  choice_->LazyAllocateGradient();
  leaf_outputs_->LazyAllocateGradient();
  choice_->ZeroGrad();
  leaf_outputs_->ZeroGrad();
  
  // Initialize x_grad to zero
  for (int b = 0; b < BT; ++b) {
    for (int j = 0; j < input_width_; ++j) {
      x_grad(b, j) = 0.0f;
    }
  }
  
  // We need to recreate the temporary buffer for right outputs since they're not stored
  auto dtype = nn::DataTypeToEnum<T>::value;
  nn::Activation left_buffer(dtype);
  nn::Activation right_buffer(dtype);

  left_buffer.LazyAllocate(BT * output_width_);
  right_buffer.LazyAllocate(BT * output_width_);
  /*
  auto right_out = right_buffer.matrix<T>(BT, output_width_);
  
  // First, we need to re-compute the forward pass to get the leaf outputs
  auto choice_matrix = choice_->const_matrix<T>(BT, 1);
  auto left_out = leaf_outputs_->matrix<T>(BT, output_width_);
  */

  // Use direct tensor views with proper dimensions
  auto left_view_start = leaf_outputs_->data<T>();
  auto left_out = TTypes<T>::Matrix(left_view_start, BT, output_width_);
  
  auto right_view_start = leaf_outputs_->data<T>() + (BT * output_width_);
  auto right_out = TTypes<T>::Matrix(right_view_start, BT, output_width_);


  // Re-compute leaf outputs
  leaf_networks_[0]->Forward(x, left_out);
  leaf_networks_[1]->Forward(x, right_out);
  
  // Create gradient tensors for leaf networks
  //right_buffer.LazyAllocateGradient();
  //right_buffer.ZeroGrad();
  
  auto choice_val = choice_->const_matrix<T>(BT, 1);
  auto choice_grad = choice_->matrix_grad<T>(BT, 1);
  /*
  auto left_grad = leaf_outputs_->matrix_grad<T>(BT, output_width_);
  auto right_grad = right_buffer.matrix_grad<T>(BT, output_width_);
  */
  // With these lines that use direct tensor views:

  left_buffer.LazyAllocateGradient();
  right_buffer.LazyAllocateGradient();
  left_buffer.ZeroGrad();
  right_buffer.ZeroGrad();

  auto left_grad = left_buffer.matrix_grad<T>(BT, output_width_);
  auto right_grad = right_buffer.matrix_grad<T>(BT, output_width_);


  // Calculate gradients for leaf outputs and routing decision
  for (int b = 0; b < BT; ++b) {
    float c = choice_val(b, 0);
    choice_grad(b, 0) = 0;
    
    for (int j = 0; j < output_width_; ++j) {
      // Gradient w.r.t decision
      choice_grad(b, 0) += y_grad(b, j) * (right_out(b, j) - left_out(b, j));
      
      // Gradient w.r.t leaf outputs
      right_grad(b, j) = y_grad(b, j) * c;
      left_grad(b, j) = y_grad(b, j) * (1.0f - c);
    }
  }
  
  // Create temporary gradient buffers
  nn::Parameter temp_grad(dtype, BT * input_width_);
  auto temp_grad_matrix = temp_grad.matrix<T>(BT, input_width_);
  
  // Backprop through right leaf
  auto right_grad_const = right_buffer.const_matrix_grad<T>(BT, output_width_);
  temp_grad.ZeroData();
  leaf_networks_[1]->Backward(x, right_grad_const, temp_grad_matrix);
  
  // Add right leaf gradients to x_grad
  for (int b = 0; b < BT; ++b) {
    for (int j = 0; j < input_width_; ++j) {
      x_grad(b, j) += temp_grad_matrix(b, j);
    }
  }
  
  // Backprop through left leaf
  auto left_grad_const = left_buffer.const_matrix_grad<T>(BT, output_width_);
  temp_grad.ZeroData();
  leaf_networks_[0]->Backward(x, left_grad_const, temp_grad_matrix);
  
  // Add left leaf gradients to x_grad
  for (int b = 0; b < BT; ++b) {
    for (int j = 0; j < input_width_; ++j) {
      x_grad(b, j) += temp_grad_matrix(b, j);
    }
  }
  
  // Backprop through decision node
  auto choice_grad_const = choice_->const_matrix_grad<T>(BT, 1);
  temp_grad.ZeroData();
  decision_nodes_[0]->Backward(x, choice_grad_const, temp_grad_matrix);
  
  // Add decision gradients to x_grad
  for (int b = 0; b < BT; ++b) {
    for (int j = 0; j < input_width_; ++j) {
      x_grad(b, j) += temp_grad_matrix(b, j);
    }
  }
}
  
size_t NumParameters() const {
  size_t total_params = 0;
  
  // Count parameters in all decision nodes
  for (const auto& decision_node : decision_nodes_) {
    total_params += decision_node->NumParameters();
  }
  
  // Count parameters in all leaf networks
  for (const auto& leaf_network : leaf_networks_) {
    total_params += leaf_network->NumParameters();
  }
  
  return total_params;
}

size_t NumActivations() const {
  size_t total_activations = 0;
  
  // Count activations in all decision nodes
  for (const auto& decision_node : decision_nodes_) {
    total_activations += decision_node->NumActivations();
  }
  
  // Count activations in all leaf networks
  for (const auto& leaf_network : leaf_networks_) {
    total_activations += leaf_network->NumActivations();
  }
  
  // Add activations from internal tensors
  total_activations += choice_->size();
  total_activations += leaf_outputs_->size();
  
  if (depth_ > 1) {
    total_activations += mixture_weights_->size();
  }
  
  return total_activations;
}

void Parameters(std::vector<nn::Parameter*>* parameters) const {
  // Add parameters from all decision nodes
  for (const auto& decision_node : decision_nodes_) {
    decision_node->Parameters(parameters);
  }
  
  // Add parameters from all leaf networks
  for (const auto& leaf_network : leaf_networks_) {
    leaf_network->Parameters(parameters);
  }
}
  
  int input_width_;
  int hidden_width_;
  int output_width_;
  int depth_;
  bool train_hardened_;
  float region_leak_;
  int n_leaves_;
  int n_nodes_;
  
  // Network components
  std::vector<std::unique_ptr<DecisionNode>> decision_nodes_;
  std::vector<std::unique_ptr<LeafNetwork>> leaf_networks_;
  
  // Activation tensors
  std::unique_ptr<nn::Activation> choice_;
  std::unique_ptr<nn::Activation> leaf_outputs_;
  std::unique_ptr<nn::Activation> mixture_weights_;
};

}  // namespace gpt
#endif  // LLM_CPP__FAST_FEEDFORWARD_NETWORK_HPP_



/*
// In your GPT2 implementation where you currently use MLP:

// Before:
std::unique_ptr<gpt::MLP> mlp_ = std::make_unique<gpt::MLP>(n_embed);

// After:
std::unique_ptr<gpt::FastFeedforwardNetwork> mlp_ = 
    std::make_unique<gpt::FastFeedforwardNetwork>(n_embed, n_embed);


*/