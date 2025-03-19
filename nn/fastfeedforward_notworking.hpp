#ifndef LLM_CPP__FASTFEEDFORWARD_HPP_
#define LLM_CPP__FASTFEEDFORWARD_HPP_

#include "nn.hpp"
#include "./sigmoid.hpp"  // Ensure this header file is included

namespace gpt {

struct FastFeedforward {
  using T = floatX;

  explicit FastFeedforward(int input_width, int leaf_width, int output_width, int depth) 
    : input_width_(input_width),
      leaf_width_(leaf_width),
      output_width_(output_width),
      depth_(depth),
      num_leaves_(1 << depth) {  // 2^depth leaves
    
    // Initialize decision nodes for each level
    decision_fcs_.reserve(depth);
    for(int level = 0; level < depth; level++) {
      decision_fcs_.push_back(std::make_unique<nn::Linear>(input_width, 1));
    }
    
    // Initialize leaf networks
    leaf_fcs_.reserve(num_leaves_);
    for(int leaf = 0; leaf < num_leaves_; leaf++) {
      leaf_fcs_.push_back(std::make_unique<nn::Linear>(input_width, leaf_width));
    }
    
    // Output projection
    output_fc_ = std::make_unique<nn::Linear>(leaf_width, output_width);

    // Activation tensors
    auto dtype = nn::DataTypeToEnum<T>::value;
    
    // Decision activations for each level
    decision_acts_.reserve(depth);
    for(int level = 0; level < depth; level++) {
      decision_acts_.push_back(std::make_unique<nn::Activation>(dtype));
    }
    
    // Leaf activations
    leaf_acts_.reserve(num_leaves_);
    for(int leaf = 0; leaf < num_leaves_; leaf++) {
      leaf_acts_.push_back(std::make_unique<nn::Activation>(dtype));
    }
    
    output_act_ = std::make_unique<nn::Activation>(dtype);
  }

  void Forward(typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("FastFeedforward");

    CHECK_EQ(x.dimension(1), input_width_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(y.dimension(1), output_width_);

    int BT = x.dimension(0);

    // Lazy allocation for activations
    for(auto& act : decision_acts_) {
    act->LazyAllocate(BT);
    }
    for(auto& act : leaf_acts_) {
    act->LazyAllocate(BT * leaf_width_);
    }
    output_act_->LazyAllocate(BT * output_width_);

    // Decision forward passes at each level
    std::vector<typename TTypes<T>::Matrix> decisions;
    decisions.reserve(depth_);

    for(int level = 0; level < depth_; level++) {
    auto decision = decision_acts_[level]->matrix<T>(BT, 1);
    decision_fcs_[level]->Forward(x, decision);
    nn::Sigmoid::Forward<float>(MakeConstFlat(decision.data(), decision.size()),
                            MakeFlat(decision.data(), decision.size()));
    decisions.push_back(decision);
    }

    // Leaf networks forward pass
    std::vector<typename TTypes<T>::Matrix> leaf_outputs;
    leaf_outputs.reserve(num_leaves_);

    for(int leaf = 0; leaf < num_leaves_; leaf++) {
    auto leaf_out = leaf_acts_[leaf]->matrix<T>(BT, leaf_width_);
    leaf_fcs_[leaf]->Forward(x, leaf_out);
    leaf_outputs.push_back(leaf_out);
    }

    // Compute routing probabilities for each leaf
    std::vector<std::vector<T>> routing_probs(BT, std::vector<T>(num_leaves_));

    for(int b = 0; b < BT; b++) {
    // Initialize with probability 1
    for(int leaf = 0; leaf < num_leaves_; leaf++) {
        routing_probs[b][leaf] = T(1.0);
    }

    // Apply decision probabilities level by level
    for(int level = 0; level < depth_; level++) {
        int level_stride = 1 << (depth_ - level - 1);
        auto decision_val = decisions[level](b, 0);
        
        for(int leaf = 0; leaf < num_leaves_; leaf++) {
            bool go_right = (leaf / level_stride) % 2 == 1;
            routing_probs[b][leaf] *= go_right ? decision_val : (T(1.0) - decision_val);
        }
    }
    }

    // Compute weighted sum of leaf outputs
    auto output = output_act_->matrix<T>(BT, leaf_width_);
    output.setZero();

    for(int b = 0; b < BT; b++) {
    for(int leaf = 0; leaf < num_leaves_; leaf++) {
        for(int j = 0; j < leaf_width_; j++) {
            output(b,j) += routing_probs[b][leaf] * leaf_outputs[leaf](b,j);
        }
    }
    }

    // Final projection to output dimension
    output_fc_->Forward(output.template cast<typename TTypes<T>::ConstMatrix>(), y);
    }

    void Backward(typename TTypes<T>::ConstMatrix x,
        typename TTypes<T>::ConstMatrix y_grad,
        typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("FastFeedforward");

    int BT = x.dimension(0);

    // Lazy allocate gradients
    for(auto& act : decision_acts_) {
    act->LazyAllocateGradient();
    }
    for(auto& act : leaf_acts_) {
    act->LazyAllocateGradient();
    }
    output_act_->LazyAllocateGradient();

    // First backprop through output projection
    auto output = output_act_->const_matrix<T>(BT, leaf_width_);
    auto output_grad = output_act_->matrix_grad<T>(BT, leaf_width_);
    output_fc_->Backward(output, y_grad, output_grad);

    // Get all decisions and leaf outputs from forward pass
    std::vector<typename TTypes<T>::ConstMatrix> decisions;
    std::vector<typename TTypes<T>::ConstMatrix> leaf_outputs;

    for(int level = 0; level < depth_; level++) {
    decisions.push_back(decision_acts_[level]->const_matrix<T>(BT, 1));
    }

    for(int leaf = 0; leaf < num_leaves_; leaf++) {
    leaf_outputs.push_back(leaf_acts_[leaf]->const_matrix<T>(BT, leaf_width_));
    }

    // Compute routing gradients
    std::vector<typename TTypes<T>::Matrix> decision_grads;
    std::vector<typename TTypes<T>::Matrix> leaf_grads;

    for(int level = 0; level < depth_; level++) {
    decision_grads.push_back(decision_acts_[level]->matrix_grad<T>(BT, 1));
    }

    for(int leaf = 0; leaf < num_leaves_; leaf++) {
    leaf_grads.push_back(leaf_acts_[leaf]->matrix_grad<T>(BT, leaf_width_));
    }

    // Backpropagate through routing structure
    for(int b = 0; b < BT; b++) {
    std::vector<T> routing_probs(num_leaves_);

    // Compute current routing probabilities
    for(int leaf = 0; leaf < num_leaves_; leaf++) {
        routing_probs[leaf] = T(1.0);
        for(int level = 0; level < depth_; level++) {
            int level_stride = 1 << (depth_ - level - 1);
            bool go_right = (leaf / level_stride) % 2 == 1;
            auto decision_val = decisions[level](b, 0);
            routing_probs[leaf] *= go_right ? decision_val : (T(1.0) - decision_val);
        }
    }

    // Compute leaf gradients
    for(int leaf = 0; leaf < num_leaves_; leaf++) {
        for(int j = 0; j < leaf_width_; j++) {
            leaf_grads[leaf](b,j) = output_grad(b,j) * routing_probs[leaf];
        }
    }

    // Compute decision gradients
    for(int level = 0; level < depth_; level++) {
        T level_grad = T(0.0);
        int level_stride = 1 << (depth_ - level - 1);
        
        for(int leaf = 0; leaf < num_leaves_; leaf++) {
            bool go_right = (leaf / level_stride) % 2 == 1;
            T leaf_contribution = T(0.0);
            
            for(int j = 0; j < leaf_width_; j++) {
                leaf_contribution += output_grad(b,j) * leaf_outputs[leaf](b,j);
            }
            
            if(go_right) {
                level_grad += leaf_contribution * routing_probs[leaf] / decisions[level](b,0);
            } else {
                level_grad -= leaf_contribution * routing_probs[leaf] / (T(1.0) - decisions[level](b,0));
            }
        }
        
        decision_grads[level](b,0) = level_grad;
    }
    }

    // Backpropagate through leaf networks
    x_grad.setZero();
    typename TTypes<T>::Matrix temp_grad = x_grad;

    for(int leaf = 0; leaf < num_leaves_; leaf++) {
    leaf_fcs_[leaf]->Backward(x, leaf_acts_[leaf]->const_matrix_grad<T>(BT, leaf_width_), temp_grad);
    }

    // Backpropagate through decision networks
    for(int level = 0; level < depth_; level++) {
    auto decision = decisions[level];
    auto decision_grad_flat = decision_acts_[level]->const_flat_grad<T>();
    auto temp_grad = decision_acts_[level]->flat_grad<T>();

    nn::Sigmoid::Backward<float>(
        MakeConstFlat(decision.data(), decision.size()),
        decision_grad_flat,
        temp_grad
    );

    decision_fcs_[level]->Backward(
        x,
        decision_acts_[level]->const_matrix_grad<T>(BT, 1),
        x_grad
    );
    }
    }

  size_t NumParameters() const {
    size_t total = 0;
    for(const auto& fc : decision_fcs_) {
      total += fc->NumParameters();
    }
    for(const auto& fc : leaf_fcs_) {
      total += fc->NumParameters();
    }
    total += output_fc_->NumParameters();
    return total;
  }

  size_t NumActivations() const {
    size_t total = 0;
    for(const auto& act : decision_acts_) {
      total += act->size();
    }
    for(const auto& act : leaf_acts_) {
      total += act->size();
    }
    total += output_act_->size();
    return total;
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    for(const auto& fc : decision_fcs_) {
      fc->Parameters(parameters);
    }
    for(const auto& fc : leaf_fcs_) {
      fc->Parameters(parameters);
    }
    output_fc_->Parameters(parameters);
  }

  // Network configuration
  int input_width_;
  int leaf_width_;
  int output_width_;
  int depth_;
  int num_leaves_;

  // Network components
  std::vector<std::unique_ptr<nn::Linear>> decision_fcs_;
  std::vector<std::unique_ptr<nn::Linear>> leaf_fcs_;
  std::unique_ptr<nn::Linear> output_fc_;

  // Activation tensors
  std::vector<std::unique_ptr<nn::Activation>> decision_acts_;
  std::vector<std::unique_ptr<nn::Activation>> leaf_acts_;
  std::unique_ptr<nn::Activation> output_act_;
};

}  // namespace gpt

#endif  // LLM_CPP__FASTFEEDFORWARD_HPP_