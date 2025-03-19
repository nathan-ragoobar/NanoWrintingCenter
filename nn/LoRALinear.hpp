#ifndef LLM_CPP__LORALINEAR_HPP_
#define LLM_CPP__LORALINEAR_HPP_

#include "nn.hpp"
#include "Parameter.hpp"  // Make sure we include this for NormalFill
#include <cmath>  // for sqrt
#include <random> // for random normal distribution

namespace nn {

struct LoRALinear {
  using T = floatX;

  LoRALinear(int in_features, int out_features, int rank, float alpha = 1.0f)
      : in_features_(in_features), out_features_(out_features), 
        rank_(rank), alpha_(alpha), scaling_(alpha / rank) {
    
    // Original linear layer
    base_ = std::make_unique<Linear>(in_features, out_features);
    
    // LoRA low-rank matrices
    lora_A_ = std::make_unique<Parameter>(DataTypeToEnum<T>::value, in_features * rank);
    lora_B_ = std::make_unique<Parameter>(DataTypeToEnum<T>::value, rank * out_features);
    
    // Initialize LoRA matrices
    auto lora_A_data = lora_A_->span<T>();
    auto lora_B_data = lora_B_->span<T>();
    
    // Initialize A with normal distribution with std=1.0f/√rank
    NormalFill(lora_A_data, 0.0f, 1.0f/sqrt(rank)); 
    
    // Initialize B with zeros
    ConstantFill(lora_B_data, 0.0f);
    
    // Temporary storage for intermediate calculations
    tmp_ = std::make_unique<Activation>(DataTypeToEnum<T>::value);
  }

  void Forward(typename TTypes<T>::ConstMatrix x, typename TTypes<T>::Matrix y) {
    int batch_size = x.dimension(0);
    
    // Base model forward pass
    base_->Forward(x, y);
    
    // LoRA forward pass
    tmp_->LazyAllocate(batch_size * rank_);
    auto tmp_matrix = tmp_->matrix<T>(batch_size, rank_);
    
    // x @ A -> tmp (batch_size, rank)
    auto lora_A_data = lora_A_->data<T>();
    auto lora_A_matrix = MakeConstMatrix(lora_A_data, in_features_, rank_); // Use const matrix
    MatMul::Forward(x, lora_A_matrix, tmp_matrix);
    
    // (tmp @ B) * scaling -> delta_y (batch_size, out_features)
    auto lora_B_data = lora_B_->data<T>();
    auto lora_B_matrix = MakeConstMatrix(lora_B_data, rank_, out_features_); // Use const matrix
    auto tmp_const = tmp_->const_matrix<T>(batch_size, rank_);
    
    // Add the LoRA output to the base output
    MatMul::MatMulWithScale(tmp_const, lora_B_matrix, y, scaling_, 1.0f);
}

void Backward(typename TTypes<T>::ConstMatrix x,
              typename TTypes<T>::ConstMatrix y_grad,
              typename TTypes<T>::Matrix x_grad) {
    int batch_size = x.dimension(0);
    
    // Base backward pass
    base_->Backward(x, y_grad, x_grad);
    
    // LoRA backward pass
    lora_A_->LazyAllocateGradient();
    lora_B_->LazyAllocateGradient();
    tmp_->LazyAllocateGradient();
    
    auto tmp_matrix = tmp_->const_matrix<T>(batch_size, rank_);
    auto tmp_grad = tmp_->matrix_grad<T>(batch_size, rank_);
    
    // Gradient for B: tmp^T @ y_grad * scaling
    auto lora_B_grad = MakeMatrix(lora_B_->grad<T>(), rank_, out_features_);
    MatMul::MatMulTransposeA(tmp_matrix, y_grad, lora_B_grad, scaling_, 1.0f);
    
    // Gradient for tmp: y_grad @ B^T * scaling
    auto lora_B_data = lora_B_->data<T>();
    auto lora_B_matrix = MakeConstMatrix(lora_B_data, rank_, out_features_); // Use const matrix
    MatMul::MatMulTransposeB(y_grad, lora_B_matrix, tmp_grad, scaling_, 0.0f);
    
    // Gradient for A: x^T @ tmp_grad
    auto lora_A_grad = MakeMatrix(lora_A_->grad<T>(), in_features_, rank_);
    auto tmp_grad_const = tmp_->const_matrix_grad<T>(batch_size, rank_); // Use const matrix
    MatMul::MatMulTransposeA(x, tmp_grad_const, lora_A_grad, 1.0f, 1.0f);
    
    // Additional gradient for x_grad from LoRA path
    auto lora_A_data = lora_A_->data<T>();
    auto lora_A_matrix = MakeConstMatrix(lora_A_data, in_features_, rank_); // Use const matrix
    MatMul::MatMulTransposeB(tmp_grad_const, lora_A_matrix, x_grad, 1.0f, 1.0f);
    }
  size_t NumParameters() const {
    return base_->NumParameters() + lora_A_->size() + lora_B_->size();
  }

  size_t NumActivations() const {
    return base_->NumActivations() + tmp_->size();
  }

  void Parameters(std::vector<Parameter*>* parameters) const {
    base_->Parameters(parameters);
    parameters->push_back(lora_A_.get());
    parameters->push_back(lora_B_.get());
  }

  // Can be used to freeze/unfreeze base model parameters
  void SetBaseTrainable(bool trainable) {
    // Implementation depends on how Parameter handles freezing
    // This would need to access each Parameter in base_ and set a trainable flag
  }

  int in_features_;
  int out_features_;
  int rank_;
  float alpha_;
  float scaling_;
  
  std::unique_ptr<Linear> base_;
  std::unique_ptr<Parameter> lora_A_;  // [in_features, rank]
  std::unique_ptr<Parameter> lora_B_;  // [rank, out_features]
  std::unique_ptr<Activation> tmp_;    // Temporary storage for activation
};

}  // namespace nn

#endif  // LLM_CPP__LORALINEAR_HPP_