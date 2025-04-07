#ifndef LLM_CPP__BIT_LINEAR_HPP_
#define LLM_CPP__BIT_LINEAR_HPP_

#include <memory>
#include <cmath>
#include "tensor/tensor_util.hpp"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "Parameter.hpp"
#include "rmsnorm.hpp"  // You'll need to implement this

namespace nn {

// Helper functions for quantization
template <typename T>
void activation_quant(typename TTypes<T>::ConstMatrix x, 
                      typename TTypes<T>::Matrix x_quant) {
  int B = x.dimension(0);
  int features = x.dimension(1);
  
  for (int b = 0; b < B; ++b) {
    // Find max absolute value (clamp to minimum 1e-5)
    T max_val = T(1e-5);
    for (int i = 0; i < features; ++i) {
      max_val = std::max(max_val, std::abs(x(b, i)));
    }
    
    // Scale factor
    T scale = T(127.0) / max_val;
    
    // Quantize
    for (int i = 0; i < features; ++i) {
      T val = x(b, i) * scale;
      val = std::round(val);
      val = std::max(T(-128), std::min(T(127), val));
      x_quant(b, i) = val / scale;
    }
  }
}

template <typename T>
void weight_quant(typename TTypes<T>::ConstMatrix w, 
                 typename TTypes<T>::Matrix w_quant) {
  int out_features = w.dimension(0);
  int in_features = w.dimension(1);
  
  // Calculate mean absolute value as scale
  T sum_abs = T(0);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      sum_abs += std::abs(w(i, j));
    }
  }
  T scale = sum_abs / (out_features * in_features);
  
  // Calculate mean
  T sum = T(0);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      sum += w(i, j);
    }
  }
  T mean = sum / (out_features * in_features);
  
  // Binary quantization
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      T sign = (w(i, j) - mean) > T(0) ? T(1) : T(-1);
      w_quant(i, j) = sign * scale;
    }
  }
}

struct BitLinear {
  using T = floatX;

  BitLinear(int in_features, int out_features, bool bias = true)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias) {
    auto dtype = DataTypeToEnum<T>::value;
    weight_ = std::make_unique<Parameter>(dtype, out_features * in_features);
    KaimingUniformFill(weight_->span<T>(), in_features);
    if (bias) {
      bias_ = std::make_unique<Parameter>(dtype, out_features);
      const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
      UniformFill(bias_->span<T>(), -bound, bound);
    }
    
    // For quantization
    norm_ = std::make_unique<SimpleRMSNorm>(in_features);
    x_norm_ = std::make_unique<Activation>(dtype);
    x_quant_ = std::make_unique<Activation>(dtype);
    w_quant_ = std::make_unique<Parameter>(dtype, out_features * in_features);
  }

  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) const {
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    int B = x.dimension(0);
    
    // Normalize inputs
    x_norm_->LazyAllocate(B * in_features_);
    auto x_norm_matrix = x_norm_->matrix<T>(B, in_features_);
    norm_->Forward(x, x_norm_matrix);
    
    // Quantize inputs to 8-bit
    x_quant_->LazyAllocate(B * in_features_);
    auto x_quant_matrix = x_quant_->matrix<T>(B, in_features_);
    activation_quant<T>(x_norm_matrix, x_quant_matrix);
    
    // Quantize weights to 1-bit
    auto weight = MakeMatrix(weight_->data<T>(), out_features_, in_features_);
    auto w_quant_matrix = MakeMatrix(w_quant_->data<T>(), out_features_, in_features_);
    weight_quant<T>(weight, w_quant_matrix);
    
    // Compute output
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    if (has_bias_) {
      auto bias = MakeFlat(bias_->data<T>(), out_features_);
      Eigen::array<int, 2> batch_by_one = {B, 1},
                         one_by_out = {1, out_features_};
      y.device(g_device) = x_quant_matrix.contract(w_quant_matrix, product_dims) +
                         bias.reshape(one_by_out).broadcast(batch_by_one);
    } else {
      y.device(g_device) = x_quant_matrix.contract(w_quant_matrix, product_dims);
    }
  }

  // Backward pass (with Straight-Through Estimator)
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    // Implementation similar to Linear::Backward but with
    // gradient pass-through for quantization (STE)
    // ...
  }

  // Other methods similar to Linear
  
  bool has_bias_;
  int in_features_;
  int out_features_;
  std::unique_ptr<Parameter> weight_;
  std::unique_ptr<Parameter> bias_;
  
  // Additional members for quantization
  std::unique_ptr<SimpleRMSNorm> norm_;
  std::unique_ptr<Activation> x_norm_;
  std::unique_ptr<Activation> x_quant_;
  std::unique_ptr<Parameter> w_quant_;
};

}  // namespace nn

struct SimpleRMSNorm {
    using T = floatX;
    
    explicit SimpleRMSNorm(int dim) : dim_(dim) {}
    
    void Forward(typename TTypes<T>::ConstMatrix x,
                 typename TTypes<T>::Matrix y) const {
      int B = x.dimension(0);
      
      for (int b = 0; b < B; ++b) {
        // Calculate RMS
        T sum_squared = T(0);
        for (int i = 0; i < dim_; ++i) {
          sum_squared += x(b, i) * x(b, i);
        }
        T rms = std::sqrt(sum_squared / dim_ + T(1e-5));
        
        // Normalize
        for (int i = 0; i < dim_; ++i) {
          y(b, i) = x(b, i) / rms;
        }
      }
    }
    
    int dim_;
  };

#endif  // LLM_CPP__BIT_LINEAR_HPP_