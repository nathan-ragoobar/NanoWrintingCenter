#ifndef LLM_CPP__LINEAR_HPP_
#define LLM_CPP__LINEAR_HPP_

#include <memory>
#include <cmath>
#include "tensor/tensor_util.hpp"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "Parameter.hpp"  // Include the Parameter header

namespace nn {

struct Linear {
  using T = floatX;

  Linear(int in_features, int out_features, bool bias = true)
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
  }

  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) const {
    // x: [B, in_features], y: [B, out_features]
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    int B = x.dimension(0);

    auto weight = MakeMatrix(weight_->data<T>(), out_features_, in_features_);
    // y = x * w^T + b
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    if (has_bias_) {
      auto bias = MakeFlat(bias_->data<T>(), out_features_);
      Eigen::array<int, 2> batch_by_one = {B, 1},
                           one_by_out = {1, out_features_};
      y.device(g_device) = x.contract(weight, product_dims) +
                           bias.reshape(one_by_out).broadcast(batch_by_one);
    } else {
      y.device(g_device) = x.contract(weight, product_dims);
    }
  }

  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    // x: [B, in_features], y_grad: [B, out_features], x_grad: [B, in_features]
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y_grad.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));

    // Lazily allocate the memory for gradients
    weight_->LazyAllocateGradient();
    auto weight = MakeMatrix(weight_->data<T>(), out_features_, in_features_);
    auto weight_grad =
        MakeMatrix(weight_->grad<T>(), out_features_, in_features_);

    // x_grad = dL/dy * dy/dx
    //        = y_grad(B, out_features) * W(out_features, in_features)
    //        = [B, in_features]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    x_grad.device(g_device) += y_grad.contract(weight, product_dims);

    // w_grad = dL/dy * dy/dw
    //        = y_grad^T(out_features, B) * x(B, in_features)
    //        = [out_features, in_features]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    weight_grad.device(g_device) += y_grad.contract(x, product_dims2);

    if (has_bias_) {
      // b_grad = dL/dy * dy/db
      //        = \sum_i^(B)(y_grad(B, out_features))
      //        = [out_features,]
      bias_->LazyAllocateGradient();
      auto bias_grad = MakeFlat(bias_->grad<T>(), out_features_);
      Eigen::array<Eigen::Index, 1> along_batch = {0};
      bias_grad.device(g_device) = y_grad.sum(along_batch);
    }
  }

  size_t NumParameters() const {
    size_t num_parameters = out_features_ * in_features_;
    if (has_bias_) {
      num_parameters += out_features_;
    }

    return num_parameters;
  }

  size_t NumActivations() const { return 0; }

  void Parameters(std::vector<Parameter*>* parameters) const {
    parameters->push_back(weight_.get());
    if (has_bias_) {
      parameters->push_back(bias_.get());
    }
  }

  bool has_bias_;
  int in_features_;
  int out_features_;
  std::unique_ptr<Parameter> weight_;  // out_features x in_features
  std::unique_ptr<Parameter> bias_;    // out_features
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_