#ifndef LLM_CPP__LAYERNORM_HPP_
#define LLM_CPP__LAYERNORM_HPP_


#include "Parameter.hpp"

//#include <unistd.h>
#include <memory>
#include <cmath>
//#include "../tensor/tensor_util.hpp"
#include "abseil-cpp/absl/log/check.h"
#include "abseil-cpp/absl/types/span.h"
//#include "Parameter.hpp"  // Include the Parameter header


namespace nn {

struct LayerNorm {
  using T = floatX;

  LayerNorm(int normalized_shape)
      : normalized_shape_(normalized_shape), eps_(1e-5) {
    auto dtype = DataTypeToEnum<T>::value;
    weight_ = std::make_unique<Parameter>(dtype, normalized_shape);
    auto w = weight_->span<T>();
    ConstantFill(w, 1.0f);
    bias_ = std::make_unique<Parameter>(dtype, normalized_shape);
    auto b = bias_->span<T>();
    ConstantFill(b, 0.0f);

    // activation gradient tensor
    norm_ = std::make_unique<Parameter>(dtype);             // [B, D]
    dnorm_ = std::make_unique<Parameter>(dtype);            // [B, D]
    dnorm_mean_ = std::make_unique<Parameter>(dtype);       // [B,]
    dnorm_norm_mean_ = std::make_unique<Parameter>(dtype);  // [B,]
  }

  void Forward(typename TTypes<T>::ConstMatrix x, typename TTypes<T>::Matrix y,
               typename TTypes<T>::Flat mean, typename TTypes<T>::Flat rstd) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.dimension(1), normalized_shape_);
    CHECK_EQ(y.dimension(1), normalized_shape_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    int B = x.dimension(0);

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);

    Eigen::array<Eigen::Index, 1> along_class = {1};
    mean.device(g_device) = x.mean(along_class);

    // x_zero_centered(B, D) = x.colwise() - m.transpose()
    // x_zero_centered_square(B, D) = x_zero_centered.array().square()
    // var(B,) = x_zero_centered_square.rowwise().mean()
    // std(B,) = (var + eps).sqrt()
    // rstd(B,) = 1.f / std;

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};
    rstd.device(g_device) =
        ((x - mean.reshape(batch_by_one).broadcast(one_by_class))
             .square()
             .mean(along_class) +
         eps_)
            .sqrt()
            .inverse();

    // normalize: (x - mean) / std
    // && scale:  (x - mean) / std * weight
    // && shift:  (x - mean) / std * weight + bias

    auto weight_1d = MakeFlat(weight_->data<T>(), normalized_shape_);
    auto bias_1d = MakeFlat(bias_->data<T>(), normalized_shape_);
    y.device(g_device) =
        (x - mean.reshape(batch_by_one).broadcast(one_by_class)) *
            rstd.reshape(batch_by_one).broadcast(one_by_class) *
            weight_1d.reshape(one_by_class).broadcast(batch_by_one) +
        bias_1d.reshape(one_by_class).broadcast(batch_by_one);
  }

  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::ConstFlat mean,
                typename TTypes<T>::ConstFlat rstd,
                typename TTypes<T>::Matrix x_grad) {
    // x: [B, D], y_grad: [B, D], x_grad: [B, D]
    CHECK_EQ(x.dimension(1), normalized_shape_);
    CHECK_EQ(y_grad.dimension(1), normalized_shape_);
    CHECK_EQ(x_grad.dimension(1), normalized_shape_);
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));
    int B = x.dimension(0), D = x.dimension(1);

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    // Lazily allocate the memory for gradients
    weight_->LazyAllocateGradient();
    bias_->LazyAllocateGradient();
    auto weight_1d = MakeFlat(weight_->data<T>(), normalized_shape_);
    auto weight_grad_1d = MakeFlat(weight_->grad<T>(), normalized_shape_);
    auto bias_1d = MakeFlat(bias_->data<T>(), normalized_shape_);
    auto bias_grad_1d = MakeFlat(bias_->grad<T>(), normalized_shape_);

    // x_grad = dL/dy * dy/dnorm
    //                * [dnorm/dxmean * dxmean/dx
    //                  + dnorm/dmean * dmean/dx
    //                  + dnorm/dstd * dstd/dx
    //                  ]

    // Eigen::Tensor<float, 2, Eigen::RowMajor>
    norm_->LazyAllocate(B * D);
    dnorm_->LazyAllocate(B * D);
    dnorm_mean_->LazyAllocate(B);
    dnorm_norm_mean_->LazyAllocate(B);
    auto norm_2d = norm_->matrix<T>(B, D);
    auto dnorm_2d = dnorm_->matrix<T>(B, D);
    auto dnorm_mean_1d = dnorm_mean_->flat<T>();
    auto dnorm_norm_mean_1d = dnorm_norm_mean_->flat<T>();
    norm_2d.device(g_device) =
        (x - mean.reshape(batch_by_one).broadcast(one_by_class)) *
        rstd.reshape(batch_by_one).broadcast(one_by_class);  // [B, D]
    dnorm_2d.device(g_device) =
        y_grad *
        weight_1d.reshape(one_by_class).broadcast(batch_by_one);  // [B, D]
    Eigen::array<Eigen::Index, 1> along_class = {1};
    dnorm_mean_1d.device(g_device) = dnorm_2d.mean(along_class);  // [B,]
    dnorm_norm_mean_1d.device(g_device) =
        (dnorm_2d * norm_2d).mean(along_class);  // [B,]
    x_grad.device(g_device) +=
        ((dnorm_2d -
          dnorm_mean_1d.reshape(batch_by_one).broadcast(one_by_class)) -
         norm_2d *
             dnorm_norm_mean_1d.reshape(batch_by_one).broadcast(one_by_class)) *
        rstd.reshape(batch_by_one).broadcast(one_by_class);

    // w_grad = dL/dy * dy/dw
    //        = dL/dy * x_norm(B,D)
    //        = \sum_i^B [y_grad(B, D) \elewise_dot x_norm(B, D)]

    Eigen::array<Eigen::Index, 1> along_batch = {0};
    weight_grad_1d.device(g_device) += (y_grad * norm_2d).sum(along_batch);

    // b_grad = dL/dy * dy/db
    //        = \sum_i^(B)(y_grad(B, D))
    //        = [D,]

    bias_grad_1d.device(g_device) += y_grad.sum(along_batch);
  }

  size_t NumParameters() const { return normalized_shape_ * 2; }

  size_t NumActivations() const {
    return norm_->size() + dnorm_->size() + dnorm_mean_->size() +
           dnorm_norm_mean_->size();
  }

  void Parameters(std::vector<Parameter*>* parameters) const {
    parameters->push_back(weight_.get());
    parameters->push_back(bias_.get());
  }

  int normalized_shape_;
  float eps_;
  std::unique_ptr<Parameter> weight_;
  std::unique_ptr<Parameter> bias_;

  // activation gradient tensor
  std::unique_ptr<Parameter> norm_;             // [B, D]
  std::unique_ptr<Parameter> dnorm_;            // [B, D]
  std::unique_ptr<Parameter> dnorm_mean_;       // [B,]
  std::unique_ptr<Parameter> dnorm_norm_mean_;  // [B,]
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_