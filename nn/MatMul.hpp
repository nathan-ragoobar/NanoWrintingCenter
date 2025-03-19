#ifndef LLM_CPP__MATMUL_HPP_
#define LLM_CPP__MATMUL_HPP_

#include "eigen/Eigen/Core"//<Eigen/Core>
#include "eigen/unsupported/Eigen/CXX11/Tensor"//<unsupported/Eigen/CXX11/Tensor>
#include "absl/log/check.h"
#include "tensor/tensor_util.hpp"
#include "Parameter.hpp"


namespace nn {

struct MatMul {
  using T = floatX;

  static void Forward(typename TTypes<T>::ConstMatrix x1,
                      typename TTypes<T>::ConstMatrix x2,
                      typename TTypes<T>::Matrix y, T scale = 1.0f) {
    // x: [M, N], x2: [N, K], y: [M, K]
    CHECK_EQ(x1.dimension(0), y.dimension(0));
    CHECK_EQ(x1.dimension(1), x2.dimension(0));
    CHECK_EQ(x2.dimension(1), y.dimension(1));

    // y = x1 * x2
    //    y.noalias() = x1 * x2;
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    y.device(g_device) = x1.contract(x2, product_dims) * scale;
  }

  static void Backward(typename TTypes<T>::ConstMatrix x1,
                       typename TTypes<T>::ConstMatrix x2,
                       typename TTypes<T>::ConstMatrix y_grad,
                       typename TTypes<T>::Matrix x1_grad,
                       typename TTypes<T>::Matrix x2_grad, T scale = 1.0) {
    // input:
    // x1: [M, N], x2:[N, K]
    // y_grad: [M, K]
    //
    // output:
    // x1_grad: [M, N], x2_grad: [N, K]
    int M = x1.dimension(0), N = x1.dimension(1), K = x2.dimension(1);
    CHECK(M == y_grad.dimension(0) && M == x1_grad.dimension(0));
    CHECK(N == x2.dimension(0) && N == x1_grad.dimension(1) &&
          N == x2_grad.dimension(0));
    CHECK(K == y_grad.dimension(1) && K == x2_grad.dimension(1));

    // x1_grad = dL/dy * dy/dx1
    //        = y_grad(M, K) * x2^T (K, N)
    //        = [M, N]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    x1_grad.device(g_device) += y_grad.contract(x2, product_dims) * scale;

    // x2_grad = dL/dy * dy/dx2
    //        = x1^T(N, M) * y_grad(M, K)
    //        = [N, K]

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    x2_grad.device(g_device) += x1.contract(y_grad, product_dims2) * scale;
  }
};

}  // namespace nn

#endif  // LLM_CPP__NN_HPP_