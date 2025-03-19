#ifndef LLM_CPP__ATTENTION_HPP_
#define LLM_CPP__ATTENTION_HPP_

#include <cmath>
#include <memory>
#include <vector>
#include "tensor/tensor_util.hpp"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "nn.hpp"  // Include the Parameter header
//#include "nnhead.hpp"
//#include "Linear.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {

struct CausalSelfAttention {
  using Type = floatX;

  CausalSelfAttention(int block_size, int n_head, int n_embed)
      : block_size_(block_size), n_head_(n_head), n_embed_(n_embed) {
    CHECK_EQ(n_embed % n_head, 0);

    // key, query, value projections for all heads, but in a batch
    c_attn_ = std::make_unique<nn::Linear>(n_embed, 3 * n_embed);

    // output projection
    c_proj_ = std::make_unique<nn::Linear>(n_embed, n_embed);

    // mask
    auto dtype = nn::DataTypeToEnum<Type>::value;
    bias_ = std::make_unique<nn::Parameter>(dtype, block_size * block_size);
    auto bias_2d = bias_->matrix<Type>(block_size, block_size);
    nn::UpperTriangularWithNegativeInf(bias_2d);

    // activation tensors
    qkv_ = std::make_unique<nn::Activation>(dtype);     // [B, T, 3C]
    q_ = std::make_unique<nn::Activation>(dtype);       // [B, NH, T, HS]
    k_ = std::make_unique<nn::Activation>(dtype);       // [B, NH, HS, T]
    v_ = std::make_unique<nn::Activation>(dtype);       // [B, NH, T, HS]
    preatt_ = std::make_unique<nn::Activation>(dtype);  // [B, NH, T, T]
    preatt_softmax_ = std::make_unique<nn::Activation>(dtype);  // [B, NH, T, T]
    att_ = std::make_unique<nn::Activation>(dtype);   // [B, NH, T, HS]
    att2_ = std::make_unique<nn::Activation>(dtype);  // [B, T, NH, HS]
  }

  void Forward(typename TTypes<Type, 3>::ConstTensor x,
               typename TTypes<Type, 3>::Tensor y) {
    PROFILE_TRACE_FN("CausalSelfAttention");

    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    //LOG_FIRST_N(INFO, 1) << "B: " << B << ", T: " << T << ", C: " << C;
    int NH = n_head_, HS = C / n_head_;
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK(C == n_embed_ && C == y.dimension(2));

    // Lazily allocate the memory for activation
    qkv_->LazyAllocate(B * T * 3 * C);
    q_->LazyAllocate(B * NH * T * HS);
    k_->LazyAllocate(B * NH * HS * T);
    v_->LazyAllocate(B * NH * T * HS);
    preatt_->LazyAllocate(B * NH * T * T);
    preatt_softmax_->LazyAllocate(B * NH * T * T);
    att_->LazyAllocate(B * NH * T * HS);
    att2_->LazyAllocate(B * T * NH * HS);

    auto _x = MakeConstMatrix(x.data(), B * T, C);
    auto qkv = MakeMatrix(qkv_->data<Type>(), B * T, 3 * C);
    c_attn_->Forward(_x, qkv);

    Eigen::array<Eigen::Index, 3> offsets_q = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offsets_k = {0, 0, C};
    Eigen::array<Eigen::Index, 3> offsets_v = {0, 0, 2 * C};
    Eigen::array<Eigen::Index, 3> extents = {B, T, C};
    Eigen::array<Eigen::Index, 4> shape = {B, T, NH, HS};
    Eigen::array<Eigen::Index, 4> shuffle_qv = {0, 2, 1, 3},
                                  shuffle_k = {0, 2, 3, 1};
    auto qkv3d = qkv_->tensor_3d<Type>(B, T, 3 * C);
    auto q_4d = q_->tensor_4d<Type>(B, NH, T, HS);
    auto k_4d = k_->tensor_4d<Type>(B, NH, HS, T);
    auto v_4d = v_->tensor_4d<Type>(B, NH, T, HS);
    q_4d.device(nn::g_device) = qkv3d
                                    .slice(offsets_q, extents)  // [B, T, C]
                                    .reshape(shape)       // [B, T, NH, HS]
                                    .shuffle(shuffle_qv)  // [B, NH, T, HS]
        ;
    k_4d.device(nn::g_device) = qkv3d
                                    .slice(offsets_k, extents)  // [B, T, C]
                                    .reshape(shape)      //  [B, T, NH, HS]
                                    .shuffle(shuffle_k)  //  [B, NH, HS, T]
        ;
    v_4d.device(nn::g_device) = qkv3d
                                    .slice(offsets_v, extents)  // [B, T, C]
                                    .reshape(shape)       //  [B, T, NH, HS]
                                    .shuffle(shuffle_qv)  //  [B, NH, T, HS]
        ;

    const float factor = 1.0f / std::sqrt(static_cast<float>(HS));
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < NH; ++h) {
        auto q2d =
            MakeConstMatrix(q_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto k2d =
            MakeConstMatrix(k_->data<Type>() + (b * NH + h) * HS * T, HS, T);
        auto v2d = MakeMatrix(v_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto preatt2d =
            MakeMatrix(preatt_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax2d = MakeConstMatrix(
            preatt_softmax_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto att2d =
            MakeMatrix(att_->data<Type>() + (b * NH + h) * T * HS, T, HS);

        nn::MatMul::Forward(q2d, k2d, preatt2d, factor);
        auto bias_2d = bias_->matrix<Type>(block_size_, block_size_);
        Eigen::array<Eigen::Index, 2> offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> extents = {T, T};
        preatt2d.device(nn::g_device) =
            preatt2d + bias_2d.slice(offsets, extents);

        // softmax
        auto preatt2d_tensor =
            MakeConstMatrix(preatt_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax2d_tensor = MakeMatrix(
            preatt_softmax_->data<Type>() + (b * NH + h) * T * T, T, T);
        nn::Softmax::Forward(preatt2d_tensor, preatt_softmax2d_tensor);

        // att * v
        typename TTypes<Type>::ConstMatrix v2d_const =
            MakeConstMatrix(v_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        nn::MatMul::Forward(preatt_softmax2d, v2d_const, att2d);
      }
    }

    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    auto att_4d = att_->tensor_4d<Type>(B, NH, T, HS);
    auto att2_4d = att2_->tensor_4d<Type>(B, T, NH, HS);
    att2_4d.device(nn::g_device) =
        att_4d.shuffle(shuffle_att);  // [B, T, NH, HS]
    auto att2_2d = MakeConstMatrix(att2_->data<Type>(), B * T, C);
    auto y2d = MakeMatrix(y.data(), B * T, C);
    c_proj_->Forward(att2_2d, y2d);
  }

  void Backward(typename TTypes<Type, 3>::ConstTensor x,
                typename TTypes<Type, 3>::ConstTensor y_grad,
                typename TTypes<Type, 3>::Tensor x_grad) {
    PROFILE_TRACE_FN("CausalSelfAttention");

    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    int NH = n_head_, HS = C / n_head_;
    CHECK(B == y_grad.dimension(0) && B == x_grad.dimension(0));
    CHECK(T == y_grad.dimension(1) && T == x_grad.dimension(1));
    CHECK(C == y_grad.dimension(2) && C == x_grad.dimension(2));

    // Lazily allocate the memory for activation
    qkv_->LazyAllocateGradient();
    q_->LazyAllocateGradient();
    k_->LazyAllocateGradient();
    v_->LazyAllocateGradient();
    preatt_->LazyAllocateGradient();
    preatt_softmax_->LazyAllocateGradient();
    att_->LazyAllocateGradient();
    att2_->LazyAllocateGradient();
    qkv_->ZeroGrad();
    q_->ZeroGrad();
    k_->ZeroGrad();
    v_->ZeroGrad();
    preatt_->ZeroGrad();
    preatt_softmax_->ZeroGrad();
    att_->ZeroGrad();
    att2_->ZeroGrad();

    // attproj backward
    auto att2_2d = MakeConstMatrix(att2_->data<Type>(), B * T, C);
    auto y_grad_2d = MakeConstMatrix(y_grad.data(), B * T, C);
    auto att2_grad_2d = MakeMatrix(att2_->grad<Type>(), B * T, C);
    c_proj_->Backward(att2_2d, y_grad_2d, att2_grad_2d);

    // shuffle backward
    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    auto att_grad = att_->tensor_4d_grad<Type>(B, NH, T, HS);
    auto att2_grad = att2_->tensor_4d_grad<Type>(B, T, NH, HS);
    att_grad.device(nn::g_device) =
        att2_grad.shuffle(shuffle_att);  // [B, NH, T, HS]

    // attention backward
    float factor = 1.0f / std::sqrt(static_cast<float>(HS));
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < NH; ++h) {
        auto q2d =
            MakeConstMatrix(q_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto q_grad2d =
            MakeMatrix(q_->grad<Type>() + (b * NH + h) * T * HS, T, HS);
        auto k2d =
            MakeConstMatrix(k_->data<Type>() + (b * NH + h) * HS * T, HS, T);
        auto k_grad2d =
            MakeMatrix(k_->grad<Type>() + (b * NH + h) * HS * T, HS, T);
        auto v2d =
            MakeConstMatrix(v_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto v_grad2d =
            MakeMatrix(v_->grad<Type>() + (b * NH + h) * T * HS, T, HS);
        auto preatt2d =
            MakeConstMatrix(preatt_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax2d = MakeConstMatrix(
            preatt_softmax_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_grad2d =
            MakeMatrix(preatt_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_grad2d_const =
            MakeConstMatrix(preatt_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax_grad2d = MakeMatrix(
            preatt_softmax_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax_grad2d_const = MakeConstMatrix(
            preatt_softmax_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto att_grad2d =
            MakeConstMatrix(att_->grad<Type>() + (b * NH + h) * T * HS, T, HS);

        // backward: att * v
        nn::MatMul::Backward(preatt_softmax2d, v2d, att_grad2d,
                             preatt_softmax_grad2d, v_grad2d);

        // backward: softmax
        nn::Softmax::Backward(preatt_softmax2d, preatt_softmax_grad2d_const,
                              preatt_grad2d);

        // backward: mask
        // backward: q * k
        nn::MatMul::Backward(q2d, k2d, preatt_grad2d_const, q_grad2d, k_grad2d,
                             factor);
      }
    }

    // backward: shuffle -> reshape
    Eigen::array<Eigen::Index, 3> offsets_q = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offsets_k = {0, 0, C};
    Eigen::array<Eigen::Index, 3> offsets_v = {0, 0, 2 * C};
    Eigen::array<Eigen::Index, 3> extents = {B, T, C};
    Eigen::array<Eigen::Index, 3> shape = {B, T, C};
    Eigen::array<Eigen::Index, 4> shuffle_qv = {0, 2, 1, 3},
                                  shuffle_k = {0, 3, 1, 2};
    // q_grad_: [B, NH, T, HS] -> [B, T, NH, HS] -> [B, T, C]
    auto qkv_grad = qkv_->tensor_3d_grad<Type>(B, T, 3 * C);
    auto q_grad = q_->tensor_4d_grad<Type>(B, NH, T, HS);
    auto k_grad = k_->tensor_4d_grad<Type>(B, NH, HS, T);
    auto v_grad = v_->tensor_4d_grad<Type>(B, NH, T, HS);
    qkv_grad.slice(offsets_q, extents).device(nn::g_device) =
        q_grad.shuffle(shuffle_qv).reshape(shape);

    // k_grad_: [B, NH, HS, T] -> [B, T, NH, HS] -> [B, T, C]
    qkv_grad.slice(offsets_k, extents).device(nn::g_device) =
        k_grad.shuffle(shuffle_k).reshape(shape);

    // v_grad_: [B, NH, T, HS] -> [B, T, NH, HS] -> [B, T, C]
    qkv_grad.slice(offsets_v, extents).device(nn::g_device) =
        v_grad.shuffle(shuffle_qv).reshape(shape);

    // backward: qkv
    auto _x = MakeConstMatrix(x.data(), B * T, C);
    auto qkv_grad_2d = MakeConstMatrix(qkv_->grad<Type>(), B * T, 3 * C);
    auto _x_grad = MakeMatrix(x_grad.data(), B * T, C);
    c_attn_->Backward(_x, qkv_grad_2d, _x_grad);
  }

  size_t NumParameters() const {
    return c_attn_->NumParameters() + c_proj_->NumParameters();
  }

  size_t NumActivations() const {
    size_t num_activations =
        c_attn_->NumActivations() + c_proj_->NumActivations() + qkv_->size() +
        q_->size() + k_->size() + v_->size() + preatt_->size() +
        preatt_softmax_->size() + att_->size() + att2_->size();
    num_activations += bias_->size();
    return num_activations;
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    c_attn_->Parameters(parameters);
    c_proj_->Parameters(parameters);
  }

  int block_size_;
  int n_head_;
  int n_embed_;
  std::unique_ptr<nn::Linear> c_attn_;
  std::unique_ptr<nn::Linear> c_proj_;

  // activation tensors
  std::unique_ptr<nn::Activation> qkv_;             // [B, T, 3C]
  std::unique_ptr<nn::Activation> q_;               // [B, NH, T, HS]
  std::unique_ptr<nn::Activation> k_;               // [B, NH, HS, T]
  std::unique_ptr<nn::Activation> v_;               // [B, NH, T, HS]
  std::unique_ptr<nn::Activation> preatt_;          // [B, NH, T, T]
  std::unique_ptr<nn::Activation> preatt_softmax_;  // [B, NH, T, T]
  std::unique_ptr<nn::Activation> att_;             // [B, NH, T, HS]
  std::unique_ptr<nn::Activation> att2_;            // [B, T, NH, HS]

  // not really a 'bias', more of a mask, but following the OpenAI/HF naming
  // though
  //  Eigen::MatrixXi bias_;
  std::unique_ptr<nn::Activation> bias_;  // [block_size, block_size]
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_