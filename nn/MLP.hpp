#ifndef LLM_CPP__MLP_HPP_
#define LLM_CPP__MLP_HPP_

#include "nn.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {

struct MLP {
  using T = floatX;

  //I think this is the structure for the class/structure?
  explicit MLP(int n_embed) : n_embed_(n_embed) {
    c_fc_ = std::make_unique<nn::Linear>(n_embed, 4 * n_embed);
    c_proj_ = std::make_unique<nn::Linear>(4 * n_embed, n_embed);

    // activation
    auto dtype = nn::DataTypeToEnum<T>::value;
    fch_ = std::make_unique<nn::Activation>(dtype);
    fch_gelu_ = std::make_unique<nn::Activation>(dtype);
  }

  //This is the forward pass for the MLP
  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("MLP");

    // x: [B*T, 4*n_embed], y: [B*T, 4*n_embed]
    CHECK_EQ(x.dimension(1), n_embed_);
    // x.shape == y.shape
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(x.dimension(1), y.dimension(1));

    int BT = x.dimension(0);
    fch_->LazyAllocate(BT * 4 * n_embed_);
    fch_gelu_->LazyAllocate(BT * 4 * n_embed_);

    // forward
    auto fch = fch_->matrix<T>(BT, 4 * n_embed_);
    auto fch_gelu = fch_gelu_->matrix<T>(BT, 4 * n_embed_);
    c_fc_->Forward(x, fch);
    nn::NewGELU::Forward(MakeConstFlat(fch.data(), fch.size()),
                         MakeFlat(fch_gelu.data(), fch_gelu.size()));
    auto fch_gelu_const = fch_gelu_->const_matrix<T>(BT, 4 * n_embed_);
    c_proj_->Forward(fch_gelu_const, y);
  }

  //Backward pass for the MLP
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("MLP");

    // x: [B*T, 4*n_embed], y_grad: [B*T, 4*n_embed]
    // x_grad: [B*T, 4*n_embed]
    CHECK_EQ(x.dimension(1), n_embed_);
    // x.shape == y_grad.shape == x_grad.shape
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(1), y_grad.dimension(1));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));
    CHECK_EQ(x.dimension(1), x_grad.dimension(1));

    // Lazily allocate the memory for activation
    int BT = x.dimension(0);
    fch_->LazyAllocateGradient();
    fch_gelu_->LazyAllocateGradient();
    fch_->ZeroGrad();
    fch_gelu_->ZeroGrad();

    auto fch_gelu = fch_gelu_->const_matrix<T>(BT, 4 * n_embed_);
    auto fch_gelu_grad = fch_gelu_->matrix_grad<T>(BT, 4 * n_embed_);
    c_proj_->Backward(fch_gelu, y_grad, fch_gelu_grad);

    auto fch = fch_->const_flat<T>();
    auto fch_gelu_grad_flat = fch_gelu_->const_flat_grad<T>();
    auto fch_grad = fch_->flat_grad<T>();
    nn::NewGELU::Backward(fch, fch_gelu_grad_flat, fch_grad);

    auto fch_grad_2d = fch_->const_matrix_grad<T>(BT, 4 * n_embed_);
    c_fc_->Backward(x, fch_grad_2d, x_grad);
  }

  size_t NumParameters() const {
    return c_fc_->NumParameters() + c_proj_->NumParameters();
  }

  size_t NumActivations() const {
    return c_fc_->NumActivations() + c_proj_->NumActivations() + fch_->size() +
           fch_gelu_->size();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    c_fc_->Parameters(parameters);
    c_proj_->Parameters(parameters);
  }

  int n_embed_;
  std::unique_ptr<nn::Linear> c_fc_;
  std::unique_ptr<nn::Linear> c_proj_;

  // activation tensors
  std::unique_ptr<nn::Activation> fch_;       // [B*T, 4*C]
  std::unique_ptr<nn::Activation> fch_gelu_;  // [B*T, 4*C]
};

}  // namespace gpt

#endif  // LLM_CPP__NN_HPP_