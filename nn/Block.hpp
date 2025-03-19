#ifndef LLM_CPP__BLOCK_HPP_
#define LLM_CPP__BLOCK_HPP_

#include "nn.hpp"
#include "AttentionLayer.hpp"
#include "MLP.hpp"

namespace gpt {

struct Block {
  using Type = floatX;

  Block(int block_size, int n_head, int n_embed) {
    ln1_ = std::make_unique<nn::LayerNorm>(n_embed);
    attn_ = std::make_unique<CausalSelfAttention>(block_size, n_head, n_embed);
    ln2_ = std::make_unique<nn::LayerNorm>(n_embed);
    mlp_ = std::make_unique<MLP>(n_embed);

    // activation
    auto dtype = nn::DataTypeToEnum<Type>::value;
    ln1_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    ln1_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    ln1_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    att_y_ = std::make_unique<nn::Activation>(dtype);      // [B, T, C]
    residual1_ = std::make_unique<nn::Activation>(dtype);  // [B, T, C]
    ln2_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    ln2_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    ln2_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    mlp_y_ = std::make_unique<nn::Activation>(dtype);      // [B, T, C]
  }

  void Forward(typename TTypes<Type, 3>::ConstTensor x,
               typename TTypes<Type, 3>::Tensor y) {
    PROFILE_TRACE_FN("Block");

    // x: [B, T, C], y: [B, T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK_EQ(C, y.dimension(2));

    ln1_y_->LazyAllocate(B * T * C);
    ln1_mean_->LazyAllocate(B * T);
    ln1_rstd_->LazyAllocate(B * T);
    att_y_->LazyAllocate(B * T * C);
    residual1_->LazyAllocate(B * T * C);
    ln2_y_->LazyAllocate(B * T * C);
    ln2_mean_->LazyAllocate(B * T);
    ln2_rstd_->LazyAllocate(B * T);
    mlp_y_->LazyAllocate(B * T * C);

    // LN1
    auto x_2d = MakeConstMatrix(x.data(), B * T, C);
    auto ln1_y_2d = MakeMatrix(ln1_y_->data<Type>(), B * T, C);
    auto ln1_mean_1d = MakeFlat(ln1_mean_->data<Type>(), B * T);
    auto ln1_rstd_1d = MakeFlat(ln1_rstd_->data<Type>(), B * T);
    ln1_->Forward(x_2d, ln1_y_2d, ln1_mean_1d, ln1_rstd_1d);

    // Attention
    auto ln1_y_3d = MakeConst3DTensor(ln1_y_2d.data(), B, T, C);
    auto att_y_3d = Make3DTensor(att_y_->data<Type>(), B, T, C);
    attn_->Forward(ln1_y_3d, att_y_3d);

    // Residual
    auto x_1d = MakeConstFlat(x.data(), B * T * C);
    auto att_y_1d = MakeConstFlat(att_y_->data<Type>(), B * T * C);
    auto residual1_1d = MakeFlat(residual1_->data<Type>(), residual1_->size());
    nn::Residual::Forward(x_1d, att_y_1d, residual1_1d);

    // LN2
    auto ln2_y_2d = MakeMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_y_2d_const = MakeConstMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_mean_1d = MakeFlat(ln2_mean_->data<Type>(), B * T);
    auto ln2_rstd_1d = MakeFlat(ln2_rstd_->data<Type>(), B * T);
    auto residual1_2d = MakeConstMatrix(residual1_->data<Type>(), B * T, C);
    ln2_->Forward(residual1_2d, ln2_y_2d, ln2_mean_1d, ln2_rstd_1d);

    // MLP
    auto mlp_y_2d = MakeMatrix(mlp_y_->data<Type>(), B * T, C);
    mlp_->Forward(ln2_y_2d_const, mlp_y_2d);

    // Residual
    auto residual1_1d_const =
        MakeConstFlat(residual1_->data<Type>(), residual1_->size());
    auto mlp_y_1d = MakeConstFlat(mlp_y_->data<Type>(), B * T * C);
    auto y_1d = MakeFlat(y.data(), y.size());
    nn::Residual::Forward(residual1_1d_const, mlp_y_1d, y_1d);
  }

  void Backward(typename TTypes<Type, 3>::ConstTensor x,
                typename TTypes<Type, 3>::ConstTensor y_grad,
                typename TTypes<Type, 3>::Tensor x_grad) {
    PROFILE_TRACE_FN("Block");

    // x: [B, T, C], y_grad: [B, T, C], x_grad: [B, T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y_grad.dimension(0));
    CHECK_EQ(T, y_grad.dimension(1));
    CHECK_EQ(C, y_grad.dimension(2));
    CHECK_EQ(B, x_grad.dimension(0));
    CHECK_EQ(T, x_grad.dimension(1));
    CHECK_EQ(C, x_grad.dimension(2));

    ln1_y_->LazyAllocateGradient();
    att_y_->LazyAllocateGradient();
    residual1_->LazyAllocateGradient();
    ln2_y_->LazyAllocateGradient();
    mlp_y_->LazyAllocateGradient();
    ln1_y_->ZeroGrad();
    att_y_->ZeroGrad();
    residual1_->ZeroGrad();
    ln2_y_->ZeroGrad();
    mlp_y_->ZeroGrad();

    // backward residual
    auto y_grad_1d = MakeConstFlat(y_grad.data(), y_grad.size());
    auto residual1_grad_1d =
        MakeFlat(residual1_->grad<Type>(), residual1_->size());
    auto mlp_y_grad_1d = MakeFlat(mlp_y_->grad<Type>(), mlp_y_->size());
    nn::Residual::Backward(y_grad_1d, residual1_grad_1d, mlp_y_grad_1d);

    // backward MLP
    auto ln2_y_2d = MakeConstMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_y_grad_2d = MakeMatrix(ln2_y_->grad<Type>(), B * T, C);
    auto mlp_y_grad_2d = MakeConstMatrix(mlp_y_->grad<Type>(), B * T, C);
    mlp_->Backward(ln2_y_2d, mlp_y_grad_2d, ln2_y_grad_2d);

    // backward LN2
    auto ln2_mean_1d = MakeConstFlat(ln2_mean_->data<Type>(), B * T);
    auto ln2_rstd_1d = MakeConstFlat(ln2_rstd_->data<Type>(), B * T);
    auto residual1_2d = MakeConstMatrix(residual1_->data<Type>(), B * T, C);
    auto ln2_y_grad_2d_const = MakeConstMatrix(ln2_y_->grad<Type>(), B * T, C);
    auto residual1_grad_2d = MakeMatrix(residual1_->grad<Type>(), B * T, C);
    ln2_->Backward(residual1_2d, ln2_y_grad_2d_const, ln2_mean_1d, ln2_rstd_1d,
                   residual1_grad_2d);

    // backward residual
    auto residual1_grad_1d_const =
        MakeConstFlat(residual1_->grad<Type>(), residual1_->size());
    auto x_grad_1d = MakeFlat(x_grad.data(), x_grad.size());
    auto att_y_grad_1d = MakeFlat(att_y_->grad<Type>(), att_y_->size());
    nn::Residual::Backward(residual1_grad_1d_const, x_grad_1d, att_y_grad_1d);

    // backward attention
    auto ln1_y_3d = MakeConst3DTensor(ln1_y_->data<Type>(), B, T, C);
    auto ln1_y_grad_3d = Make3DTensor(ln1_y_->grad<Type>(), B, T, C);
    auto att_y_grad_3d = MakeConst3DTensor(att_y_->grad<Type>(), B, T, C);
    attn_->Backward(ln1_y_3d, att_y_grad_3d, ln1_y_grad_3d);

    // backward LN1
    auto x_2d = MakeConstMatrix(x.data(), B * T, C);
    auto ln1_mean_1d = MakeConstFlat(ln1_mean_->data<Type>(), B * T);
    auto ln1_rstd_1d = MakeConstFlat(ln1_rstd_->data<Type>(), B * T);
    auto ln1_y_grad_2d = MakeConstMatrix(ln1_y_->grad<Type>(), B * T, C);
    auto x_grad_2d = MakeMatrix(x_grad.data(), B * T, C);
    ln1_->Backward(x_2d, ln1_y_grad_2d, ln1_mean_1d, ln1_rstd_1d, x_grad_2d);
  }

  size_t NumParameters() const {
    return ln1_->NumParameters() + attn_->NumParameters() +
           ln2_->NumParameters() + mlp_->NumParameters();
  }

  size_t NumActivations() const {
    return ln1_->NumActivations() + attn_->NumActivations() +
           ln2_->NumActivations() + mlp_->NumActivations() + ln1_y_->size() +
           ln1_mean_->size() + ln2_rstd_->size() + att_y_->size() +
           residual1_->size() + ln2_y_->size() + ln2_mean_->size() +
           ln2_rstd_->size() + mlp_y_->size();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    ln1_->Parameters(parameters);
    attn_->Parameters(parameters);
    ln2_->Parameters(parameters);
    mlp_->Parameters(parameters);
  }

  std::unique_ptr<nn::LayerNorm> ln1_;
  std::unique_ptr<CausalSelfAttention> attn_;
  std::unique_ptr<nn::LayerNorm> ln2_;
  std::unique_ptr<MLP> mlp_;

  // activation tensors
  std::unique_ptr<nn::Activation> ln1_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> ln1_mean_, ln1_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> att_y_;                // [B, T, C]
  std::unique_ptr<nn::Activation> residual1_;            // [B, T, C]
  std::unique_ptr<nn::Activation> ln2_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> ln2_mean_, ln2_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> mlp_y_;                // [B, T, C]
};

}  // namespace gpt

#endif  // LLM_CPP__BLOCK_HPP_