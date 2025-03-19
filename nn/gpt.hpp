#ifndef LLM_CPP__GPT_HPP_
#define LLM_CPP__GPT_HPP_

#include "nn.hpp"
#include "MLP.hpp"
#include "AttentionLayer.hpp"
#include "Block.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {



struct GPT {
  using Type = floatX;

  GPT(int block_size, int vocab_size, int padded_vocab_size, int n_layer,
      int n_head, int n_embed)
      : block_size_(block_size),
        vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        n_layer_(n_layer),
        n_embed_(n_embed),
        lm_head_(nullptr),
        lm_head_grad_(nullptr) {
    CHECK_GT(n_layer, 0);

    wte_ = std::make_unique<nn::Embedding>(padded_vocab_size, n_embed);
    wpe_ = std::make_unique<nn::Embedding>(block_size, n_embed);
    for (int i = 0; i < n_layer; ++i) {
      h_.emplace_back(std::make_unique<Block>(block_size, n_head, n_embed));
    }
    lnf_ = std::make_unique<nn::LayerNorm>(n_embed);

    lm_head_unused_ = std::make_unique<nn::Linear>(n_embed, vocab_size);
    // https://paperswithcode.com/method/weight-tying
    nn::g_device.memcpy(wte_->weight_->data<Type>(),
                        lm_head_unused_->weight_->template data<Type>(),
                        sizeof(float) * vocab_size * n_embed);
    nn::g_device.memset(
        wte_->weight_->data<Type>() + vocab_size * n_embed, 0,
        sizeof(float) * (padded_vocab_size - vocab_size) * n_embed);
    lm_head_ = wte_->weight_->data<Type>();
    softmax_cross_entropy_ = std::make_unique<nn::SoftmaxCrossEntropy>();

    // activation
    auto dtype = nn::DataTypeToEnum<Type>::value;
    tok_emb_ = std::make_unique<nn::Activation>(dtype);    // [B, T, C]
    pos_emb_ = std::make_unique<nn::Activation>(dtype);    // [T, C]
    encoded_ = std::make_unique<nn::Activation>(dtype);    // [B, T, C]
    block_y_ = std::make_unique<nn::Activation>(dtype);    // [L, B, T, C]
    lnf_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    lnf_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    lnf_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    scratch_ = std::make_unique<nn::Activation>(dtype);    // [B*T]
    loss_ = std::make_unique<nn::Activation>(dtype);       // [B*T]
    loss_mean_ = std::make_unique<nn::Activation>(dtype);  // [1]
    probs_ = std::make_unique<nn::Activation>(dtype);      // [B*T, vocab_size]
    logits_grad_ =
        std::make_unique<nn::Activation>(dtype);  // [B*T, vocab_size]
  }

  void __Forward(typename TTypes<int>::ConstMatrix idx) {
    PROFILE_TRACE_FN("GPT");

    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_,
              L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;

    CHECK_LE(T, block_size_) << "Cannot forward sequence of length " << T
                             << ", block size is only " << block_size_;
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);

    // Lazily allocate memory
    tok_emb_->LazyAllocate(B * T * C);
    pos_emb_->LazyAllocate(T * C);
    encoded_->LazyAllocate(B * T * C);
    block_y_->LazyAllocate(L * B * T * C);
    lnf_y_->LazyAllocate(BT * C);
    lnf_mean_->LazyAllocate(BT);
    lnf_rstd_->LazyAllocate(BT);

    wte_->Forward(idx,
                  absl::MakeSpan(tok_emb_->data<Type>(), tok_emb_->size()));
    wpe_->Forward(pos,
                  absl::MakeSpan(pos_emb_->data<Type>(), pos_emb_->size()));

    auto tok_emb = tok_emb_->matrix<Type>(B, TC);
    auto pos_emb = pos_emb_->flat<Type>();
    auto encoded = encoded_->matrix<Type>(B, TC);
    Eigen::array<Eigen::Index, 2> batch_by_one = {B, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, TC};
    encoded.device(nn::g_device) =
        tok_emb + pos_emb.reshape(one_by_class).broadcast(batch_by_one);

    for (int l = 0; l < n_layer_; ++l) {
      const auto& block = h_[l];
      Type* x = l == 0 ? encoded_->data<Type>()
                       : block_y_->data<Type>() + (l - 1) * BTC;
      Type* y = block_y_->data<Type>() + l * BTC;
      auto block_x_3d = MakeConst3DTensor(x, B, T, C);
      auto block_y_3d = Make3DTensor(y, B, T, C);
      block->Forward(block_x_3d, block_y_3d);
    }

    auto block_out_2d =
        MakeConstMatrix(block_y_->data<Type>() + (L - 1) * BTC, BT, C);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lnf_mean = MakeFlat(lnf_mean_->data<Type>(), BT);
    auto lnf_rstd = MakeFlat(lnf_rstd_->data<Type>(), BT);
    lnf_->Forward(block_out_2d, lnf_y, lnf_mean, lnf_rstd);
  }

  void Forward(typename TTypes<int>::ConstMatrix idx,
               typename TTypes<Type, 3>::Tensor logits) {
    PROFILE_TRACE_FN("GPT");

    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    // OPTIMIZE:
    // inference-time mini-optimization: only forward the lm_head on the very
    // last position
    //    auto lnf_y_3d = Eigen::TensorMap<nn::Tensor3D>(lnf_y_.data(), B, T,
    //    C); nn::Tensor2D lnf_y_last_t = lnf_y_3d.chip(T - 1, 1);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, C);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);
    //    nn::MatMul::Forward(lnf_y, lm_head, logits_2d);
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);
  }

  void SoftmaxForwardCPU(typename TTypes<Type>::ConstMatrix logits,
                         absl::Span<const int> targets, float* loss) {
    PROFILE_TRACE_FN("GPT");

    int BT = logits.dimension(0);
    CHECK_EQ(BT, targets.size());
    CHECK_EQ(vocab_size_, logits.dimension(1));
    probs_->LazyAllocate(BT * vocab_size_);
    auto probs_2d = MakeMatrix(probs_->data<Type>(), BT, vocab_size_);
    softmax_cross_entropy_->Forward(logits, targets, probs_2d, loss);
  }

  void SoftmaxForwardGPU(typename TTypes<Type>::ConstMatrix logits,
                         typename TTypes<Type>::ConstMatrix labels,
                         float* loss) {
    PROFILE_TRACE_FN("GPT");

    int BT = logits.dimension(0);
    CHECK_EQ(BT, labels.dimension(0));
    CHECK_EQ(vocab_size_, logits.dimension(1));
    CHECK_EQ(vocab_size_, labels.dimension(1));
    scratch_->LazyAllocate(BT);
    loss_->LazyAllocate(BT);
    loss_mean_->LazyAllocate(1);
    logits_grad_->LazyAllocate(BT * vocab_size_);
    logits_grad_->ZeroData();
    auto logits_grad = MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    nn::SoftmaxCrossEntropy::ForwardAndBackward(
        logits, labels, scratch_->template flat<Type>(),
        loss_->template flat<Type>(), logits_grad);
    logits_grad.device(nn::g_device) = logits_grad * (1.0f / BT);

#ifdef EIGEN_USE_GPU
    TTypes<Type>::UnalignedScalar loss_mean(loss_mean_->data<Type>());
    loss_mean.device(nn::g_device) = loss_->template flat<Type>().mean();
    nn::g_device.memcpyDeviceToHost(loss, loss_mean.data(), sizeof(Type));
    nn::g_device.synchronize();
#else
    LOG(FATAL) << "Never reach here!!!";
#endif
    //    TTypes<float>::Scalar loss_scalar(loss);
    //    loss_scalar.device(nn::g_device) = loss_->template
    //    flat<Type>().mean();
  }

  void ForwardCPU(typename TTypes<int>::ConstMatrix idx,
                  typename TTypes<int>::ConstMatrix targets,
                  typename TTypes<Type, 3>::Tensor logits, float* loss) {
    PROFILE_TRACE_FN("GPT");

    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(targets.dimension(0) == B && targets.dimension(1) == T);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);
    auto probs_2d = MakeMatrix(probs_->data<Type>(), BT, vocab_size_);

    // [BT, C] x [C, vocab_size] -> [BT, vocab_size]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);

    auto logits_2d_const = MakeConstMatrix(logits.data(), BT, vocab_size_);
    SoftmaxForwardCPU(logits_2d_const, targets, loss);
  }

  void ForwardGPU(typename TTypes<int>::ConstMatrix idx,
                  typename TTypes<Type, 3>::ConstTensor labels,
                  typename TTypes<Type, 3>::Tensor logits, float* loss) {
    PROFILE_TRACE_FN("GPT");

    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(labels.dimension(0) == B && labels.dimension(1) == T &&
          labels.dimension(2) == vocab_size_);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);

    // [BT, C] x [C, vocab_size] -> [BT, vocab_size]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);

    auto logits_2d_const = MakeConstMatrix(logits.data(), BT, vocab_size_);
    auto labels_2d_const = MakeConstMatrix(labels.data(), BT, vocab_size_);
    SoftmaxForwardGPU(logits_2d_const, labels_2d_const, loss);
  }

  void SoftmaxBackwardCPU(absl::Span<const int> targets) {
    PROFILE_TRACE_FN("GPT");

    int BT = targets.size();
    logits_grad_->LazyAllocate(BT * vocab_size_);
    logits_grad_->ZeroData();
    auto probs_2d = MakeConstMatrix(probs_->data<Type>(), BT, vocab_size_);
    auto logits_grad_2d =
        MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    softmax_cross_entropy_->Backward(probs_2d, targets, logits_grad_2d);
  }

  void BackwardCPU(typename TTypes<int>::ConstMatrix idx,
                   typename TTypes<int>::ConstMatrix targets) {
    PROFILE_TRACE_FN("GPT");

    SoftmaxBackwardCPU(targets);
    BackwardGPU(idx);
  }

  void BackwardGPU(typename TTypes<int>::ConstMatrix idx) {
    PROFILE_TRACE_FN("GPT");

    // idx: [B, T], targets: [B, T]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_,
              L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;

    wte_->weight_->LazyAllocateGradient();
    if (lm_head_grad_ == nullptr) {
      lm_head_grad_ = wte_->weight_->grad<Type>();
    }

    tok_emb_->LazyAllocateGradient();
    pos_emb_->LazyAllocateGradient();
    encoded_->LazyAllocateGradient();
    block_y_->LazyAllocateGradient();
    lnf_y_->LazyAllocateGradient();

    tok_emb_->ZeroGrad();
    pos_emb_->ZeroGrad();
    encoded_->ZeroGrad();
    block_y_->ZeroGrad();
    lnf_y_->ZeroGrad();

    // backward lm_head
    auto logits_grad_2d =
        MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lnf_y_grad = MakeMatrix(lnf_y_->grad<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, C);
    auto lm_head_grad = MakeMatrix(lm_head_grad_, vocab_size_, C);
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    lnf_y_grad.device(nn::g_device) += logits_grad_2d.contract(
        lm_head, product_dims);  // [BT, vocab_size] x [vocab_size, C]
    lm_head_grad.device(nn::g_device) += logits_grad_2d.contract(
        lnf_y, product_dims2);  // [vocab_size, BT] x [BT, C]

    // backward LNF
    auto block_out_2d =
        MakeConstMatrix(block_y_->data<Type>() + (L - 1) * BTC, BT, C);
    auto block_out_grad_2d =
        MakeMatrix(block_y_->grad<Type>() + (L - 1) * BTC, BT, C);
    auto lnf_mean = MakeConstFlat(lnf_mean_->data<Type>(), BT);
    auto lnf_rstd = MakeConstFlat(lnf_rstd_->data<Type>(), BT);
    auto lnf_y_grad_2d = MakeConstMatrix(lnf_y_->grad<Type>(), BT, C);
    lnf_->Backward(block_out_2d, lnf_y_grad_2d, lnf_mean, lnf_rstd,
                   block_out_grad_2d);

    // backward blocks
    for (int l = n_layer_ - 1; l >= 0; --l) {
      const auto& block = h_[l];
      Type* x = l == 0 ? encoded_->data<Type>()
                       : block_y_->data<Type>() + (l - 1) * BTC;
      Type* x_grad = l == 0 ? encoded_->grad<Type>()
                            : block_y_->grad<Type>() + (l - 1) * BTC;
      Type* y_grad = block_y_->grad<Type>() + l * BTC;
      auto block_x_3d = MakeConst3DTensor(x, B, T, C);
      auto block_x_grad_3d = Make3DTensor(x_grad, B, T, C);
      auto block_y_grad_3d = MakeConst3DTensor(y_grad, B, T, C);
      block->Backward(block_x_3d, block_y_grad_3d, block_x_grad_3d);
    }

    // backward tok_emb, pos_emb
    auto encoded_grad = encoded_->matrix_grad<Type>(B, TC);
    auto tok_emb_grad = tok_emb_->matrix_grad<Type>(B, TC);
    auto pos_emb_grad = pos_emb_->flat_grad<Type>();
    Eigen::array<Eigen::Index, 1> along_batch = {0};
    tok_emb_grad.device(nn::g_device) = encoded_grad;
    pos_emb_grad.device(nn::g_device) = tok_emb_grad.sum(along_batch);

    // backward wte, wpe
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);
    wte_->Backward(idx, tok_emb_grad);
    wpe_->Backward(pos, pos_emb_grad);
  }

  size_t NumParameters() const {
    size_t num_parameters = 0;
    num_parameters += wte_->NumParameters();
    num_parameters += wpe_->NumParameters();
    for (const auto& b : h_) {
      num_parameters += b->NumParameters();
    }
    num_parameters += lnf_->NumParameters();
    return num_parameters;
  }

  size_t NumActivations() const {
    size_t num_activations = 0;
    num_activations += wte_->NumActivations();
    num_activations += wpe_->NumActivations();
    for (const auto& b : h_) {
      num_activations += b->NumActivations();
    }
    num_activations += lnf_->NumActivations();
    num_activations += tok_emb_->size();
    num_activations += pos_emb_->size();
    num_activations += encoded_->size();
    num_activations += block_y_->size();
    num_activations += lnf_y_->size();
    num_activations += lnf_mean_->size();
    num_activations += lnf_rstd_->size();
#ifdef EIGEN_USE_GPU
    num_activations += scratch_->size();
    num_activations += loss_->size();
    num_activations += loss_mean_->size();
#else
    num_activations += probs_->size();
#endif
    num_activations += logits_grad_->size();
    return num_activations;
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    wte_->Parameters(parameters);
    wpe_->Parameters(parameters);
    for (const auto& b : h_) {
      b->Parameters(parameters);
    }
    lnf_->Parameters(parameters);
  }

 public:
  int block_size_;
  int vocab_size_;
  int padded_vocab_size_;
  int n_layer_;
  int n_embed_;

  // transformer
  std::unique_ptr<nn::Embedding> wte_;
  std::unique_ptr<nn::Embedding> wpe_;
  std::vector<std::unique_ptr<Block>> h_;
  std::unique_ptr<nn::LayerNorm> lnf_;
  std::unique_ptr<nn::SoftmaxCrossEntropy> softmax_cross_entropy_;

  // head
  std::unique_ptr<nn::Linear> lm_head_unused_;
  Type *lm_head_, *lm_head_grad_;  // [vocal_size, C]

  // activation tensors and gradients
  std::unique_ptr<nn::Activation> tok_emb_;              // [B, T, C]
  std::unique_ptr<nn::Activation> pos_emb_;              // [T, C]
  std::unique_ptr<nn::Activation> encoded_;              // [B, T, C]
  std::unique_ptr<nn::Activation> block_y_;              // [L, B, T, C]
  std::unique_ptr<nn::Activation> lnf_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> lnf_mean_, lnf_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> probs_;                // [B*T, vocab_size]
  std::unique_ptr<nn::Activation> scratch_;              // [B*T]
  std::unique_ptr<nn::Activation> loss_;                 // [B*T]
  std::unique_ptr<nn::Activation> loss_mean_;            // [1]
  std::unique_ptr<nn::Activation> logits_grad_;          // [B*T, vocab_size]
};

}  // namespace gpt

#endif  // LLM_CPP__GPT_HPP_
