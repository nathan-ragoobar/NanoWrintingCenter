#ifndef LLM_CPP__MLP_HPP_
#define LLM_CPP__MLP_HPP_

#include "nn.hpp"
#include "LoRALinear.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {

  struct MLP {
    using T = floatX;
  
    // Constructor with LoRA support
    explicit MLP(int n_embed, bool use_lora = false, int lora_rank = 8, float lora_alpha = 16.0f) 
        : n_embed_(n_embed), use_lora_(use_lora) {
      
      if (use_lora) {
        // Use LoRA linear layers
        c_fc_lora_ = std::make_unique<nn::LoRALinear>(n_embed, 4 * n_embed, lora_rank, lora_alpha);
        c_proj_lora_ = std::make_unique<nn::LoRALinear>(4 * n_embed, n_embed, lora_rank, lora_alpha);
      } else {
        // Use standard linear layers
        c_fc_ = std::make_unique<nn::Linear>(n_embed, 4 * n_embed);
        c_proj_ = std::make_unique<nn::Linear>(4 * n_embed, n_embed);
      }
  
      // activation
      auto dtype = nn::DataTypeToEnum<T>::value;
      fch_ = std::make_unique<nn::Activation>(dtype);
      fch_gelu_ = std::make_unique<nn::Activation>(dtype);
    }
  
    // Modified Forward pass to use LoRA if enabled
    void Forward(typename TTypes<T>::ConstMatrix x,
                 typename TTypes<T>::Matrix y) {
      PROFILE_TRACE_FN("MLP");
  
      CHECK_EQ(x.dimension(1), n_embed_);
      CHECK_EQ(x.dimension(0), y.dimension(0));
      CHECK_EQ(x.dimension(1), y.dimension(1));
  
      int BT = x.dimension(0);
      fch_->LazyAllocate(BT * 4 * n_embed_);
      fch_gelu_->LazyAllocate(BT * 4 * n_embed_);
  
      auto fch = fch_->matrix<T>(BT, 4 * n_embed_);
      auto fch_gelu = fch_gelu_->matrix<T>(BT, 4 * n_embed_);
      
      // Use LoRA or standard layers based on use_lora_ flag
      if (use_lora_) {
        c_fc_lora_->Forward(x, fch);
      } else {
        c_fc_->Forward(x, fch);
      }
      
      nn::NewGELU::Forward(MakeConstFlat(fch.data(), fch.size()),
                           MakeFlat(fch_gelu.data(), fch_gelu.size()));
                           
      auto fch_gelu_const = fch_gelu_->const_matrix<T>(BT, 4 * n_embed_);
      
      if (use_lora_) {
        c_proj_lora_->Forward(fch_gelu_const, y);
      } else {
        c_proj_->Forward(fch_gelu_const, y);
      }
    }
  
    // Modified Backward pass to use LoRA if enabled
    void Backward(typename TTypes<T>::ConstMatrix x,
                  typename TTypes<T>::ConstMatrix y_grad,
                  typename TTypes<T>::Matrix x_grad) {
      PROFILE_TRACE_FN("MLP");
  
      CHECK_EQ(x.dimension(1), n_embed_);
      CHECK_EQ(x.dimension(0), y_grad.dimension(0));
      CHECK_EQ(x.dimension(1), y_grad.dimension(1));
      CHECK_EQ(x.dimension(0), x_grad.dimension(0));
      CHECK_EQ(x.dimension(1), x_grad.dimension(1));
  
      int BT = x.dimension(0);
      fch_->LazyAllocateGradient();
      fch_gelu_->LazyAllocateGradient();
      fch_->ZeroGrad();
      fch_gelu_->ZeroGrad();
  
      auto fch_gelu = fch_gelu_->const_matrix<T>(BT, 4 * n_embed_);
      auto fch_gelu_grad = fch_gelu_->matrix_grad<T>(BT, 4 * n_embed_);
      
      if (use_lora_) {
        c_proj_lora_->Backward(fch_gelu, y_grad, fch_gelu_grad);
      } else {
        c_proj_->Backward(fch_gelu, y_grad, fch_gelu_grad);
      }
  
      auto fch = fch_->const_flat<T>();
      auto fch_gelu_grad_flat = fch_gelu_->const_flat_grad<T>();
      auto fch_grad = fch_->flat_grad<T>();
      nn::NewGELU::Backward(fch, fch_gelu_grad_flat, fch_grad);
  
      auto fch_grad_2d = fch_->const_matrix_grad<T>(BT, 4 * n_embed_);
      
      if (use_lora_) {
        c_fc_lora_->Backward(x, fch_grad_2d, x_grad);
      } else {
        c_fc_->Backward(x, fch_grad_2d, x_grad);
      }
    }
  
    size_t NumParameters() const {
      if (use_lora_) {
        return c_fc_lora_->NumParameters() + c_proj_lora_->NumParameters();
      } else {
        return c_fc_->NumParameters() + c_proj_->NumParameters();
      }
    }
  
    size_t NumActivations() const {
      if (use_lora_) {
        return c_fc_lora_->NumActivations() + c_proj_lora_->NumActivations() + 
               fch_->size() + fch_gelu_->size();
      } else {
        return c_fc_->NumActivations() + c_proj_->NumActivations() + 
               fch_->size() + fch_gelu_->size();
      }
    }
  
    void Parameters(std::vector<nn::Parameter*>* parameters) const {
      if (use_lora_) {
        c_fc_lora_->Parameters(parameters);
        c_proj_lora_->Parameters(parameters);
      } else {
        c_fc_->Parameters(parameters);
        c_proj_->Parameters(parameters);
      }
    }
  
    // Method to freeze base model and train only LoRA parameters
    void FreezeBaseModel() {
      if (use_lora_) {
        c_fc_lora_->SetBaseTrainable(false);
        c_proj_lora_->SetBaseTrainable(false);
      }
    }
  
    int n_embed_;
    bool use_lora_;
    
    // Standard linear layers
    std::unique_ptr<nn::Linear> c_fc_;
    std::unique_ptr<nn::Linear> c_proj_;
    
    // LoRA linear layers
    std::unique_ptr<nn::LoRALinear> c_fc_lora_;
    std::unique_ptr<nn::LoRALinear> c_proj_lora_;
  
    // activation tensors
    std::unique_ptr<nn::Activation> fch_;
    std::unique_ptr<nn::Activation> fch_gelu_;
  };
  
  }  // namespace gpt
#endif  // LLM_CPP__NN_HPP_