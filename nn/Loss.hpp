#ifndef LLM_CPP__LOSS_HPP_
#define LLM_CPP__LOSS_HPP_

#include <cmath>
#include "Parameter.hpp"  // Includes necessary headers for nn::Parameter, etc.
#include "absl/types/span.h"
#include "absl/log/check.h"
#include "tensor/tensor_util.hpp"

namespace nn {

struct CrossEntropy {
  using T = floatX;
  enum Reduction { MEAN, SUM };

  CrossEntropy(Reduction reduction = Reduction::MEAN) : reduction_(reduction) {}

  void Forward(typename TTypes<T>::ConstMatrix probs,
               absl::Span<const int> targets, float* loss) {
    // probs:[B, C], targets: [B,] loss: scalar
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK_EQ(B, targets.size());

    // targets: [B,]
    for (int i = 0; i < targets.size(); ++i) {
      int ix = targets[i];
      *loss += -std::log(probs(i, ix));
    }

    if (reduction_ == Reduction::MEAN) {
      *loss /= static_cast<float>(B);
    }
  }

  void Backward(typename TTypes<T>::ConstMatrix probs,
                absl::Span<const int> targets,
                typename TTypes<T>::Matrix probs_grad) {
    // probs: [B, C], targets: [B,]
    // probs_grad: [B, C]
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK(B == targets.size() && B == probs_grad.dimension(0));
    CHECK_EQ(C, probs_grad.dimension(1));

    float factor =
        reduction_ == Reduction::MEAN ? 1.0f / static_cast<float>(B) : 1.0f;

    for (int b = 0; b < B; ++b) {
      int ix = targets[b];
      probs_grad(b, ix) += -1.0f / probs(b, ix) * factor;
    }
  }

  Reduction reduction_;
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_