#ifndef LLM_CPP__SOFTMAX_HPP_
#define LLM_CPP__SOFTMAX_HPP_

#include <cmath>
#include "tensor/tensor_util.hpp"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "Parameter.hpp"


namespace nn {

struct Softmax {
  using T = floatX;

  static void Forward(typename TTypes<T>::ConstMatrix x,
                      typename TTypes<T>::Matrix y) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(x.dimension(1), y.dimension(1));

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    y.device(g_device) = (x - x.maximum(along_class)
                                  .eval()
                                  .reshape(batch_by_one)
                                  .broadcast(one_by_class))
                             .exp();
    y.device(g_device) = y * y.sum(along_class)
                                 .inverse()
                                 .eval()
                                 .reshape(batch_by_one)
                                 .broadcast(one_by_class);
  }

  static void Backward(typename TTypes<T>::ConstMatrix y,
                       typename TTypes<T>::ConstMatrix y_grad,
                       typename TTypes<T>::Matrix x_grad) {
    // y:[B, D], y_grad: [B, D], x_grad: [B, D]
    int B = y.dimension(0), D = y.dimension(1);
    CHECK(B == y_grad.dimension(0) && B == x_grad.dimension(0));
    CHECK(D == y_grad.dimension(1) && D == x_grad.dimension(1));

    // Using alternative formula:
    // dL/dx = dL/dy * y - sum(dL/dy * y) * y
    //    = (dL/dy - sum(dL/dy * y)) * y
    int batch_size = y.dimension(0), num_class = y.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};
    Eigen::array<Eigen::Index, 1> along_class = {1};
    auto dyy = y_grad * y;
    auto sum = dyy.sum(along_class).reshape(batch_by_one);
    auto sub = y_grad - sum.broadcast(one_by_class);
    x_grad.device(g_device) += sub * y;

    /*
    // dy_j / dx_i = S_i(1 - S_j) for i==j
    //             = -S_j*S_i     for i!=j
    // dL/dx_i = \sum_j dL/dy_j * dy_j / dx_i
    auto fn = [D, &x_grad, &y_grad, &y](int begin, int end) {
      for (int b = begin; b < end; ++b) {
        float* x_grad_b = x_grad.data() + b * D;
        float* y_grad_b = y_grad.data() + b * D;
        float* y_b = y.data() + b * D;
        for (int i = 0; i < D; ++i) {
          for (int j = 0; j < D; ++j) {
            float indicator = i == j ? 1.0f : 0.0f;
            //            x_grad(b, i) += y_grad(b, j) * y(b, i) * (indicator -
            //            y(b, j));
            x_grad_b[i] += y_grad_b[j] * y_b[i] * (indicator - y_b[j]);
          }
        }
      }
    };

    int thread_num = g_cpu_device.numThreads();
    Eigen::Barrier barrier(thread_num);
    for (int t = 0; t < thread_num; ++t) {
      auto range = SplitRange(B, t, thread_num);
      g_cpu_device.enqueue_with_barrier(&barrier, fn, range.first,
                                        range.second);
    }
    barrier.Wait();
    */
  }
};

//From what I could see, this only uses the Eigen library ~NR
struct SoftmaxCrossEntropy {
  using T = floatX;
  enum Reduction { MEAN, SUM };

  SoftmaxCrossEntropy(Reduction reduction = Reduction::MEAN)
      : reduction_(reduction) {}

  void Forward(typename TTypes<T>::ConstMatrix logits,
               absl::Span<const int> targets, typename TTypes<T>::Matrix probs,
               float* loss) {
    // logits: [B, C], targets: [B,], probs:[B, C], loss: scalar
    int B = logits.dimension(0), C = logits.dimension(1);
    CHECK(B == targets.size() && B == probs.dimension(0));
    CHECK_EQ(C, probs.dimension(1));

    // apply softmax to convert logits to (normalized) probabilities
    Softmax::Forward(logits, probs);

    // targets: [B,]
    *loss = 0.0f;
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
                typename TTypes<T>::Matrix logits_grad) {
    // probs: [B, C], targets: [B,]
    // logits_grad: [B, C]
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK(B == targets.size() && B == logits_grad.dimension(0));
    CHECK_EQ(C, logits_grad.dimension(1));

    float factor =
        reduction_ == Reduction::MEAN ? 1.0f / static_cast<float>(B) : 1.0f;

    for (int b = 0; b < B; ++b) {
      int ix = targets[b];
      for (int c = 0; c < C; ++c) {
        float indicator = c == ix ? 1.0f : 0.0f;
        logits_grad(b, c) += (probs(b, c) - indicator) * factor;
      }
    }
  }

  static void ForwardAndBackward(typename TTypes<T>::ConstMatrix logits,
                                 typename TTypes<T>::ConstMatrix labels,
                                 typename TTypes<T>::Flat scratch,
                                 typename TTypes<T>::Flat loss,
                                 typename TTypes<T>::Matrix logit_grad) {
    // logits: [B, C], targets: [B,], probs:[B, C], loss: scalar
    int B = logits.dimension(0), C = logits.dimension(1);
    CHECK(B == labels.dimension(0) && C == labels.dimension(1));
    CHECK(B == logit_grad.dimension(0) && C == logit_grad.dimension(1));
    CHECK_EQ(B, scratch.size());
    CHECK_EQ(B, loss.size());

    const int batch_size = B, num_class = C;
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    // max_logits along classes.
    scratch.device(g_device) = logits.maximum(along_class);

    // logits - max_logits.
    logit_grad.device(g_device) =
        logits - scratch.reshape(batch_by_one).broadcast(one_by_class);

    // sum(exp(logits - max_logits)) along classes.
    scratch.device(g_device) = logit_grad.exp().sum(along_class);

    // NOTE: Eigen on GPU dispatches to an optimized implementation
    // for an expression of the form lhs = rhs.sum().
    // lhs = -rhs.sum() doesn't match the above pattern, so folding in the
    // negation before calling sum().
    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    loss.device(g_device) =
        (labels * (scratch.log().reshape(batch_by_one).broadcast(one_by_class) -
                   logit_grad))
            .sum(along_class);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    logit_grad.device(g_device) =
        (logit_grad.exp() /
         scratch.reshape(batch_by_one).broadcast(one_by_class)) -
        labels;
  }

  Reduction reduction_;
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_