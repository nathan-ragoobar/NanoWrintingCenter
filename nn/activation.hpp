#ifndef LLM_CPP__SIGMOID_HPP_
#define LLM_CPP__SIGMOID_HPP_

#include "nn.hpp"
#include <cmath>

// Add fallback definition if not already defined
#ifndef PROFILE_TRACE_FN
#define PROFILE_TRACE_FN(name)
#endif

namespace nn {

class Sigmoid {
public:
  template <typename T>
  static void Forward(typename TTypes<T>::ConstFlat x,
                     typename TTypes<T>::Flat y) {
    PROFILE_TRACE_FN("Sigmoid");
    
    CHECK_EQ(x.size(), y.size());
    const int n = x.size();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      y(i) = T(1.0) / (T(1.0) + std::exp(-x(i)));
    }
  }

  template <typename T>
  static void Backward(typename TTypes<T>::ConstFlat y,
                      typename TTypes<T>::ConstFlat grad_y,
                      typename TTypes<T>::Flat grad_x) {
    PROFILE_TRACE_FN("Sigmoid");
    
    CHECK_EQ(y.size(), grad_y.size());
    CHECK_EQ(y.size(), grad_x.size());
    const int n = y.size();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      // derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
      grad_x(i) = grad_y(i) * y(i) * (T(1.0) - y(i));
    }
  }
};


class ReLU {
  public:
   template <typename T>
   static void Forward(typename TTypes<T>::ConstFlat input,
                      typename TTypes<T>::Flat output) {
     const int size = input.dimension(0);
     for (int i = 0; i < size; ++i) {
       output(i) = input(i) > 0 ? input(i) : 0;
     }
   }
 
   template <typename T>
   static void Backward(typename TTypes<T>::ConstFlat input,
                       typename TTypes<T>::ConstFlat output_grad,
                       typename TTypes<T>::Flat input_grad) {
     const int size = input.dimension(0);
     for (int i = 0; i < size; ++i) {
       input_grad(i) = input(i) > 0 ? output_grad(i) : 0;
     }
   }
 };

} // namespace nn

#endif // LLM_CPP__SIGMOID_HPP_