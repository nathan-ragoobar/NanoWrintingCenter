// Add missing PROFILE_TRACE_FN if not GPU
#ifndef PROFILE_TRACE_FN
#define PROFILE_TRACE_FN(name)
#endif

#include <gtest/gtest.h>
#include "activation.hpp"
#include "./../tensor/tensor_types.hpp"
#include "./Parameter.hpp"
#include <vector>
#include <cmath>
#include <limits>

using namespace nn;

// Helper function to create tensors for testing
template <typename T>
std::pair<typename TTypes<T>::ConstFlat, std::vector<T>> MakeTestTensor(const std::vector<T>& data) {
  std::vector<T> tensor_data = data;
  auto flat = MakeConstFlat(tensor_data.data(), tensor_data.size());
  return {flat, tensor_data};
}

template <typename T>
std::pair<typename TTypes<T>::Flat, std::vector<T>> MakeOutputTensor(size_t size) {
  std::vector<T> tensor_data(size, 0);
  auto flat = MakeFlat(tensor_data.data(), tensor_data.size());
  return {flat, tensor_data};
}

class SigmoidTest : public ::testing::Test {
  protected:
      void SetUp() override {
          // No setup needed
      }
  };
  
  TEST_F(SigmoidTest, ForwardPassFloat) {
      // Setup parameters
      Parameter x(DT_FLOAT, 4);
      Parameter y(DT_FLOAT, 4);
      
      // Set input values
      auto x_span = x.span<float>();
      x_span[0] = 0.0f;    // sigmoid(0) = 0.5
      x_span[1] = 1.0f;    // sigmoid(1) ≈ 0.731
      x_span[2] = -1.0f;   // sigmoid(-1) ≈ 0.269
      x_span[3] = 10.0f;   // sigmoid(10) ≈ 1.0
      
      // Forward pass
      nn::Sigmoid::Forward<float>(x.const_flat<float>(),
                          y.flat<float>());
      
      // Check results
      auto y_span = y.span<float>();
      EXPECT_NEAR(y_span[0], 0.5f, 1e-6);
      EXPECT_NEAR(y_span[1], 0.731058f, 1e-6);
      EXPECT_NEAR(y_span[2], 0.268941f, 1e-6);
      EXPECT_NEAR(y_span[3], 0.999954f, 1e-6);
  }
  
  TEST_F(SigmoidTest, BackwardPassFloat) {
      // Setup parameters
      Parameter y(DT_FLOAT, 4);
      Parameter grad_y(DT_FLOAT, 4);
      Parameter grad_x(DT_FLOAT, 4);
      
      // Set activation values
      auto y_span = y.span<float>();
      y_span[0] = 0.5f;      // sigmoid'(0) = 0.25
      y_span[1] = 0.731058f; // sigmoid'(1) ≈ 0.197
      y_span[2] = 0.268941f; // sigmoid'(-1) ≈ 0.197
      y_span[3] = 0.999954f; // sigmoid'(10) ≈ 0
      
      // Set gradient values
      auto grad_y_span = grad_y.span<float>();
      for (int i = 0; i < 4; i++) {
          grad_y_span[i] = 1.0f;
      }
      
      // Backward pass
      nn::Sigmoid::Backward<float>(y.const_flat<float>(),
                           grad_y.const_flat<float>(),
                           grad_x.flat<float>());
      
      // Check results
      auto grad_x_span = grad_x.span<float>();
      EXPECT_NEAR(grad_x_span[0], 0.25f, 1e-6);
      EXPECT_NEAR(grad_x_span[1], 0.196612f, 1e-6);
      EXPECT_NEAR(grad_x_span[2], 0.196612f, 1e-6);
      EXPECT_NEAR(grad_x_span[3], 4.54e-5f, 1e-6);
  }
  class ReLUTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // No setup needed
        }
    };
    
    TEST_F(ReLUTest, ForwardPassFloat) {
        // Setup parameters
        Parameter x(DT_FLOAT, 5);
        Parameter y(DT_FLOAT, 5);
        
        // Set input values
        auto x_span = x.span<float>();
        x_span[0] = -2.0f;   // relu(-2) = 0
        x_span[1] = -1.0f;   // relu(-1) = 0
        x_span[2] = 0.0f;    // relu(0) = 0
        x_span[3] = 1.0f;    // relu(1) = 1
        x_span[4] = 2.0f;    // relu(2) = 2
        
        // Forward pass
        nn::ReLU::Forward<float>(x.const_flat<float>(),
                          y.flat<float>());
        
        // Check results
        auto y_span = y.span<float>();
        EXPECT_FLOAT_EQ(y_span[0], 0.0f);
        EXPECT_FLOAT_EQ(y_span[1], 0.0f);
        EXPECT_FLOAT_EQ(y_span[2], 0.0f);
        EXPECT_FLOAT_EQ(y_span[3], 1.0f);
        EXPECT_FLOAT_EQ(y_span[4], 2.0f);
    }
    
    TEST_F(ReLUTest, ForwardPassBorderline) {
        // Setup parameters for borderline cases
        Parameter x(DT_FLOAT, 3);
        Parameter y(DT_FLOAT, 3);
        
        // Set input values with epsilon for numerical precision testing
        const float epsilon = std::numeric_limits<float>::epsilon();
        auto x_span = x.span<float>();
        x_span[0] = -epsilon;  // Should be 0
        x_span[1] = 0.0f;      // Should be 0
        x_span[2] = epsilon;   // Should be epsilon
        
        // Forward pass
        nn::ReLU::Forward<float>(x.const_flat<float>(),
                          y.flat<float>());
        
        // Check results
        auto y_span = y.span<float>();
        EXPECT_FLOAT_EQ(y_span[0], 0.0f);
        EXPECT_FLOAT_EQ(y_span[1], 0.0f);
        EXPECT_FLOAT_EQ(y_span[2], epsilon);
    }
    
    TEST_F(ReLUTest, BackwardPassFloat) {
        // Setup parameters
        Parameter x(DT_FLOAT, 5);
        Parameter grad_y(DT_FLOAT, 5);
        Parameter grad_x(DT_FLOAT, 5);
        
        // Set input values
        auto x_span = x.span<float>();
        x_span[0] = -2.0f;   // relu'(-2) = 0
        x_span[1] = -1.0f;   // relu'(-1) = 0
        x_span[2] = 0.0f;    // relu'(0) = 0 (by convention)
        x_span[3] = 1.0f;    // relu'(1) = 1
        x_span[4] = 2.0f;    // relu'(2) = 1
        
        // Set gradient values
        auto grad_y_span = grad_y.span<float>();
        for (int i = 0; i < 5; i++) {
            grad_y_span[i] = 1.0f;
        }
        
        // Backward pass
        nn::ReLU::Backward<float>(x.const_flat<float>(),
                           grad_y.const_flat<float>(),
                           grad_x.flat<float>());
        
        // Check results
        auto grad_x_span = grad_x.span<float>();
        EXPECT_FLOAT_EQ(grad_x_span[0], 0.0f);
        EXPECT_FLOAT_EQ(grad_x_span[1], 0.0f);
        EXPECT_FLOAT_EQ(grad_x_span[2], 0.0f);
        EXPECT_FLOAT_EQ(grad_x_span[3], 1.0f);
        EXPECT_FLOAT_EQ(grad_x_span[4], 1.0f);
    }
    
    TEST_F(ReLUTest, BackwardPassWithNonUnitaryGradient) {
        // Setup parameters
        Parameter x(DT_FLOAT, 3);
        Parameter grad_y(DT_FLOAT, 3);
        Parameter grad_x(DT_FLOAT, 3);
        
        // Set input values
        auto x_span = x.span<float>();
        x_span[0] = -1.0f;   // negative (gradient should be 0)
        x_span[1] = 0.0f;    // zero (gradient should be 0)
        x_span[2] = 1.0f;    // positive (should pass through)
        
        // Set gradient values - non-unitary this time
        auto grad_y_span = grad_y.span<float>();
        grad_y_span[0] = 2.0f;
        grad_y_span[1] = 3.0f;
        grad_y_span[2] = 4.0f;
        
        // Backward pass
        nn::ReLU::Backward<float>(x.const_flat<float>(),
                           grad_y.const_flat<float>(),
                           grad_x.flat<float>());
        
        // Check results
        auto grad_x_span = grad_x.span<float>();
        EXPECT_FLOAT_EQ(grad_x_span[0], 0.0f);
        EXPECT_FLOAT_EQ(grad_x_span[1], 0.0f);
        EXPECT_FLOAT_EQ(grad_x_span[2], 4.0f);
    }


  /*
  
TEST(ReLUTest, ForwardBasicValues) {
  std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  auto [input_tensor, input_vec] = MakeTestTensor<float>(input_data);
  
  auto [output_tensor, output_vec] = MakeOutputTensor<float>(input_data.size());
  
  nn::ReLU::Forward<float>(input_tensor, output_tensor);
  
  // Expected: max(0, x)
  std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
  
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(output_vec[i], expected[i]) 
        << "Mismatch at position " << i << ": expected " << expected[i] 
        << " but got " << output_vec[i];
  }
}

TEST(ReLUTest, ForwardBorderlineValues) {
  // Test with values close to zero
  const float epsilon = std::numeric_limits<float>::epsilon();
  std::vector<float> input_data = {-epsilon, 0.0f, epsilon};
  auto [input_tensor, input_vec] = MakeTestTensor<float>(input_data);
  
  auto [output_tensor, output_vec] = MakeOutputTensor<float>(input_data.size());
  
  nn::ReLU::Forward<float>(input_tensor, output_tensor);
  
  // Expected values
  EXPECT_FLOAT_EQ(output_vec[0], 0.0f);
  EXPECT_FLOAT_EQ(output_vec[1], 0.0f);
  EXPECT_FLOAT_EQ(output_vec[2], epsilon);
}

TEST(ReLUTest, BackwardBasicValues) {
  // For ReLU backward, we need:
  // 1. Original input (x)
  // 2. Gradient of loss w.r.t. ReLU output (grad_y)
  
  std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  auto [input_tensor, input_vec] = MakeTestTensor<float>(input_data);
  
  std::vector<float> output_grad = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  auto [grad_y_tensor, grad_y_vec] = MakeTestTensor<float>(output_grad);
  
  auto [grad_x_tensor, grad_x_vec] = MakeOutputTensor<float>(input_data.size());
  
  nn::ReLU::Backward<float>(input_tensor, grad_y_tensor, grad_x_tensor);
  
  // Expected: grad_x = grad_y if x > 0 else 0
  std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f};
  
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(grad_x_vec[i], expected[i]) 
        << "Mismatch at position " << i << ": expected " << expected[i] 
        << " but got " << grad_x_vec[i];
  }
}

TEST(ReLUTest, BackwardWithNonunitaryGradient) {
  // Test with gradient values other than 1.0
  std::vector<float> input_data = {-1.0f, 0.0f, 1.0f};
  auto [input_tensor, input_vec] = MakeTestTensor<float>(input_data);
  
  std::vector<float> output_grad = {2.0f, 3.0f, 4.0f};
  auto [grad_y_tensor, grad_y_vec] = MakeTestTensor<float>(output_grad);
  
  auto [grad_x_tensor, grad_x_vec] = MakeOutputTensor<float>(input_data.size());
  
  nn::ReLU::Backward<float>(input_tensor, grad_y_tensor, grad_x_tensor);
  
  // Expected: grad_x = grad_y if x > 0 else 0
  std::vector<float> expected = {0.0f, 0.0f, 4.0f};
  
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(grad_x_vec[i], expected[i]);
  }
}

TEST(ReLUTest, BackwardExactlyZero) {
  // Test the behavior exactly at x=0 (should be 0 gradient by convention)
  std::vector<float> input_data = {0.0f};
  auto [input_tensor, input_vec] = MakeTestTensor<float>(input_data);
  
  std::vector<float> output_grad = {1.0f};
  auto [grad_y_tensor, grad_y_vec] = MakeTestTensor<float>(output_grad);
  
  auto [grad_x_tensor, grad_x_vec] = MakeOutputTensor<float>(input_data.size());
  
  nn::ReLU::Backward<float>(input_tensor, grad_y_tensor, grad_x_tensor);
  
  // By convention, ReLU gradient at x=0 is 0
  EXPECT_FLOAT_EQ(grad_x_vec[0], 0.0f);
}

*/

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}