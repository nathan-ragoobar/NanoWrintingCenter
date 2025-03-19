// Add missing PROFILE_TRACE_FN if not GPU
#ifndef PROFILE_TRACE_FN
#define PROFILE_TRACE_FN(name)
#endif

#include <gtest/gtest.h>
#include "./activation.hpp"
#include "./Parameter.hpp"
#include <vector>
#include <cmath>


using namespace nn;  // Add namespace

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


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}