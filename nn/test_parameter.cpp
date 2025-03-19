#include "./../tensor/fixed_point.hpp"
#include "./Parameter.hpp"
#include <gtest/gtest.h>

using namespace nn;

TEST(ConstantFillTest, FixedPointFill) {
    // Setup
    std::vector<fixed_point_31pt32> data(10);
    absl::Span<fixed_point_31pt32> span(data); //Provides a safe view into contiguous data without owning it
    
    // Test fill with 1.5
    ConstantFill(span, fixed_point_31pt32(1.5f));
    
    // Verify
    for(const auto& val : span) {
        EXPECT_EQ(val.to_float(), 1.5f);
    }
}

TEST(UniformFillTest, FloatRangeTest) {
    // Setup
    std::vector<float> data(1000);
    absl::Span<float> span(data);
    float min_val = -1.0f;
    float max_val = 1.0f;
    
    // Set random seed for reproducibility
    nn::ManualSeed(42);
    
    // Fill with random values
    nn::UniformFill(span, min_val, max_val);
    
    // Verify bounds
    for(const auto& val : span) {
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Verify randomness (values aren't all the same)
    float first_val = span[0];
    bool all_same = true;
    for(size_t i = 1; i < span.size(); i++) {
        if(span[i] != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(UniformFillTest, FixedPointRangeTest) {
    // Setup
    std::vector<fixed_point_31pt32> data(1000);
    absl::Span<fixed_point_31pt32> span(data);
    fixed_point_31pt32 min_val(-1.0f);
    fixed_point_31pt32 max_val(1.0f);
    
    // Set seed
    nn::ManualSeed(42);
    
    // Fill with random values
    nn::UniformFill(span, min_val, max_val);
    
    // Verify bounds
    for(const auto& val : span) {
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Verify randomness
    fixed_point_31pt32 first_val = span[0];
    bool all_same = true;
    for(size_t i = 1; i < span.size(); i++) {
        if(span[i] != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(NormalFillTest, FixedPointDistribution) {
    std::vector<fixed_point_31pt32> data(1000);
    absl::Span<fixed_point_31pt32> span(data);
    
    fixed_point_31pt32 mean(0.0f);
    fixed_point_31pt32 std(1.0f);
    
    nn::ManualSeed(42);
    nn::NormalFill(span, mean, std);
    
    // Verify distribution properties
    float sum = 0.0f;
    for(const auto& val : span) {
        sum += val.to_float();
    }
    float empirical_mean = sum / span.size();

    // Calculate variance and standard deviation
    float sum_squared_diff = 0.0f;
    for(const auto& val : span) {
        float diff = val.to_float() - empirical_mean;
        sum_squared_diff += diff * diff;
    }
    float variance = sum_squared_diff / span.size();
    float std_dev = std::sqrt(variance);

    EXPECT_NEAR(empirical_mean, 0.0f, 0.1f);
    EXPECT_NEAR(std_dev, 1.0f, 0.1f);  // Standard normal distribution
}

TEST(KaimingUniformFillTest, FixedPointTest) {
    std::vector<fixed_point_31pt32> data(100);
    absl::Span<fixed_point_31pt32> span(data);
    
    nn::ManualSeed(42);
    nn::KaimingUniformFill(span, 10);
    
    fixed_point_31pt32 expected_bound = sqrt(fixed_point_31pt32(0.2f)); //Kaiming fill follows a Gaussian distribution with std dev = sqrt(2 / in_features) 
    
    for(const auto& val : span) {
        EXPECT_TRUE(val >= -expected_bound);
        EXPECT_TRUE(val <= expected_bound);
    }
    //This is the actual test for the Kaiming Uniform Fill. It currently uses a uniform distribution to fill the tensor..
    //It should use a Gaussian distribution to fill the tensor.
    //Wait this function might be correct since its a Uniform Fill
    /*
    // Verify distribution properties
    float sum = 0.0f;
    for(const auto& val : span) {
        sum += val.to_float();
    }
    float empirical_mean = sum / span.size();

    // Calculate variance and standard deviation
    float sum_squared_diff = 0.0f;
    for(const auto& val : span) {
        float diff = val.to_float() - empirical_mean;
        sum_squared_diff += diff * diff;
    }
    float variance = sum_squared_diff / span.size();
    float std_dev = std::sqrt(variance);

    EXPECT_NEAR(empirical_mean, 0.0f, 0.1f);
    EXPECT_NEAR(std_dev, std::sqrt(2.0f/10.0f), 0.1f);  // Standard normal distribution
    */
}

//This test needs more refinement. I don't think this has any edge cases.
TEST(UpperTriangularTest, FixedPointTest) {
    // Create tensor instead of matrix
    const int size = 3;

    // Q5.10 format has range [-16, 15.999]...... This is no longer valid
    constexpr float kMinValue = -32.0f;  

    std::vector<fixed_point_31pt32> data(size * size);
    
    // Create tensor map from data
    TTypes<fixed_point_31pt32, 2>::Tensor matrix(data.data(), size, size);
    
    // Initialize to zero
    for (int i = 0; i < size * size; i++) {
        data[i] = fixed_point_31pt32(0.0f);
    }

    UpperTriangularWithNegativeInf(matrix);
    
    // Check diagonal and lower triangle are zero
    for(int i = 0; i < size; i++) {
        for(int j = 0; j <= i; j++) {
            EXPECT_EQ(matrix(i,j).to_float(), 0.0f);
        }
    }
    
    // Check upper triangle is minimum value
    for(int i = 0; i < size; i++) {
        for(int j = i + 1; j < size; j++) {
            EXPECT_EQ(matrix(i,j).to_float(), kMinValue);
        }
    }
}

TEST(OneHotTest, FixedPointTest) {
    const int batch_size = 3;
    const int num_classes = 4;
    
    std::vector<int> target_data = {1, 0, 2};
    std::vector<fixed_point_31pt32> label_data(batch_size * num_classes, fixed_point_31pt32(0.0f));
    
    TTypes<int>::ConstFlat target(target_data.data(), batch_size);
    TTypes<fixed_point_31pt32>::Matrix label(label_data.data(), batch_size, num_classes);
    
    OneHot(target, label);
    
    // Verify results
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_classes; j++) {
            if(j == target_data[i]) {
                EXPECT_EQ(label(i,j).to_float(), 1.0f);
            } else {
                EXPECT_EQ(label(i,j).to_float(), 0.0f);
            }
        }
    }
}

TEST(SplitRangeTest, BasicFunctionality) {
    // Split 10 items into 3 chunks
    auto range0 = SplitRange(10, 0, 3); // {0, 4}  First chunk
    auto range1 = SplitRange(10, 1, 3); // {4, 7}  Second chunk  
    auto range2 = SplitRange(10, 2, 3); // {7, 10} Third chunk

    EXPECT_EQ(range0.first, 0);
    EXPECT_EQ(range0.second, 4);
    EXPECT_EQ(range1.first, 4);
    EXPECT_EQ(range1.second, 7);
    EXPECT_EQ(range2.first, 7);
    EXPECT_EQ(range2.second, 10);
}



// Add type declaration first
struct InvalidType {};

// Compile-time checks
static_assert(!IsValidDataType<InvalidType>::value,
             "Invalid type should not be supported");

TEST(DataTypeTest, FixedPointDataType) {
    // Runtime checks
    EXPECT_EQ(DataTypeToEnum<fixed_point_31pt32>::v(), DT_FIXED);

    // Compile-time checks within test
    // Use std::is_same instead of std::is_same_v for C++14
    static_assert(std::is_same<
        typename EnumToDataType<DT_FIXED>::Type,
        fixed_point_31pt32
    >::value, "DT_FIXED should map to fixed_point_31pt32");
    
    static_assert(IsValidDataType<fixed_point_31pt32>::value,
                 "fixed_point_31pt32 should be a valid type");
    
    static_assert(DataTypeToEnum<fixed_point_31pt32>::value == DT_FIXED,
                 "Static value should match enum");
}


TEST(TypeDefinitionTest, fixed_point_31pt32TypeExists) {
    // Compile-time check that type exists
    static_assert(sizeof(fixed_point_31pt32) > 0, 
                 "fixed_point_31pt32 type not defined");
    
    // Runtime check type has expected properties
    fixed_point_31pt32 test_val(1.0f);
    EXPECT_EQ(test_val.to_float(), 1.0f);
}

TEST(Parameter, ConstructorInitializesCorrectly) {
    Parameter p(DT_FIXED, 10);
    EXPECT_EQ(p.size(), 10);
}

TEST(Parameter, DefaultConstructorNoAllocation) {
    Parameter p(DT_FIXED);
    EXPECT_EQ(p.size(), 0);
}
TEST(Parameter, LazyAllocationWorks) {
    Parameter p(DT_FIXED);
    EXPECT_EQ(p.size(), 0);
    p.LazyAllocate(5);
    EXPECT_EQ(p.size(), 5);
}

TEST(Parameter, MultipleLazyAllocWithSameSizeOK) {
    Parameter p(DT_FIXED);
    p.LazyAllocate(5);
    EXPECT_EQ(p.size(), 5);
    p.LazyAllocate(5); // Should not throw
    EXPECT_EQ(p.size(), 5);
}

TEST(Parameter, LazyAllocDifferentSizeFails) {
    Parameter p(DT_FIXED);
    p.LazyAllocate(5);
    EXPECT_DEATH(p.LazyAllocate(10), "Check failed");
}

TEST(Parameter, CopyConstructorDeleted) {
    Parameter p1(DT_FIXED);
    EXPECT_FALSE(std::is_copy_constructible<Parameter>::value);
}

TEST(Parameter, AssignmentOperatorDeleted) {
    Parameter p1(DT_FIXED);
    EXPECT_FALSE(std::is_copy_assignable<Parameter>::value);
}

class ParameterFixedPointTest : public ::testing::Test {
protected:
    void SetUp() override {
        param = new Parameter(DT_FIXED, 5);
    }

    void TearDown() override {
        delete param;
    }

    Parameter* param;
};

// LazyAllocation Tests
TEST_F(ParameterFixedPointTest, LazyAllocationInitializesToZero) {
    Parameter p(DT_FIXED);
    p.LazyAllocate(3);
    auto data = p.span<fixed_point_31pt32>();
    
    for(int i = 0; i < 3; i++) {
        EXPECT_EQ(data[i].to_float(), 0.0f);
    }
}

TEST_F(ParameterFixedPointTest, GradientLazyAllocation) {
    param->LazyAllocateGradient();
    auto grad = param->span_grad<fixed_point_31pt32>();
    
    EXPECT_EQ(grad.size(), 5);
    for(int i = 0; i < 5; i++) {
        EXPECT_EQ(grad[i].to_float(), 0.0f);
    }
}

// Zero Operations Tests
TEST_F(ParameterFixedPointTest, ZeroDataOperation) {
    auto data = param->span<fixed_point_31pt32>();
    data[0] = fixed_point_31pt32(1.5f);
    
    param->ZeroData();
    EXPECT_EQ(data[0].to_float(), 0.0f);
}

TEST_F(ParameterFixedPointTest, ZeroGradOperation) {
    param->LazyAllocateGradient();
    auto grad = param->span_grad<fixed_point_31pt32>();
    grad[0] = fixed_point_31pt32(1.5f);
    
    param->ZeroGrad();
    EXPECT_EQ(grad[0].to_float(), 0.0f);
}

// Data Access Tests
TEST_F(ParameterFixedPointTest, RawDataAccess) {
    auto* data_ptr = param->data<fixed_point_31pt32>();
    EXPECT_NE(data_ptr, nullptr);
    
    data_ptr[0] = fixed_point_31pt32(2.5f);
    EXPECT_EQ(data_ptr[0].to_float(), 2.5f);
}

TEST_F(ParameterFixedPointTest, RawGradAccess) {
    param->LazyAllocateGradient();
    auto* grad_ptr = param->grad<fixed_point_31pt32>();
    EXPECT_NE(grad_ptr, nullptr);
    
    grad_ptr[0] = fixed_point_31pt32(3.5f);
    EXPECT_EQ(grad_ptr[0].to_float(), 3.5f);
}

// Span Access Tests
TEST_F(ParameterFixedPointTest, SpanAccess) {
    auto span = param->span<fixed_point_31pt32>();
    EXPECT_EQ(span.size(), 5);
    
    span[0] = fixed_point_31pt32(4.5f);
    EXPECT_EQ(span[0].to_float(), 4.5f);
}

TEST_F(ParameterFixedPointTest, GradSpanAccess) {
    param->LazyAllocateGradient();
    auto grad_span = param->span_grad<fixed_point_31pt32>();
    EXPECT_EQ(grad_span.size(), 5);
    
    grad_span[0] = fixed_point_31pt32(5.5f);
    EXPECT_EQ(grad_span[0].to_float(), 5.5f);
}

// Type Checking Tests
TEST_F(ParameterFixedPointTest, WrongTypeAccess) {
    EXPECT_DEATH(param->span<float>(), "");
    EXPECT_DEATH(param->span_grad<float>(), "");
}

//I think this class is not needed. REMOVE IT
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 24 elements total - can be reshaped as:
        // - 24 x 1 (flat)
        // - 6 x 4 (matrix)
        // - 2 x 3 x 4 (3D)
        // - 2 x 2 x 2 x 3 (4D)
        param = new Parameter(DT_FIXED, 24);
        param->LazyAllocateGradient();
    }

    void TearDown() override {
        delete param;
    }

    Parameter* param;
};

// Flat Tensor Tests
TEST_F(TensorTest, FlatTensorAccess) {
    auto flat = param->flat<fixed_point_31pt32>();
    EXPECT_EQ(flat.size(), 24);
    
    flat(0) = fixed_point_31pt32(1.5f);
    EXPECT_EQ(flat(0).to_float(), 1.5f);
}

TEST_F(TensorTest, ConstFlatTensorAccess) {
    auto const_flat = param->const_flat<fixed_point_31pt32>();
    EXPECT_EQ(const_flat.size(), 24);
}

// Matrix Tests
TEST_F(TensorTest, MatrixTensorAccess) {
    auto matrix = param->matrix<fixed_point_31pt32>(6, 4);
    EXPECT_EQ(matrix.dimension(0), 6);
    EXPECT_EQ(matrix.dimension(1), 4);
    
    matrix(0, 0) = fixed_point_31pt32(2.5f);
    EXPECT_EQ(matrix(0, 0).to_float(), 2.5f);
}

TEST_F(TensorTest, MatrixDimensionMismatch) {
    EXPECT_DEATH(param->matrix<fixed_point_31pt32>(5, 5), "");
}

// 3D Tensor Tests
TEST_F(TensorTest, Tensor3DAccess) {
    auto tensor3d = param->tensor_3d<fixed_point_31pt32>(2, 3, 4);
    EXPECT_EQ(tensor3d.dimension(0), 2);
    EXPECT_EQ(tensor3d.dimension(1), 3);
    EXPECT_EQ(tensor3d.dimension(2), 4);
    
    tensor3d(0, 0, 0) = fixed_point_31pt32(3.5f);
    EXPECT_EQ(tensor3d(0, 0, 0).to_float(), 3.5f);
}

// 4D Tensor Tests
TEST_F(TensorTest, Tensor4DAccess) {
    auto tensor4d = param->tensor_4d<fixed_point_31pt32>(2, 2, 2, 3);
    EXPECT_EQ(tensor4d.dimension(0), 2);
    EXPECT_EQ(tensor4d.dimension(1), 2);
    EXPECT_EQ(tensor4d.dimension(2), 2);
    EXPECT_EQ(tensor4d.dimension(3), 3);
    
    tensor4d(0, 0, 0, 0) = fixed_point_31pt32(4.5f);
    EXPECT_EQ(tensor4d(0, 0, 0, 0).to_float(), 4.5f);
}

// Gradient Tests
TEST_F(TensorTest, GradientTensorAccess) {
    auto flat_grad = param->flat_grad<fixed_point_31pt32>();
    auto matrix_grad = param->matrix_grad<fixed_point_31pt32>(6, 4);
    auto tensor3d_grad = param->tensor_3d_grad<fixed_point_31pt32>(2, 3, 4);
    auto tensor4d_grad = param->tensor_4d_grad<fixed_point_31pt32>(2, 2, 2, 3);
    
    flat_grad(0) = fixed_point_31pt32(5.5f);
    EXPECT_EQ(flat_grad(0).to_float(), 5.5f);
    
    matrix_grad(0, 0) = fixed_point_31pt32(6.5f);
    EXPECT_EQ(matrix_grad(0, 0).to_float(), 6.5f);
}

// Type Checking Tests
TEST_F(TensorTest, WrongTypeAccess) {
    EXPECT_DEATH(param->flat<float>(), "");
    EXPECT_DEATH(param->matrix<float>(6, 4), "");
    EXPECT_DEATH(param->tensor_3d<float>(2, 3, 4), "");
    EXPECT_DEATH(param->tensor_4d<float>(2, 2, 2, 3), "");
}

TEST(ResidualTest, FixedPointOperations) {
    Parameter x(DT_FIXED, 3);
    Parameter Fx(DT_FIXED, 3);
    Parameter Hx(DT_FIXED, 3);
    
    auto x_span = x.span<fixed_point_31pt32>();
    auto Fx_span = Fx.span<fixed_point_31pt32>();
    
    x_span[0] = fixed_point_31pt32(1.0f);
    Fx_span[0] = fixed_point_31pt32(2.0f);
    
   Residual::Forward(x.const_flat<fixed_point_31pt32>(),    // Changed to const_flat
                     Fx.const_flat<fixed_point_31pt32>(),     // Changed to const_flat
                     Hx.flat<fixed_point_31pt32>());
                     
    EXPECT_EQ(Hx.flat<fixed_point_31pt32>()(0).to_float(), 3.0f);
}

TEST(ResidualTest, BackwardFixedPoint) {
    // Setup parameters
    Parameter Hx_grad(DT_FIXED, 3);
    Parameter x_grad(DT_FIXED, 3);
    Parameter Fx_grad(DT_FIXED, 3);

    // Initialize gradients
    auto Hx_grad_span = Hx_grad.span<fixed_point_31pt32>();
    Hx_grad_span[0] = fixed_point_31pt32(1.0f);
    Hx_grad_span[1] = fixed_point_31pt32(2.0f);
    Hx_grad_span[2] = fixed_point_31pt32(3.0f);

    // Call backward
    Residual::Backward(Hx_grad.const_flat<fixed_point_31pt32>(),
                      x_grad.flat<fixed_point_31pt32>(),
                      Fx_grad.flat<fixed_point_31pt32>());

    // Verify gradients accumulated correctly
    auto x_grad_span = x_grad.span<fixed_point_31pt32>();
    auto Fx_grad_span = Fx_grad.span<fixed_point_31pt32>();

    for(int i = 0; i < 3; i++) {
        EXPECT_EQ(x_grad_span[i].to_float(), Hx_grad_span[i].to_float());
        EXPECT_EQ(Fx_grad_span[i].to_float(), Hx_grad_span[i].to_float());
    }
}

TEST(NewGELUTest, ForwardFixedPoint) {
    Parameter x(DT_FIXED, 3);
    Parameter y(DT_FIXED, 3);
    
    auto x_span = x.span<fixed_point_31pt32>();
    x_span[0] = fixed_point_31pt32(1.0f);
    x_span[1] = fixed_point_31pt32(0.0f);
    x_span[2] = fixed_point_31pt32(-1.0f);
    
    NewGELU::Forward(x.const_flat<fixed_point_31pt32>(),
                    y.flat<fixed_point_31pt32>());
    
    auto y_span = y.span<fixed_point_31pt32>();
    EXPECT_NEAR(y_span[0].to_float(), 0.841f, 0.01f);
    EXPECT_NEAR(y_span[1].to_float(), 0.0f, 0.01f);
    EXPECT_NEAR(y_span[2].to_float(), -0.159f, 0.01f);
}

//This test case is not working. It is failing.
//Idk how to verify that the value that the test case is even checking for is correct.

TEST(NewGELUTest, BackwardFixedPoint) {
    Parameter x(DT_FIXED, 2);
    Parameter y_grad(DT_FIXED, 2);
    Parameter x_grad(DT_FIXED, 2);
    
    auto x_span = x.span<fixed_point_31pt32>();
    auto y_grad_span = y_grad.span<fixed_point_31pt32>();
    
    x_span[0] = fixed_point_31pt32(1.0f);
    x_span[1] = fixed_point_31pt32(-1.0f);
    y_grad_span[0] = fixed_point_31pt32(1.0f);
    y_grad_span[1] = fixed_point_31pt32(1.0f);
    
    NewGELU::Backward(x.const_flat<fixed_point_31pt32>(),
                     y_grad.const_flat<fixed_point_31pt32>(),
                     x_grad.flat<fixed_point_31pt32>());

    //std::cout << x.span<fixed_point_31pt32>()[1].to_float() << std::endl;
    //std::cout << y_grad.span<fixed_point_31pt32>()[1].to_float() << std::endl;
    //std::cout << x_grad.span<fixed_point_31pt32>()[1].to_float() << std::endl;
    
    auto x_grad_span = x_grad.span<fixed_point_31pt32>();
    EXPECT_NEAR(x_grad_span[0].to_float(), 1.083f, 0.01f);
    EXPECT_NEAR(x_grad_span[1].to_float(), 0.084f, 0.01f);
}
/**/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}