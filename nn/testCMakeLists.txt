cmake_minimum_required(VERSION 3.10)

# Project name
project(nn_tests)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(GTest REQUIRED)
enable_testing()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../abseil-cpp
    ${CMAKE_SOURCE_DIR}/../eigen
)
# Eigen
set(EIGEN3_INCLUDE_DIR /eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add Abseil as subdirectory or find package
add_subdirectory(${CMAKE_SOURCE_DIR}/../abseil-cpp ${CMAKE_BINARY_DIR}/abseil-cpp)


# Add test executable
add_executable(testsigmoid 
    test_sigmoid.cpp
)

# Link libraries
target_link_libraries(testsigmoid
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test
add_test(NAME SigmoidTest COMMAND testsigmoid)

add_executable(test_fastfeedforward
    test_fastfeedforward.cpp
)

target_link_libraries(test_fastfeedforward PRIVATE
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

add_test(NAME FastFeedforwardTest COMMAND test_fastfeedforward)

# Add test_activations executable
add_executable(test_activation
    test_activation.cpp
)

# Link libraries for test_activations
target_link_libraries(test_activation PRIVATE
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add activation tests to test suite
add_test(NAME ActivationsTest COMMAND test_activation)

add_executable(test_fastfeedforward_training
    test_fastfeedforward_training.cpp
)

target_link_libraries(test_fastfeedforward_training PRIVATE
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add MLP training test executable
add_executable(test_mlp_training
    test_MLP_training.cpp
)

# Link libraries for MLP training test
target_link_libraries(test_mlp_training PRIVATE
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)