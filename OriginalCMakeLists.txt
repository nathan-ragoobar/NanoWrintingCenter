cmake_minimum_required(VERSION 3.16)
project(nano LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add MSVC-specific compiler options
if(MSVC)
    add_compile_options(/bigobj)
endif()

#find_package(Threads REQUIRED)

# Add the Abseil library
add_subdirectory(abseil-cpp)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/abseil-cpp
    ${CMAKE_SOURCE_DIR}
)

include_directories("/llmc")
include_directories("/eigen")
include_directories("/tensor")

include_directories(.)

# Eigen
set(EIGEN3_INCLUDE_DIR /eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})


# Source files

# Add libraries and executables
add_subdirectory(nn)

add_subdirectory(optimizer)

add_library(nano nano.hpp)
target_link_libraries(nano nn)

#add_library(gpt gpt.hpp)
#target_link_libraries(gpt nn)

# GPT2 library
add_library(gpt2 INTERFACE)
target_link_libraries(gpt2 INTERFACE gpt)

#add_library(optim optim.hpp)
#target_link_libraries(optim nn)



add_executable(train_gpt2_cpu train_gpt2.cpp)
target_link_libraries(train_gpt2_cpu gpt2 optim)
#target_compile_options(train_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(inference_gpt2_cpu inference_gpt2.cpp)
target_link_libraries(inference_gpt2_cpu gpt2 optim)
#target_compile_options(inference_gpt2_cpu PRIVATE -Ofast -march=native)


# Platform-specific settings
if (WIN32)
    target_compile_definitions(train_gpt2_cpu PRIVATE _CRT_SECURE_NO_WARNINGS)
    #target_compile_options(train_gpt2_cpu PRIVATE 
    #    /O2     # Maximum optimization
    #    /arch:AVX2  # Use Advanced Vector Extensions 2
    #    /GL     # Whole program optimization
    #)
    target_link_libraries(train_gpt2_cpu ws2_32)
else()
    target_compile_options(train_gpt2_cpu PRIVATE 
        -Ofast 
        -march=native 
        -g3             # Maximum debug info
        -fno-omit-frame-pointer  # Better stack traces
        )
    target_compile_options(inference_gpt2_cpu PRIVATE 
        -Ofast 
        -march=native 
        -g3
        -fno-omit-frame-pointer
        )
endif()



option(USE_CUDA "Enable CUDA for GPU acceleration" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
# Add CUDA support
# GPU specific configuration
    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        
        # Add CUDA executable
        add_executable(train_gpt2_gpu train_gpt2.cu)

        # Add GPU definition only for the GPU target
        target_compile_definitions(train_gpt2_gpu PRIVATE -DEIGEN_USE_GPU)
        
        target_link_libraries(train_gpt2_gpu 
            PRIVATE 
            gpt2 
            optim 
            ${CUDA_LIBRARIES}
        )
        set_target_properties(train_gpt2_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(train_gpt2_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-Ofast -march=native>)
    endif()
endif()

