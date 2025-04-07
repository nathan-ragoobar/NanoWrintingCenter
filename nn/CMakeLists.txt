cmake_minimum_required(VERSION 3.16)

# Make sure CUDA is enabled for this subdirectory
#enable_language(CUDA)

add_library(nn INTERFACE)

set(NN_DEPENDENCIES
    absl::strings 
    absl::log 
    absl::check
)

if(UNIX)
    list(APPEND NN_DEPENDENCIES pthread)
endif()

target_link_libraries(nn INTERFACE ${NN_DEPENDENCIES})

add_library(gpt INTERFACE)
target_link_libraries(gpt INTERFACE nn)


#[[

cmake_minimum_required(VERSION 3.16)

# Make sure CUDA is enabled for this subdirectory
enable_language(CUDA)

# Add libraries
add_library(nn STATIC nn.cpp nn.hpp)  # Change to actual source file
target_link_libraries(nn
    absl::strings absl::log absl::check pthread)

add_library(gpt STATIC gpt.cpp gpt.hpp)  # Change to actual source file
target_link_libraries(gpt nn)

# Make sure headers are visible
target_include_directories(nn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

]]
