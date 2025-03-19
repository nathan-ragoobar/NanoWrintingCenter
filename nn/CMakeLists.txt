cmake_minimum_required(VERSION 3.16)

add_library(nn INTERFACE)

# Base dependencies for all platforms
set(NN_DEPENDENCIES
    absl::strings 
    absl::log 
    absl::check
)

# Add pthread only for Unix-like systems
if(UNIX)
    list(APPEND NN_DEPENDENCIES pthread)
endif()

target_link_libraries(nn INTERFACE ${NN_DEPENDENCIES})

add_library(gpt INTERFACE)
target_link_libraries(gpt INTERFACE nn)