cmake_minimum_required(VERSION 3.16)

project(NanoInference VERSION 0.1 LANGUAGES CXX)

# Print compiler info for debugging
message("Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message("Compiler Path: ${CMAKE_CXX_COMPILER}")

# Common CMake settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt-specific settings
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Add common compiler options
if(MSVC)
    add_compile_options(/bigobj)
endif()

# Point to Qt installation only on Windows
if(WIN32)
    set(CMAKE_PREFIX_PATH "C:/Qt/6.8.0/mingw_64")
endif()

# Common dependencies
find_package(Threads REQUIRED)

# Check for Abseil and add it if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/abseil-cpp")
    add_subdirectory(abseil-cpp)
else()
    # Create stub targets for Abseil
    message(STATUS "Abseil not found - creating stub targets")
    add_library(absl_strings INTERFACE)
    add_library(absl_log INTERFACE)
    add_library(absl_check INTERFACE)
    add_library(absl::strings ALIAS absl_strings)
    add_library(absl::log ALIAS absl_log)
    add_library(absl::check ALIAS absl_check)
endif()

# Include directories for ML components
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/abseil-cpp
    ${CMAKE_SOURCE_DIR}/llmc
    ${CMAKE_SOURCE_DIR}/eigen
    ${CMAKE_SOURCE_DIR}/tensor
    ${CMAKE_SOURCE_DIR}
)

# Eigen settings
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
add_definitions(-DEIGEN_USE_THREADS)

###################################################################
# PART 1: Qt GUI Application
###################################################################

# Find Qt components
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Concurrent)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Concurrent)

# Qt project sources
set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    inference_interface.cpp
    mainwindow.ui
    resources.qrc
)

# Configure Qt executable
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(NanoInference
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(NanoInference SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(NanoInference
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# Link against Qt components
target_link_libraries(NanoInference PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Concurrent
    Threads::Threads
)

# Platform-specific compiler settings for Qt app
if(WIN32)
    target_compile_definitions(NanoInference PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(MSVC)
        target_compile_options(NanoInference PRIVATE 
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/GL>
        )
    endif()
    if(MSVC)
        target_link_libraries(NanoInference PRIVATE ws2_32)
    endif()
else()
    # Linux/Mac compiler optimizations
    target_compile_options(NanoInference PRIVATE 
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
    )
endif()

# Bundle configuration
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.NanoInference)
endif()
set_target_properties(NanoInference PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

###################################################################
# PART 2: CLI Application
###################################################################

# Add subdirectories needed for ML components
if(EXISTS "${CMAKE_SOURCE_DIR}/nn")
    add_subdirectory(nn)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/optimizer")
    add_subdirectory(optimizer)
endif()

# Add core ML libraries
if(NOT TARGET nano)
    add_library(nano INTERFACE)
    if(TARGET nn)
        target_link_libraries(nano INTERFACE nn)
    endif()
endif()

if(NOT TARGET gpt2)
    add_library(gpt2 INTERFACE)
    if(TARGET gpt)
        target_link_libraries(gpt2 INTERFACE gpt)
    endif()
endif()

# Add CLI executable for QT integration
add_executable(inference_cli inference_cli.cpp)

# Enable RTTI for the CLI
if(MSVC)
    target_compile_options(inference_cli PRIVATE /GR)
else()
    target_compile_options(inference_cli PRIVATE -frtti)
endif()

# Link ML components to CLI
target_link_libraries(inference_cli PRIVATE 
    gpt2 
    nano 
    Threads::Threads
)

# Add test executable for CLI
add_executable(test_inference_cli test_inference_cli.cpp)
target_link_libraries(test_inference_cli PRIVATE Threads::Threads)

# Copy resources to test executable directory
add_custom_command(TARGET test_inference_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:inference_cli>
        $<TARGET_FILE_DIR:test_inference_cli>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/vocab.bpe"
        "${CMAKE_SOURCE_DIR}/encoder.json"
        "${CMAKE_SOURCE_DIR}/gpt2_124M100Steps.bin"
        $<TARGET_FILE_DIR:test_inference_cli>/
    COMMENT "Copying dependencies to test_inference_cli output directory..."
)

# Static linking for CLI on MinGW
if(MINGW)
    target_link_options(inference_cli PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Add optimization flags to CLI
if(MSVC)
    target_compile_options(inference_cli PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(inference_cli PRIVATE -Ofast -march=native)
endif()

###################################################################
# PART 3: Common Deployment and Resources
###################################################################

# Create models directory
add_custom_command(TARGET NanoInference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:NanoInference>/models
    COMMENT "Creating models directory..."
)

# Copy images to output directory
file(GLOB IMAGE_FILES "${CMAKE_SOURCE_DIR}/images/*.png" "${CMAKE_SOURCE_DIR}/images/*.jpg")
foreach(IMAGE_FILE ${IMAGE_FILES})
    get_filename_component(IMAGE_FILENAME ${IMAGE_FILE} NAME)
    add_custom_command(TARGET NanoInference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${IMAGE_FILE}"
            "$<TARGET_FILE_DIR:NanoInference>/images/${IMAGE_FILENAME}"
        COMMENT "Copying ${IMAGE_FILENAME} to build directory..."
    )
endforeach()

# Copy ML model files to output directory if they exist
foreach(FILE "gpt2_124M100Steps.bin" "vocab.bpe" "encoder.json")
    if(EXISTS "${CMAKE_SOURCE_DIR}/${FILE}")
        add_custom_command(TARGET NanoInference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/${FILE}"
                "$<TARGET_FILE_DIR:NanoInference>/${FILE}"
            COMMENT "Copying ${FILE} to NanoInference build directory..."
        )
        
        # Also copy to inference_cli target directory
        add_custom_command(TARGET inference_cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/${FILE}"
                "$<TARGET_FILE_DIR:inference_cli>/${FILE}"
            COMMENT "Copying ${FILE} to inference_cli build directory..."
        )
    else()
        message(STATUS "${FILE} not found in source directory")
    endif()
endforeach()

# Also copy the inference_cli executable to the NanoInference directory
add_custom_command(TARGET inference_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:inference_cli>
        $<TARGET_FILE_DIR:NanoInference>/
    COMMENT "Copying inference_cli executable to NanoInference directory..."
)

# Installation configuration
include(GNUInstallDirs)
install(TARGETS NanoInference inference_cli
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Finalize Qt executable
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(NanoInference)
endif()

# Platform-specific deployment for Qt
if(WIN32)
    # Check if we're using MSYS2
    if(MINGW)
        add_custom_command(TARGET NanoInference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/msys64/mingw64/bin/libgcc_s_seh-1.dll"
                $<TARGET_FILE_DIR:NanoInference>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/msys64/mingw64/bin/libwinpthread-1.dll"
                $<TARGET_FILE_DIR:NanoInference>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/msys64/mingw64/bin/libstdc++-6.dll"
                $<TARGET_FILE_DIR:NanoInference>
            COMMAND C:/Qt/6.8.0/mingw_64/bin/windeployqt.exe 
                --verbose 1
                --no-compiler-runtime
                --no-translations
                --no-opengl-sw
                $<TARGET_FILE:NanoInference>
            COMMENT "Deploying Qt dependencies for Windows/MSYS2..."
        )
    else()
        add_custom_command(TARGET NanoInference POST_BUILD
            COMMAND C:/Qt/6.8.0/mingw_64/bin/windeployqt.exe 
                --verbose 1
                --no-compiler-runtime
                --no-translations
                --no-opengl-sw
                $<TARGET_FILE:NanoInference>
            COMMENT "Deploying Qt dependencies for Windows..."
        )
    endif()
elseif(UNIX AND NOT APPLE)
    # Ubuntu/Linux deployment
    find_program(DEPLOY_QT_TOOL qt6-deploy)
    if(NOT DEPLOY_QT_TOOL)
        message(WARNING "qt6-deploy not found. Install with: sudo apt install qt6-tools-dev")
    else()
        add_custom_command(TARGET NanoInference POST_BUILD
            # Copy Qt plugins and libraries
            COMMAND ${DEPLOY_QT_TOOL} 
                $<TARGET_FILE:NanoInference>
                --appimage
            # Set RPATH
            COMMAND patchelf 
                --set-rpath '$ORIGIN/lib' 
                $<TARGET_FILE:NanoInference>
            COMMENT "Deploying Qt dependencies for Linux..."
        )
    endif()
endif()