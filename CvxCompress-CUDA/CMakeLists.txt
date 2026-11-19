cmake_minimum_required(VERSION 3.18) 
project(CvxCompress-CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CVXCOMPRESS_CUDA_BUILD_TESTS "Build the GoogleTest unit tests (downloads googletest at configure time)" ON)
option(CVXCOMPRESS_CUDA_BUILD_DEVITOCOMP "Build libdevitocomp.so, the DevitoPRO DevitoComp provider" OFF)

if(CVXCOMPRESS_CUDA_BUILD_DEVITOCOMP)
    # DevitoPRO operators link libcudart.so; the provider must share that runtime.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

# Library target 
add_library(cvxcompress_cuda STATIC
    src/CvxCompress_CUDA.cu
    src/CvxCompress_CUDA_helpers.cu
    io/Read_Raw_Volume.cpp
    io/Write_Raw_Volume.cpp
)
target_compile_options(cvxcompress_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=64>
)
target_include_directories(cvxcompress_cuda PUBLIC include include/Common include/IO src)
set_target_properties(cvxcompress_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CVXCOMPRESS_CUDA_BUILD_DEVITOCOMP)
    find_package(CUDAToolkit REQUIRED)
    find_path(DEVITOCOMP_INCLUDE_DIR devitocomp.h
        DOC "Directory with devitocomp.h (devitopro_ext/devitocomp/include in DevitoPRO)")
    if(NOT DEVITOCOMP_INCLUDE_DIR)
        message(FATAL_ERROR "devitocomp.h not found; set -DDEVITOCOMP_INCLUDE_DIR=<devitopro>/devitopro_ext/devitocomp/include")
    endif()

    add_library(devitocomp SHARED devitocomp/devitocomp_provider.cu)
    target_include_directories(devitocomp PRIVATE ${DEVITOCOMP_INCLUDE_DIR})
    target_link_libraries(devitocomp PRIVATE cvxcompress_cuda)
    # Export only the devitocomp_* C ABI, not the static library's C++/thrust symbols.
    target_link_options(devitocomp PRIVATE "LINKER:--exclude-libs,ALL")
    set_target_properties(devitocomp PROPERTIES INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

    # DevitoPRO looks for exactly $DEVITOCOMP_ROOT/include/devitocomp.h and $DEVITOCOMP_ROOT/lib/libdevitocomp.so.
    install(TARGETS devitocomp LIBRARY DESTINATION lib)
    install(FILES ${DEVITOCOMP_INCLUDE_DIR}/devitocomp.h DESTINATION include)
endif()

if(CVXCOMPRESS_CUDA_BUILD_TESTS)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Register with CTest
enable_testing()
include(GoogleTest)

add_executable(numerical_correctness_test
    tests/unit/numerical_correctness/error_bound_test.cu
    tests/unit/numerical_correctness/cr_vs_scale_monotonicity_test.cu
    tests/unit/numerical_correctness/device_rms_accuracy_test.cu
    tests/unit/numerical_correctness/compressed_length_accuracy_test.cu
    tests/unit/numerical_correctness/determinism_test.cu)
set_target_properties(numerical_correctness_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(numerical_correctness_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Common
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/unit/numerical_correctness)
target_link_libraries(numerical_correctness_test cvxcompress_cuda GTest::gtest_main)
gtest_discover_tests(numerical_correctness_test)

add_executable(robustness_test
    tests/unit/robustness/degenerate_input_test.cu
    tests/unit/robustness/dimensions_test.cu
    tests/unit/robustness/buffer_and_reuse_test.cu)
set_target_properties(robustness_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(robustness_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Common
    ${CMAKE_SOURCE_DIR}/tests/unit/robustness)
target_link_libraries(robustness_test cvxcompress_cuda GTest::gtest_main)
gtest_discover_tests(robustness_test)

add_executable(stage_isolation_test
    tests/unit/stage_isolation/stage_harness.cu
    tests/unit/stage_isolation/wavelet_stage_test.cu
    tests/unit/stage_isolation/quantization_stage_test.cu
    tests/unit/stage_isolation/rle_stage_test.cu)
set_target_properties(stage_isolation_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(stage_isolation_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Common
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests/unit/stage_isolation)
target_link_libraries(stage_isolation_test GTest::gtest_main)
gtest_discover_tests(stage_isolation_test)
endif()
