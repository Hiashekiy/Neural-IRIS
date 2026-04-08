cmake_minimum_required(VERSION 3.18)
project(corridor_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CUDA_PROVIDER "Enable ONNX Runtime CUDA provider" ON)

if(NOT DEFINED ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Please set ONNXRUNTIME_ROOT to your onnxruntime install root.")
endif()

set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

if(WIN32)
  set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
  set(ORT_LIB "${ORT_LIB_DIR}/onnxruntime.lib")
else()
  set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
  set(ORT_LIB "${ORT_LIB_DIR}/libonnxruntime.so")
endif()

add_executable(corridor_cpp_infer
  src/main_ort.cpp
  src/geometry2d.cpp
)

add_library(corridor_cpp_bridge SHARED
  src/bridge.cpp
  src/geometry2d.cpp
)

target_include_directories(corridor_cpp_infer PRIVATE
  ${ORT_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(corridor_cpp_bridge PRIVATE
  ${ORT_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(corridor_cpp_infer PRIVATE ${ORT_LIB})
target_link_libraries(corridor_cpp_bridge PRIVATE ${ORT_LIB})

if(WIN32)
  add_custom_command(TARGET corridor_cpp_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_LIB_DIR}/onnxruntime.dll"
      "$<TARGET_FILE_DIR:corridor_cpp_infer>/onnxruntime.dll")
  add_custom_command(TARGET corridor_cpp_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_LIB_DIR}/onnxruntime.dll"
      "$<TARGET_FILE_DIR:corridor_cpp_bridge>/onnxruntime.dll")

  if(USE_CUDA_PROVIDER)
    foreach(target corridor_cpp_infer corridor_cpp_bridge)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ORT_LIB_DIR}/onnxruntime_providers_cuda.dll"
          "$<TARGET_FILE_DIR:${target}>/onnxruntime_providers_cuda.dll")
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ORT_LIB_DIR}/onnxruntime_providers_shared.dll"
          "$<TARGET_FILE_DIR:${target}>/onnxruntime_providers_shared.dll")
    endforeach()
  endif()
endif()

if(MSVC)
  target_compile_options(corridor_cpp_infer PRIVATE /O2)
  target_compile_options(corridor_cpp_bridge PRIVATE /O2)
else()
  target_compile_options(corridor_cpp_infer PRIVATE -O3)
  target_compile_options(corridor_cpp_bridge PRIVATE -O3)
endif()

if(USE_CUDA_PROVIDER)
  target_compile_definitions(corridor_cpp_infer PRIVATE USE_CUDA_PROVIDER=1)
  target_compile_definitions(corridor_cpp_bridge PRIVATE USE_CUDA_PROVIDER=1)
endif()
