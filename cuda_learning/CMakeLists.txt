cmake_minimum_required(VERSION 3.2)
project(cuda_learning CUDA)

# 设置 CUDA 架构版本
set(CMAKE_CUDA_ARCHITECTURES 80)

# 设置全局的头文件搜索路径
include_directories(${CMAKE_SOURCE_DIR}/src)

# 添加可执行文件及其源文件
add_executable(demo src/main.cu)

# 指定 CUDA 版本和语言标准
set_target_properties(demo PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
