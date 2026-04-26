cmake_minimum_required(VERSION 3.15)
project(LBM_Simulation VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)          

include_directories(include)

add_library(lbm_lib 
    src/lbm/lbm_gpu.cu
    src/filter/smc_gpu.cu
    utils/metrics.cu
)

add_executable(LBM_App 
    app/main.cu
    renderer/render_gpu.cu 
)

target_link_libraries(LBM_App 
    PRIVATE 
    lbm_lib 
    glfw 
    OpenGL::GL
    GLEW::GLEW                 
    curand                     
)

set_target_properties(LBM_App PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)