cmake_minimum_required(VERSION 3.12)
project(lidar2image_GAN)

# Find PCL and Boost packages
if(APPLE)
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

# Find required packages
find_package(pybind11 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
# Set source files

# Add executable
add_executable(read_pointcloud src/read_pointcloud.cpp)
target_link_libraries(read_pointcloud ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(lidar2image src/pc_to_img.cpp)
target_link_libraries(lidar2image ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})


add_library(prepare_data MODULE src/prepare_data.cpp)
target_include_directories(prepare_data PRIVATE Pybind11::module ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(prepare_data PRIVATE  ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(prepare_data PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_compile_features(prepare_data PRIVATE cxx_std_17)
