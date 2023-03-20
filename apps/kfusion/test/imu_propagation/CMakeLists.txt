cmake_minimum_required(VERSION 3.10)
project(imu_propagation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ../../cmake)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(TooN   REQUIRED)
find_package(OpenCL )
find_package(CUDA   )
find_package(GLUT   )
find_package(OpenGL )
find_package(OpenNI2)
find_package(PAPI)
find_package(OpenMP)

find_package(PkgConfig)


include_directories(../../include ../../src/cpp/segmentation ../../src/cpp/preprocessing
        ../../src/cpp/object ../../src/cpp/tracking ../../src/cpp/dataProcessing
        ../../src/cpp/imuMeasurement)

file(GLOB TRACKING "../../src/cpp/tracking/*.cpp")
file(GLOB DATAPROCESSING "../../src/cpp/dataProcessing/*.cpp")
file(GLOB IMUMEASUREMENT "../../src/cpp/imuMeasurement/*.cpp")


set(PROJECT_TEST_NAME imu_propagation)
set(UNIT_TEST_NAME ${PROJECT_TEST_NAME}-unittest)
add_executable(${UNIT_TEST_NAME} imu_propagation_unittest.cpp ${IMUMEASUREMENT})
target_link_libraries(${UNIT_TEST_NAME}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OKVIS_LIBRARIES}
        ${BRISK_LIBRARIES}
        ${CERES_LIBRARIES}
        okvis_util
        okvis_kinematics
        okvis_time
        okvis_cv
        okvis_common
        okvis_ceres
        okvis_timing
        okvis_matcher
        okvis_frontend
        okvis_multisensor_processing)


set(PROJECT_TEST_NAME imu_optimization)
set(UNIT_TEST_NAME ${PROJECT_TEST_NAME}-unittest)
add_executable(${UNIT_TEST_NAME} imu_optimization_unittest.cpp ${IMUMEASUREMENT} ${DATAPROCESSING})
target_link_libraries(${UNIT_TEST_NAME}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OKVIS_LIBRARIES}
        ${BRISK_LIBRARIES}
        ${CERES_LIBRARIES}
        okvis_util
        okvis_kinematics
        okvis_time
        okvis_cv
        okvis_common
        okvis_ceres
        okvis_timing
        okvis_matcher
        okvis_frontend
        okvis_multisensor_processing)
