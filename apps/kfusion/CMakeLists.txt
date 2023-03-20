# SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa 
# SPDX-FileCopyrightText: 2020-2023 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2017-2023 Binbin Xu
# SPDX-FileCopyrightText: 2021-2023 Yifei Ren
# SPDX-License-Identifier: CC0-1.0


cmake_minimum_required(VERSION 3.1)

project(vimid)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wno-unused-function -Wno-error=deprecated-declarations -Wno-deprecated-declarations")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ")
  add_definitions(-DDEBUG=1)
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O1  ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O1")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# add_compile_options(-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -Werror -Wno-error=deprecated-declarations -Wno-error=array-bounds)
add_compile_options(-std=c++1y -march=native)

if(APPLE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__APPLE__")
endif(APPLE)


#  ----------------- okvis -----------------
set(OKVIS_INSTALLATION /usr/local/okvis/usr/local) # point to installation

# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS})

# require okvis
find_package( okvis 1.1 REQUIRED)
include_directories(${OKVIS_INCLUDE_DIRS})

# require brisk
find_package( brisk 2 REQUIRED)
include_directories(${BRISK_INCLUDE_DIRS})

# require ceres
list(APPEND CMAKE_PREFIX_PATH ${OKVIS_INSTALLATION})
find_package( Ceres REQUIRED )
include_directories(${CERES_INCLUDE_DIRS})

# require OpenGV
find_package(opengv REQUIRED)

# require boost
find_package(Boost COMPONENTS filesystem system REQUIRED)
if (Boost_FOUND)
  message("Using Boost=" ${Boost_INCLUDE_DIRS} ${Boost_LIBRARIES})
endif()
include_directories(${Boost_INCLUDE_DIRS})


#  ----------------- okvis -----------------
#add_subdirectory( thirdparty/g2o     )

# ---- PREPARE COMMON DEPENDENCIES  ------------ 

file(GLOB SEGMENTATION "src/cpp/segmentation/*.cpp")
file(GLOB PREPROCESSING "src/cpp/preprocessing/*.cpp")
file(GLOB OBJECT "src/cpp/object/*.cpp")
file(GLOB TRACKING "src/cpp/tracking/*.cpp")
file(GLOB RENDERING "src/cpp/rendering/*.cpp")
file(GLOB DATAPROCESSING "src/cpp/dataProcessing/*.cpp")

function(add_version appname version compile_flags link_flags)

  if(APPLE)
    set(main_common_libraries     ${link_flags} stdc++ pthread)
  else(APPLE)
    set(main_common_libraries     ${link_flags} rt stdc++ pthread)
  endif(APPLE)

  set(main_common_compile_flags ${compile_flags})

  if (OPENNI2_FOUND)
    include_directories(${OPENNI2_INCLUDE_PATHS})
    set(main_common_libraries ${main_common_libraries} ${OPENNI2_LIBRARIES})
    set(main_common_compile_flags "-DDO_OPENNI ${main_common_compile_flags}")
  endif(OPENNI2_FOUND)

  if (librealsense_FOUND)
    include_directories(${REALSENSE_INCLUDE_DIR})
    set(main_common_libraries ${main_common_libraries} ${librealsense_LIBRARIES})
    set(main_common_compile_flags "-DDO_REALSENSE ${main_common_compile_flags}")
  endif(librealsense_FOUND)


  #  ------ BENCHMARK --------
  # add_executable(${appname}-benchmark-${version} src/benchmark.cpp
  #         src/PowerMonitor.cpp ${PREPROCESSING} ${SEGMENTATION} ${OBJECT}
  #         ${TRACKING} ${RENDERING} ${IMUMEASUREMENT})
  # target_link_libraries(${appname}-benchmark-${version} ${appname}-${version} ${main_common_libraries})
  # SET_TARGET_PROPERTIES(${appname}-benchmark-${version}
  #   PROPERTIES COMPILE_FLAGS "${main_common_compile_flags}")

  #  ------ MAIN --------
  if (GLUT_FOUND)
    add_executable(${appname}-main-${version} src/main.cpp src/PowerMonitor.cpp
            src/reader.cpp src/vimidAction.cpp 
            ${PREPROCESSING} ${SEGMENTATION} ${OBJECT}
            ${TRACKING} ${RENDERING} ${DATAPROCESSING} ${IMUMEASUREMENT})
    
            SET_TARGET_PROPERTIES(${appname}-main-${version}
      PROPERTIES COMPILE_FLAGS "${main_common_compile_flags}")


    if(APPLE)
      target_link_libraries(${appname}-main-${version} ${appname}-${version} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  ${main_common_libraries} )
    else(APPLE)
      target_link_libraries(${appname}-main-${version} ${appname}-${version} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  ${main_common_libraries} GL
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

    endif(APPLE)
  endif(GLUT_FOUND)
  #  ------ QT --------
  if (Qt5OpenGL_FOUND)

    file(GLOB qt_srcs qt/*)

    QT5_ADD_RESOURCES( ResourceSources  "qt/images.qrc")
    set(qt_srcs ${qt_srcs} ${ResourceSources} include/PowerMonitor.h ../../utils/perfstats.h)
    SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
    SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    include_directories(qt)
    add_definitions( -D_FILE_OFFSET_BITS=64 -fPIC )

    add_executable(${appname}-qt-${version} src/main.cpp src/PowerMonitor.cpp ${qt_srcs}
            src/reader.cpp src/vimidAction.cpp ${PREPROCESSING} ${SEGMENTATION} ${OBJECT} 
            ${TRACKING} ${RENDERING} ${DATAPROCESSING} ${IMUMEASUREMENT})

    ### qt5_use_modules is not available with cmake 2.8.7 ###
    #qt5_use_modules(qtgui-${version} Core Gui Widgets OpenGL PrintSupport)

    add_definitions(${Qt5Widgets_DEFINITIONS})
    add_definitions(${Qt5OpenGL_DEFINITIONS})
    add_definitions(${Qt5PrintSuppoer_DEFINITIONS})
    include_directories(${Qt5Widgets_INCLUDE_DIRS})
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    include_directories(${Qt5PrintSupport_INCLUDE_DIRS})
    include_directories(${Qt5Gui_INCLUDE_DIRS})

    SET_TARGET_PROPERTIES(${appname}-qt-${version} PROPERTIES COMPILE_FLAGS "${main_common_compile_flags} -std=c++0x  -D__QT__")

    if(APPLE)
      target_link_libraries(${appname}-qt-${version} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::PrintSupport 
        ${main_common_libraries} ${appname}-${version}  ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}   )  
    else(APPLE) 
      target_link_libraries(${appname}-qt-${version} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::PrintSupport 
        ${main_common_libraries} ${appname}-${version}  ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  GL )  
    endif(APPLE)


  endif(Qt5OpenGL_FOUND)

endfunction()

set(appname kfusion)

find_package(TooN   REQUIRED)
find_package(OpenCL )
find_package(CUDA   )
find_package(GLUT   )
find_package(OpenGL )
find_package(OpenNI2)
find_package(PAPI)
find_package(OpenMP)

find_package(PkgConfig)
# find_package(Qt5Core)
# find_package(Qt5Widgets)
# find_package(Qt5Gui)
# find_package(Qt5OpenGL)
# find_package(Qt5PrintSupport)

#find_package(G2O    REQUIRED)
#if(NOT G2O_FOUND)
#  message(FATAL_ERROR "G2O not found.")
#endif()
#if (G2O_FOUND)
#  include_directories(${G2O_INCLUDE_DIRS})
##  link_directories(${G2O_LIBRARY_DIRS})
##  message(STATUS "G2O=" ${G2O_INCLUDE_DIRS} ${G2O_LIBRARY_DIRS})
#endif()

find_package(OpenCV 3.0)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package (Eigen3 REQUIRED NO_MODULE)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_libraries(${OpenCV_LIBRARIES})
endif ()

include_directories(./include ${TOON_INCLUDE_PATHS} ${EIGEN3_INCLUDE_DIR} ./thirdparty ./thirdparty/cnpy
        ../../lib ../../utils ${PCL_INCLUDE_DIRS} ./src/cpp/segmentation ./src/cpp/preprocessing
        ./src/cpp/object ./src/cpp/tracking ./src/cpp/dataProcessing ./src/cpp/frontEnd)


find_package (GeographicLib)
if (GeographicLib_FOUND)
    include_directories (${GeographicLib_INCLUDE_DIRS})
    add_definitions (${GeographicLib_DEFINITIONS})
    link_libraries(${GeographicLib_LIBRARIES})
endif ()

# ---- PREPARE COMMON DEPENDENCIES  ------------ 
add_library(tiny_obj thirdparty/tiny_obj_loader.cc)
if(APPLE)
  set(common_libraries stdc++ pthread tiny_obj lodepng)
else(APPLE)
  set(common_libraries rt stdc++ pthread tiny_obj lodepng)
endif(APPLE)


set(common_compile_flags "")

OPTION(STATS "Enable stats collection" OFF)
IF(STATS)
  add_definitions(-DSTATS)
ENDIF(STATS)

# ----------------- CPP VERSION -----------------

add_library(${appname}-cpp  src/cpp/kfusion.cpp src/cpp/frontEnd/pose_estimation.h src/cpp/KeyFrame/KeyFrame.h src/cpp/KeyFrame/KeyFrame.cpp)
target_compile_options(${appname}-cpp PRIVATE -Wall -Wextra -Werror -Wno-unknown-pragmas)
target_link_libraries(${appname}-cpp   ${common_libraries} cnpy)
add_version(${appname} cpp "" "")

# ----------------- OPENMP VERSION ----------------- 

if(OPENMP_FOUND)
  add_library(${appname}-openmp  src/cpp/kfusion.cpp src/cpp/KeyFrame/KeyFrame.h src/cpp/KeyFrame/KeyFrame.cpp)
  # target_compile_options(${appname}-openmp PRIVATE ${OpenMP_CXX_FLAGS} -std=c++11 -Wall -Wextra  -Wno-unknown-pragmas -qopt-report=2 -qopt-report-file=./icc_report.txt -qopt-report-phase=vec)
  target_compile_options(${appname}-openmp PRIVATE ${OpenMP_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unknown-pragmas)
  target_link_libraries(${appname}-openmp   ${common_libraries} cnpy)
  add_version(${appname} openmp ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
endif()

#  ----------------- thirdparty ----------------- 

add_subdirectory( thirdparty         )
add_subdirectory( thirdparty/cnpy    )