cmake_minimum_required(VERSION 2.8.11)

project(okvis_common)

IF(libvisensor_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  MESSAGE(STATUS "Found libvisensor. Setting HAVE_LIBVISENSOR flag.")
ENDIF()

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# build the library
#if(CAMERA_DRIVER MATCHES "realsense")
if(librealsense_FOUND)
add_library(${PROJECT_NAME} STATIC 
  src/VioInterface.cpp
  src/VioParametersReader.cpp
  src/ZR300Reader.cpp
  include/okvis/FrameTypedefs.hpp
  include/okvis/Measurements.hpp
  include/okvis/Parameters.hpp
  include/okvis/Variables.hpp
  include/okvis/VioBackendInterface.hpp
  include/okvis/VioFrontendInterface.hpp
  include/okvis/VioInterface.hpp
  include/okvis/VioParametersReader.hpp
  include/okvis/ZR300Reader.h)
else()
add_library(${PROJECT_NAME} STATIC 
  src/VioInterface.cpp
  src/VioParametersReader.cpp
  src/ZR300Reader.cpp
  include/okvis/FrameTypedefs.hpp
  include/okvis/Measurements.hpp
  include/okvis/Parameters.hpp
  include/okvis/Variables.hpp
  include/okvis/VioBackendInterface.hpp
  include/okvis/VioFrontendInterface.hpp
  include/okvis/VioInterface.hpp
  include/okvis/VioParametersReader.hpp
  include/okvis/ZR300Reader.h)
endif()

if(${librealsense_FOUND})
  target_link_libraries(${PROJECT_NAME}
    PUBLIC okvis_util
    PUBLIC okvis_kinematics 
    PUBLIC okvis_time 
    PUBLIC okvis_cv
    PUBLIC ${librealsense_LIBRARIES}
  )
else()
  # and link it                     
  target_link_libraries(${PROJECT_NAME}
    PUBLIC okvis_util
    PUBLIC okvis_kinematics 
    PUBLIC okvis_time 
    PUBLIC okvis_cv
    PUBLIC ${REALSENSE_LIB}
  )
endif()

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
