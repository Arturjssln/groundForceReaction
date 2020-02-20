project(contactForces)

cmake_minimum_required(VERSION 3.2)

set(BLD_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build/")

file(MAKE_DIRECTORY ${BLD_PATH})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BLD_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BLD_PATH})

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES *.h)

set(OPENSIM_INSTALL_DIR $ENV{OPENSIM_HOME} CACHE PATH "Top-level directory of OpenSim install")

set(PLUGIN_NAME "contactForces" CACHE STRING "Name of shared library to create")

# OpenSim uses C++11 language features.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenSim REQUIRED PATHS "${OPENSIM_INSTALL_DIR}")

add_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${INCLUDE_FILES})

target_link_libraries(${PLUGIN_NAME}
  ${OpenSim_LIBRARIES}
)

set_target_properties(${PLUGIN_NAME} PROPERTIES
  DEFINE_SYMBOL OSIMPLUGIN_EXPORTS
  PROJECT_LABEL "Libraries - ${PLUGIN_NAME}"
)