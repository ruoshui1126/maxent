cmake_minimum_required (VERSION 2.8.0)
project("maxent")

include_directories( ${PROJECT_SOURCE_DIR}
 ${PROJECT_SOURCE_DIR}/utils)

add_executable(train train.cpp maxentor.cpp model.cpp)
