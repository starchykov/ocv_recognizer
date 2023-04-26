# This is a CMake script that builds an executable called "ocv_recognizer".

# Minimum version of CMake that is required to build the project.
cmake_minimum_required(VERSION 3.24)

# Defines the project name as "ocv_recognizer".
project(ocv_recognizer)

# Set the C++ standard to C++17.
set(CMAKE_CXX_STANDARD 17)

# Find the OpenCV package, which is a library for computer vision and image processing, required to build the project.
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(TESSERACT REQUIRED tesseract)
pkg_search_module(LEPTONICA REQUIRED lept)


# The next two lines include directories that contain header files.
# The first directory is "ocv_recognizer", which is the directory containing the source code files for the project.
include_directories(ocv_recognizer)

# Following directory is the directory containing the header files for the OpenCV library.
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TESSERACT_INCLUDE_DIRS})
include_directories(${LEPTONICA_INCLUDE_DIRS})


link_directories(${TESSERACT_LIBRARY_DIRS})
link_directories(${LEPTONICA_LIBRARY_DIRS})


# The following lines add the source code files to the executable using the "add_executable" function.
# The source code files are listed as arguments to this function.
add_executable(
        ocv_recognizer
        )

# The "target_link_libraries" function is used to link the OpenCV library to the executable.
# The OpenCV_LIBS variable contains the names of the libraries required to link with OpenCV.
target_link_libraries(ocv_recognizer ${OpenCV_LIBS})
target_link_libraries(ocv_recognizer ${TESSERACT_LIBRARIES})
target_link_libraries(ocv_recognizer ${LEPTONICA_LIBRARIES})
