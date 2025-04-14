cmake_minimum_required(VERSION 3.10)
project(poker_simulator VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and set PYTHON_EXECUTABLE
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# Find dependencies
# Try to find pybind11 using find_package first
find_package(pybind11 QUIET)

# If not found, add it via add_subdirectory if possible
if(NOT pybind11_FOUND)
    # First, try to use the Python's pybind11 module
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE PYBIND11_FOUND_PYTHON
    )
    
    if(NOT PYBIND11_FOUND_PYTHON)
        message(STATUS "pybind11 not found via Python, trying to fetch it...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz
            URL_HASH SHA256=eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec
        )
        FetchContent_MakeAvailable(pybind11)
    else()
        message(STATUS "Found pybind11 at ${PYBIND11_CMAKE_DIR}")
        # Add the directory containing pybind11Config.cmake to CMAKE_PREFIX_PATH
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    endif()
endif()

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define library sources
set(LIB_SOURCES
    src/utils.cpp
    src/card.cpp
    src/deck.cpp
    src/hand_evaluator.cpp
    src/player.cpp
    src/action.cpp
    src/game_state.cpp
    src/spingo_game.cpp
)

# Create shared library
add_library(poker_core SHARED ${LIB_SOURCES})
target_include_directories(poker_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create Python bindings
pybind11_add_module(poker_core_python python/poker_module.cpp)
target_link_libraries(poker_core_python PRIVATE poker_core)
set_target_properties(poker_core_python PROPERTIES OUTPUT_NAME poker_core)

# Enable testing
enable_testing()

# Add GoogleTest
find_package(GTest QUIET)
if(NOT GTEST_FOUND)
    # Download and build GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
        URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Add test executables
add_executable(test_card tests/test_card.cpp)
target_link_libraries(test_card PRIVATE poker_core GTest::gtest GTest::gtest_main)

add_executable(test_game tests/test_game.cpp)
target_link_libraries(test_game PRIVATE poker_core GTest::gtest GTest::gtest_main)

add_executable(test_hand_evaluator tests/test_hand_evaluator.cpp)
target_link_libraries(test_hand_evaluator PRIVATE poker_core GTest::gtest GTest::gtest_main)

# Add tests to CTest
add_test(NAME TestCard COMMAND test_card)
add_test(NAME TestGame COMMAND test_game)
add_test(NAME TestHandEvaluator COMMAND test_hand_evaluator)

# Installation
install(TARGETS poker_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include)

# Set Python site-packages directory if not already set
if(NOT DEFINED PYTHON_SITE_PACKAGES_DIR)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

install(TARGETS poker_core_python
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES_DIR}
        RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES_DIR})

install(DIRECTORY include/ DESTINATION include)
