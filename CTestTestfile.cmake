# CMake generated Testfile for 
# Source directory: /home/runner/workspace
# Build directory: /home/runner/workspace
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TestCard "/home/runner/workspace/test_card")
set_tests_properties(TestCard PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/workspace/CMakeLists.txt;117;add_test;/home/runner/workspace/CMakeLists.txt;0;")
add_test(TestGame "/home/runner/workspace/test_game")
set_tests_properties(TestGame PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/workspace/CMakeLists.txt;118;add_test;/home/runner/workspace/CMakeLists.txt;0;")
add_test(TestHandEvaluator "/home/runner/workspace/test_hand_evaluator")
set_tests_properties(TestHandEvaluator PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/workspace/CMakeLists.txt;119;add_test;/home/runner/workspace/CMakeLists.txt;0;")
subdirs("_deps/pybind11-build")
subdirs("_deps/googletest-build")
