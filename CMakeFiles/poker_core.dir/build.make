# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /nix/store/q1nssraba326p2kp6627hldd2bhg254c-cmake-3.29.2/bin/cmake

# The command to remove a file.
RM = /nix/store/q1nssraba326p2kp6627hldd2bhg254c-cmake-3.29.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/runner/workspace

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/runner/workspace

# Include any dependencies generated for this target.
include CMakeFiles/poker_core.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/poker_core.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/poker_core.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/poker_core.dir/flags.make

CMakeFiles/poker_core.dir/src/utils.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/utils.cpp.o: src/utils.cpp
CMakeFiles/poker_core.dir/src/utils.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/poker_core.dir/src/utils.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/utils.cpp.o -MF CMakeFiles/poker_core.dir/src/utils.cpp.o.d -o CMakeFiles/poker_core.dir/src/utils.cpp.o -c /home/runner/workspace/src/utils.cpp

CMakeFiles/poker_core.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/utils.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/utils.cpp > CMakeFiles/poker_core.dir/src/utils.cpp.i

CMakeFiles/poker_core.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/utils.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/utils.cpp -o CMakeFiles/poker_core.dir/src/utils.cpp.s

CMakeFiles/poker_core.dir/src/card.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/card.cpp.o: src/card.cpp
CMakeFiles/poker_core.dir/src/card.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/poker_core.dir/src/card.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/card.cpp.o -MF CMakeFiles/poker_core.dir/src/card.cpp.o.d -o CMakeFiles/poker_core.dir/src/card.cpp.o -c /home/runner/workspace/src/card.cpp

CMakeFiles/poker_core.dir/src/card.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/card.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/card.cpp > CMakeFiles/poker_core.dir/src/card.cpp.i

CMakeFiles/poker_core.dir/src/card.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/card.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/card.cpp -o CMakeFiles/poker_core.dir/src/card.cpp.s

CMakeFiles/poker_core.dir/src/deck.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/deck.cpp.o: src/deck.cpp
CMakeFiles/poker_core.dir/src/deck.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/poker_core.dir/src/deck.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/deck.cpp.o -MF CMakeFiles/poker_core.dir/src/deck.cpp.o.d -o CMakeFiles/poker_core.dir/src/deck.cpp.o -c /home/runner/workspace/src/deck.cpp

CMakeFiles/poker_core.dir/src/deck.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/deck.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/deck.cpp > CMakeFiles/poker_core.dir/src/deck.cpp.i

CMakeFiles/poker_core.dir/src/deck.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/deck.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/deck.cpp -o CMakeFiles/poker_core.dir/src/deck.cpp.s

CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o: src/hand_evaluator.cpp
CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o -MF CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o.d -o CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o -c /home/runner/workspace/src/hand_evaluator.cpp

CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/hand_evaluator.cpp > CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.i

CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/hand_evaluator.cpp -o CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.s

CMakeFiles/poker_core.dir/src/player.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/player.cpp.o: src/player.cpp
CMakeFiles/poker_core.dir/src/player.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/poker_core.dir/src/player.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/player.cpp.o -MF CMakeFiles/poker_core.dir/src/player.cpp.o.d -o CMakeFiles/poker_core.dir/src/player.cpp.o -c /home/runner/workspace/src/player.cpp

CMakeFiles/poker_core.dir/src/player.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/player.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/player.cpp > CMakeFiles/poker_core.dir/src/player.cpp.i

CMakeFiles/poker_core.dir/src/player.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/player.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/player.cpp -o CMakeFiles/poker_core.dir/src/player.cpp.s

CMakeFiles/poker_core.dir/src/action.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/action.cpp.o: src/action.cpp
CMakeFiles/poker_core.dir/src/action.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/poker_core.dir/src/action.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/action.cpp.o -MF CMakeFiles/poker_core.dir/src/action.cpp.o.d -o CMakeFiles/poker_core.dir/src/action.cpp.o -c /home/runner/workspace/src/action.cpp

CMakeFiles/poker_core.dir/src/action.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/action.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/action.cpp > CMakeFiles/poker_core.dir/src/action.cpp.i

CMakeFiles/poker_core.dir/src/action.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/action.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/action.cpp -o CMakeFiles/poker_core.dir/src/action.cpp.s

CMakeFiles/poker_core.dir/src/game_state.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/game_state.cpp.o: src/game_state.cpp
CMakeFiles/poker_core.dir/src/game_state.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/poker_core.dir/src/game_state.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/game_state.cpp.o -MF CMakeFiles/poker_core.dir/src/game_state.cpp.o.d -o CMakeFiles/poker_core.dir/src/game_state.cpp.o -c /home/runner/workspace/src/game_state.cpp

CMakeFiles/poker_core.dir/src/game_state.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/game_state.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/game_state.cpp > CMakeFiles/poker_core.dir/src/game_state.cpp.i

CMakeFiles/poker_core.dir/src/game_state.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/game_state.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/game_state.cpp -o CMakeFiles/poker_core.dir/src/game_state.cpp.s

CMakeFiles/poker_core.dir/src/spingo_game.cpp.o: CMakeFiles/poker_core.dir/flags.make
CMakeFiles/poker_core.dir/src/spingo_game.cpp.o: src/spingo_game.cpp
CMakeFiles/poker_core.dir/src/spingo_game.cpp.o: CMakeFiles/poker_core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/poker_core.dir/src/spingo_game.cpp.o"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/poker_core.dir/src/spingo_game.cpp.o -MF CMakeFiles/poker_core.dir/src/spingo_game.cpp.o.d -o CMakeFiles/poker_core.dir/src/spingo_game.cpp.o -c /home/runner/workspace/src/spingo_game.cpp

CMakeFiles/poker_core.dir/src/spingo_game.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poker_core.dir/src/spingo_game.cpp.i"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/runner/workspace/src/spingo_game.cpp > CMakeFiles/poker_core.dir/src/spingo_game.cpp.i

CMakeFiles/poker_core.dir/src/spingo_game.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poker_core.dir/src/spingo_game.cpp.s"
	/nix/store/9bv7dcvmfcjnmg5mnqwqlq2wxfn8d7yi-gcc-wrapper-13.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/runner/workspace/src/spingo_game.cpp -o CMakeFiles/poker_core.dir/src/spingo_game.cpp.s

# Object files for target poker_core
poker_core_OBJECTS = \
"CMakeFiles/poker_core.dir/src/utils.cpp.o" \
"CMakeFiles/poker_core.dir/src/card.cpp.o" \
"CMakeFiles/poker_core.dir/src/deck.cpp.o" \
"CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o" \
"CMakeFiles/poker_core.dir/src/player.cpp.o" \
"CMakeFiles/poker_core.dir/src/action.cpp.o" \
"CMakeFiles/poker_core.dir/src/game_state.cpp.o" \
"CMakeFiles/poker_core.dir/src/spingo_game.cpp.o"

# External object files for target poker_core
poker_core_EXTERNAL_OBJECTS =

libpoker_core.so: CMakeFiles/poker_core.dir/src/utils.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/card.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/deck.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/hand_evaluator.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/player.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/action.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/game_state.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/src/spingo_game.cpp.o
libpoker_core.so: CMakeFiles/poker_core.dir/build.make
libpoker_core.so: CMakeFiles/poker_core.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/runner/workspace/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library libpoker_core.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/poker_core.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/poker_core.dir/build: libpoker_core.so
.PHONY : CMakeFiles/poker_core.dir/build

CMakeFiles/poker_core.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/poker_core.dir/cmake_clean.cmake
.PHONY : CMakeFiles/poker_core.dir/clean

CMakeFiles/poker_core.dir/depend:
	cd /home/runner/workspace && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/runner/workspace /home/runner/workspace /home/runner/workspace /home/runner/workspace /home/runner/workspace/CMakeFiles/poker_core.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/poker_core.dir/depend

