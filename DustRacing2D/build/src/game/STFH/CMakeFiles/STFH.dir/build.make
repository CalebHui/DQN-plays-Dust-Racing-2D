# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build

# Include any dependencies generated for this target.
include src/game/STFH/CMakeFiles/STFH.dir/depend.make

# Include the progress variables for this target.
include src/game/STFH/CMakeFiles/STFH.dir/progress.make

# Include the compile flags for this target's objects.
include src/game/STFH/CMakeFiles/STFH.dir/flags.make

src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o: src/game/STFH/CMakeFiles/STFH.dir/flags.make
src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o: ../src/game/STFH/data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/STFH.dir/data.cpp.o -c /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/data.cpp

src/game/STFH/CMakeFiles/STFH.dir/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/STFH.dir/data.cpp.i"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/data.cpp > CMakeFiles/STFH.dir/data.cpp.i

src/game/STFH/CMakeFiles/STFH.dir/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/STFH.dir/data.cpp.s"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/data.cpp -o CMakeFiles/STFH.dir/data.cpp.s

src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.requires:

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.requires

src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.provides: src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.requires
	$(MAKE) -f src/game/STFH/CMakeFiles/STFH.dir/build.make src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.provides.build
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.provides

src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.provides.build: src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o


src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o: src/game/STFH/CMakeFiles/STFH.dir/flags.make
src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o: ../src/game/STFH/device.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/STFH.dir/device.cpp.o -c /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/device.cpp

src/game/STFH/CMakeFiles/STFH.dir/device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/STFH.dir/device.cpp.i"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/device.cpp > CMakeFiles/STFH.dir/device.cpp.i

src/game/STFH/CMakeFiles/STFH.dir/device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/STFH.dir/device.cpp.s"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/device.cpp -o CMakeFiles/STFH.dir/device.cpp.s

src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.requires:

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.requires

src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.provides: src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.requires
	$(MAKE) -f src/game/STFH/CMakeFiles/STFH.dir/build.make src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.provides.build
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.provides

src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.provides.build: src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o


src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o: src/game/STFH/CMakeFiles/STFH.dir/flags.make
src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o: ../src/game/STFH/listener.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/STFH.dir/listener.cpp.o -c /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/listener.cpp

src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/STFH.dir/listener.cpp.i"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/listener.cpp > CMakeFiles/STFH.dir/listener.cpp.i

src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/STFH.dir/listener.cpp.s"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/listener.cpp -o CMakeFiles/STFH.dir/listener.cpp.s

src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.requires:

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.requires

src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.provides: src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.requires
	$(MAKE) -f src/game/STFH/CMakeFiles/STFH.dir/build.make src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.provides.build
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.provides

src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.provides.build: src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o


src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o: src/game/STFH/CMakeFiles/STFH.dir/flags.make
src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o: ../src/game/STFH/location.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/STFH.dir/location.cpp.o -c /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/location.cpp

src/game/STFH/CMakeFiles/STFH.dir/location.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/STFH.dir/location.cpp.i"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/location.cpp > CMakeFiles/STFH.dir/location.cpp.i

src/game/STFH/CMakeFiles/STFH.dir/location.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/STFH.dir/location.cpp.s"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/location.cpp -o CMakeFiles/STFH.dir/location.cpp.s

src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.requires:

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.requires

src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.provides: src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.requires
	$(MAKE) -f src/game/STFH/CMakeFiles/STFH.dir/build.make src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.provides.build
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.provides

src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.provides.build: src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o


src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o: src/game/STFH/CMakeFiles/STFH.dir/flags.make
src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o: ../src/game/STFH/source.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/STFH.dir/source.cpp.o -c /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/source.cpp

src/game/STFH/CMakeFiles/STFH.dir/source.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/STFH.dir/source.cpp.i"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/source.cpp > CMakeFiles/STFH.dir/source.cpp.i

src/game/STFH/CMakeFiles/STFH.dir/source.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/STFH.dir/source.cpp.s"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH/source.cpp -o CMakeFiles/STFH.dir/source.cpp.s

src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.requires:

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.requires

src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.provides: src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.requires
	$(MAKE) -f src/game/STFH/CMakeFiles/STFH.dir/build.make src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.provides.build
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.provides

src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.provides.build: src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o


src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o: src/game/STFH/CMakeFiles/STFH.dir/flags.make
src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o: src/game/STFH/STFH_autogen/mocs_compilation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o -c /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH/STFH_autogen/mocs_compilation.cpp

src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.i"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH/STFH_autogen/mocs_compilation.cpp > CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.i

src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.s"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH/STFH_autogen/mocs_compilation.cpp -o CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.s

src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.requires:

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.requires

src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.provides: src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.requires
	$(MAKE) -f src/game/STFH/CMakeFiles/STFH.dir/build.make src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.provides.build
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.provides

src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.provides.build: src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o


# Object files for target STFH
STFH_OBJECTS = \
"CMakeFiles/STFH.dir/data.cpp.o" \
"CMakeFiles/STFH.dir/device.cpp.o" \
"CMakeFiles/STFH.dir/listener.cpp.o" \
"CMakeFiles/STFH.dir/location.cpp.o" \
"CMakeFiles/STFH.dir/source.cpp.o" \
"CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o"

# External object files for target STFH
STFH_EXTERNAL_OBJECTS =

src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/build.make
src/game/STFH/libSTFH.a: src/game/STFH/CMakeFiles/STFH.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libSTFH.a"
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && $(CMAKE_COMMAND) -P CMakeFiles/STFH.dir/cmake_clean_target.cmake
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/STFH.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/game/STFH/CMakeFiles/STFH.dir/build: src/game/STFH/libSTFH.a

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/build

src/game/STFH/CMakeFiles/STFH.dir/requires: src/game/STFH/CMakeFiles/STFH.dir/data.cpp.o.requires
src/game/STFH/CMakeFiles/STFH.dir/requires: src/game/STFH/CMakeFiles/STFH.dir/device.cpp.o.requires
src/game/STFH/CMakeFiles/STFH.dir/requires: src/game/STFH/CMakeFiles/STFH.dir/listener.cpp.o.requires
src/game/STFH/CMakeFiles/STFH.dir/requires: src/game/STFH/CMakeFiles/STFH.dir/location.cpp.o.requires
src/game/STFH/CMakeFiles/STFH.dir/requires: src/game/STFH/CMakeFiles/STFH.dir/source.cpp.o.requires
src/game/STFH/CMakeFiles/STFH.dir/requires: src/game/STFH/CMakeFiles/STFH.dir/STFH_autogen/mocs_compilation.cpp.o.requires

.PHONY : src/game/STFH/CMakeFiles/STFH.dir/requires

src/game/STFH/CMakeFiles/STFH.dir/clean:
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH && $(CMAKE_COMMAND) -P CMakeFiles/STFH.dir/cmake_clean.cmake
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/clean

src/game/STFH/CMakeFiles/STFH.dir/depend:
	cd /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/src/game/STFH /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH /home/caleb/Desktop/caleb/CMSC/CMSC5721/DustRacing2D/build/src/game/STFH/CMakeFiles/STFH.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/game/STFH/CMakeFiles/STFH.dir/depend

