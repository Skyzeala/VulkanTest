Learning how to use vulkan one step at a time

The goal with the build process was to easily build on other systems as well as have a portable binary

Build steps:
- open mingw64.exe and navigate to the build folder, then run these commands
- rm -rf CMakeCache.txt CMakeFiles VulkanTest.exe
- cmake .. -G "MinGW Makefiles" -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe     -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe
- mingw32-make

Big thanks to https://vulkan-tutorial.com/



Topics explored:
- Vulkan overview - what it takes to draw the first triangle
- Environment setup - making sure Vulkan, glfw, and glm are properly included
- Basic program setup - initialize, main loop, cleanup
- Window setup - use GLFW to create a space to draw in
- Vulkan instance - create a basic instance of Vulkan with extensions from GLFW
- GLFW extensions - more extension options and info about extension support
- Validation layers - how to get debug output when things go wrong using layers built in to the Vulkan SDK
- Physical devices - find GPUs and check their computing and Vulkan support capabilities, then pick one to use
- Queue families - check device support for different command subsets
- Logical devices - set up the interface for interacting with the chosen physical device
- WSI - Windows System Integration extensions - present rendered images
