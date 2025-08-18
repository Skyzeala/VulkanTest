Learning how to use Vulkan one step at a time.

"Even the smallest step's a leap toward your dream" Linus from the Stars - It's Alright, It's Okay

The goal with the build process was to easily build on other systems as well as have a portable binary.

Build steps after environment is set up:
- open mingw64.exe and navigate to the build folder, then run these commands
- rm -rf CMakeCache.txt CMakeFiles VulkanTest.exe
- cmake .. -G "MinGW Makefiles" -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe     -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe
- mingw32-make

Big thanks to https://vulkan-tutorial.com/



Topics explored so far:
- Vulkan overview - what it takes to draw the first triangle
- Environment setup - making sure Vulkan, GLFW, and GLM are properly included
- Basic program setup - initialize, main loop, cleanup
- Window setup - use GLFW to create a space to draw in
- Vulkan instance - create a basic instance of Vulkan with extensions from GLFW
- GLFW extensions - more extension options and info about extension support
- Validation layers - how to get debug output when things go wrong using layers built in to the Vulkan SDK
- Physical devices - find GPUs and check their computing and Vulkan support capabilities, then pick one to use
- Queue families - check device support for different command subsets
- Logical devices - set up the interface for interacting with the chosen physical device
- WSI & Surfaces - Windows System Integration extensions - present rendered images
- Swap chain - setup of swap chain surfaces, access to swap chain elements, and prepping for drawing
- Graphics pipeline - review of graphics pipeline elements and shaders and how it works in Vulkan
- GLSL - review of GLSL basics for vertex and fragment shaders and conversion to SPIR-V
- Viewports and scissors - region of the frame buffer to render output to
- Rasterizer - convert geometry into fragments
- Color blending - alpha blending of fragment shader return value and existing color in the frame buffer
- Render Passes
