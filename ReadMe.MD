Learning how to use vulkan one step at a time

The goal with the build process was to easily build on other systems as well as have a portable binary

Build steps:
    open mingw64.exe and navigate to the build folder, then run these commands
    rm -rf CMakeCache.txt CMakeFiles
    cmake .. -G "MinGW Makefiles" -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe
    mingw32-make

Big thanks to https://vulkan-tutorial.com/