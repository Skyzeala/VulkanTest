#THIS WILL ONLY WORK IN THE MSYS2 CLI

#open mingw64.exe and navigate to the build folder, then run these commands
#rm -rf CMakeCache.txt CMakeFiles
#cmake .. -G "MinGW Makefiles" -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe
#mingw32-make
cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER C:/msys64/mingw64/bin/g++.exe)
set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin/gcc.exe")

set(CMAKE_PREFIX_PATH "C:/msys64/mingw64")

#set(CMAKE_BUILD_TYPE Debug) #you can turn this on to generate debug exe

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -static-libgcc -static-libstdc++ -pthread")
add_definitions(-DGLFW_STATIC)

project(VulkanTest VERSION 0.1.0 LANGUAGES CXX)

add_executable(VulkanTest main.cpp)

#pthread
find_package(Threads REQUIRED)
target_link_libraries(VulkanTest PRIVATE ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(VulkanTest PRIVATE C:/msys64/mingw64/lib/libwinpthread.a)

#Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(VulkanTest PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(VulkanTest PRIVATE Vulkan::Vulkan)

#GLFW
find_package(glfw3 REQUIRED)
#target_link_libraries(VulkanTest PRIVATE glfw) #this doesn't pull the static library
target_include_directories(VulkanTest PRIVATE C:/msys64/mingw64/include) #also includes GLM
target_link_libraries(VulkanTest PRIVATE C:/msys64/mingw64/lib/libglfw3.a)


#include(FetchContent)

#FetchContent_Declare(
  #glm
  #GIT_REPOSITORY https://github.com/g-truc/glm.git
  #GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e # Use a specific commit or tag
#)
#FetchContent_MakeAvailable(glm)

#------set(glfw3_DIR "C:/msys64/mingw64/lib/cmake/glfw3")
#set(Vulkan_DIR "C:/msys64/mingw64/lib/cmake/VulkanLoader")
#set(glm_DIR "C:/msys64/mingw64/lib/cmake/uhhhhh")


#------find_package(glfw3 REQUIRED)
#target_include_directories(Vulkan PRIVATE "C:/msys64/mingw64/include/GLFW")


#------target_link_libraries(VulkanTest Vulkan)

#------target_link_libraries(VulkanTest glfw) #not glfw3?

#------target_link_libraries(VulkanTest glm)
