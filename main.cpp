#define GLFW_STATIC
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>





#include <iostream>
#include <stdexcept>
#include <cstdlib>


class HelloTriangleVulkan {
public:
    void run() { 
        initWindow(); 
        initVulkan(); 
        mainLoop(); 
        cleanup(); 
    }

private:
    GLFWwindow* window;

    void initWindow() {
        glfwInit(); //first thing to do

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //tells glfw not to initialize opengl context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); //turn off window resizing, special care? did glew have that?
        
        window = glfwCreateWindow(1000, 800, "Vulkan", nullptr, nullptr);
    
    }

    void initVulkan() {

    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};






int main() {
    HelloTriangleVulkan app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;

}
