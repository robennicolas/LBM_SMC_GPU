#include <GL/glew.h>      
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <iostream>
#include "../include/lbm/lbm_gpu.h"
#include "../renderer/render_gpu.h"

Display::Display(int w, int h) : width_(w), height_(h) {


    speed_host_ = (float*)malloc(width_ * height_ * sizeof(float));
    bar_host_   = (float*)malloc(width_ * height_ * sizeof(float));

    // Note: CUDA/OpenGL interop via PBO was implemented but is not supported
    // under WSL2 (cudaGraphicsGLRegisterBuffer returns OS_CALL_FAILED).
    // The code below uses CPU colormap + cudaMemcpy. For native Linux/Windows,
    // you could re‑enable PBO by uncommenting the block in render().
    // See the "Future Work" section in README.

    if (!glfwInit()) {
        std::cerr << "Erreur GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    window_ = glfwCreateWindow(w, h, "LBM Simulation", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Erreur fenetre" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }


    glfwMakeContextCurrent(window_);
    glewExperimental = GL_TRUE;
    cudaSetDevice(0);  
    cudaFree(0);       
    glewInit();



    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);



    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cerr << "CUDA device: " << prop.name << std::endl;

}





void Display::render(float* d_speed, float* d_bar) {

    // Copy GPU data to host
    cudaMemcpy(speed_host_, d_speed, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bar_host_,   d_bar,   width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);


    // Build RGBA pixels on CPU
    std::vector<unsigned char> pixels(width_ * height_ * 4);
    float u0 = 0.1f;
    for (int i = 0; i < width_ * height_; i++) {
        int p = i * 4;
        if (bar_host_[i] > 0.5f) {
            pixels[p] = 50; pixels[p+1] = 50; pixels[p+2] = 50; pixels[p+3] = 255;
            continue;
        }
        float t = fmaxf(0.0f, fminf(speed_host_[i] / u0, 1.0f));
        pixels[p+0] = (unsigned char)(fminf(2.0f * t, 1.0f) * 255);
        pixels[p+1] = (unsigned char)(fminf(2.0f * t, 2.0f - 2.0f * t) * 255);
        pixels[p+2] = (unsigned char)(fmaxf(1.0f - 2.0f * t, 0.0f) * 255);
        pixels[p+3] = 255;
    }




    // Upload to GPU texture and draw
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());



    glBegin(GL_QUADS);
        glTexCoord2f(0,0); glVertex2f(-1,-1);
        glTexCoord2f(1,0); glVertex2f( 1,-1);
        glTexCoord2f(1,1); glVertex2f( 1, 1);
        glTexCoord2f(0,1); glVertex2f(-1, 1);
    glEnd();


    
    glfwSwapBuffers(window_);
    glfwPollEvents();
}






bool Display::isOpen() {
    return !glfwWindowShouldClose(window_);
}





Display::~Display() {
    glDeleteTextures(1, &texture_);
    glfwDestroyWindow(window_);
    glfwTerminate();
    free(speed_host_);
    free(bar_host_);
}